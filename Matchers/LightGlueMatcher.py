import logging
import numpy as np
import torch
import kornia.feature as KF
from utils.tools import *


class LightGlueMatch(object):
    """
    Classe mock para simular a estrutura de um cv2.DMatch.
    Permite que a VisualOdometry acesse .queryIdx e .trainIdx de forma transparente.
    """
    def __init__(self, queryIdx, trainIdx, distance=0.0):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


class LightGlueMatcher(object):
    """
    Wrapper em torno do kornia.feature.LightGlue.

    A API real do LightGlue do kornia seleciona o conjunto de pesos através do
    parâmetro `features` (string), e não através de um caminho de pesos solto.
    As opções relevantes para este pipeline são:
        - "superpoint": usa descritores SuperPoint (dim=256)
        - "sift":       usa descritores SIFT (dim=128) e EXIGE escala e
                         orientação por keypoint (campos 'scales' e 'oris')

    Ou seja, o conjunto de features escolhido aqui precisa bater com o
    extrator de descritor+feature usado a montante (seu código de
    descritor/feature): se você extrai com SuperPoint, use features="superpoint";
    se extrai com SIFT, use features="sift" e garanta que 'scales'/'oris'
    sejam passados em kptdescs.
    """

    default_config = {
        "features": "superpoint",  # "superpoint" ou "sift"
        "cuda": True,
        "min_conf": 0.1,           # mapeado para o filter_threshold do LightGlue
    }

    # Conjuntos de features que exigem escala/orientação por keypoint
    # (mesma lógica usada internamente pelo kornia: conf.add_scale_ori)
    _SCALE_ORI_FEATURES = {"sift", "dog_affnet_hardnet", "doghardnet"}

    def __init__(self, config={}):
        self.config = {**self.default_config, **config}
        logging.info("LightGlue matcher config:")
        logging.info(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        features = self.config["features"]
        valid_features = set(KF.LightGlue.features.keys())
        if features not in valid_features:
            raise ValueError(
                f"features='{features}' inválido. Opções suportadas pelo kornia: {sorted(valid_features)} "
                f"(para este pipeline, use 'superpoint' ou 'sift')."
            )
        self.needs_scale_ori = features in self._SCALE_ORI_FEATURES

        logging.info(f"creating LightGlue matcher (features='{features}')...")
        # filter_threshold é o equivalente, na API real, ao antigo min_conf
        self.lighterglue = KF.LightGlue(features=features, filter_threshold=self.config.get("min_conf", 0.1))
        self.lighterglue.to(self.device)
        self.lighterglue.eval()

        # Armazenamento interno para manter compatibilidade com a chamada dividida da VO
        self.good = []
        self.ret_dict = {}

    def _build_image_dict(self, kdesc):
        """Monta o dicionário 'image0'/'image1' no formato exigido pelo LightGlue real do kornia."""
        img_dict = {
            'keypoints': torch.from_numpy(kdesc['keypoints']).unsqueeze(0).to(self.device).float(),
            'descriptors': torch.from_numpy(kdesc['descriptors']).unsqueeze(0).to(self.device).float(),
            'image_size': torch.tensor(kdesc['image_size']).unsqueeze(0).to(self.device).long(),
        }
        if self.needs_scale_ori:
            if 'scales' not in kdesc or 'oris' not in kdesc:
                raise ValueError(
                    "As features escolhidas (ex: 'sift') exigem 'scales' e 'oris' por keypoint "
                    "em kptdescs — preencha-os no seu extrator de descritor/feature."
                )
            img_dict['scales'] = torch.from_numpy(np.asarray(kdesc['scales'])).unsqueeze(0).to(self.device).float()
            img_dict['oris'] = torch.from_numpy(np.asarray(kdesc['oris'])).unsqueeze(0).to(self.device).float()
        return img_dict

    @torch.no_grad()
    def match(self, kptdescs):
        """
        Executa a correspondência de features usando o LightGlue e popula as estruturas
        esperadas pela classe VisualOdometry.
        """
        # Prepara os dados no formato real esperado pelo kornia.feature.LightGlue:
        # data = {"image0": {...}, "image1": {...}}
        data = {
            'image0': self._build_image_dict(kptdescs['ref']),
            'image1': self._build_image_dict(kptdescs['cur']),
        }

        # Executa a inferência do matcher baseado em redes neurais
        logging.debug("matching keypoints with LightGlue...")
        out = self.lighterglue(data)

        # Extrai os índices dos matches válidos (lista por item de batch; batch=1 aqui)
        idxs = out["matches"][0]

        # Converte os keypoints originais para numpy para fatiamento
        kp_ref = data["image0"]["keypoints"][0].cpu().numpy()
        kp_cur = data["image1"]["keypoints"][0].cpu().numpy()

        if idxs.numel() == 0:
            self.good = []
            self.ret_dict = {"ref_keypoints": np.zeros((0, 2)), "cur_keypoints": np.zeros((0, 2)), "match_score": np.array([])}
            return self.good

        idxs_np = idxs.cpu().numpy()
        mkpts0 = kp_ref[idxs_np[:, 0]]
        mkpts1 = kp_cur[idxs_np[:, 1]]

        # Extrai os scores de confiança (métrica nativa do modelo)
        scores = out.get("scores", None)
        if scores is not None:
            match_scores = scores[0].cpu().numpy()
        else:
            match_scores = np.ones(len(idxs_np))

        # Monta o dicionário de retorno idêntico ao do FrameByFrameMatcher
        self.ret_dict = {
            "ref_keypoints": mkpts0,
            "cur_keypoints": mkpts1,
            "match_score": match_scores
        }

        # Constrói a lista 'good' no formato de lista de listas contendo os índices originais mapeados
        self.good = []
        for i, (idx_ref, idx_cur) in enumerate(idxs_np):
            score = match_scores[i]
            # Encapsula no LightGlueMatch simulando o cv2.DMatch
            # A distância pode ser interpretada inversamente ao score (1.0 - conf)
            match_obj = LightGlueMatch(queryIdx=int(idx_ref), trainIdx=int(idx_cur), distance=float(1.0 - score))
            self.good.append([match_obj])

        return self.good

    def get_good_keypoints(self, kptdescs=None):
        """Retorna o dicionário processado na última chamada do método match."""
        return self.ret_dict

    def __call__(self, data):
        """Mantém compatibilidade com chamadas diretas no formato antigo."""
        self.match(data)
        return self.get_good_keypoints()