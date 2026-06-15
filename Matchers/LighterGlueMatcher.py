import logging
from pathlib import Path
import torch
import numpy as np
from utils.tools import *
from Detectors.accelerated_features.modules.lighterglue import LighterGlue


class LighterGlueMatch(object):
    """
    Classe mock para simular a estrutura de um cv2.DMatch.
    Permite que a VisualOdometry acesse .queryIdx e .trainIdx de forma transparente.
    """
    def __init__(self, queryIdx, trainIdx, distance=0.0):
        self.queryIdx = queryIdx
        self.trainIdx = trainIdx
        self.distance = distance


class LighterGlueMatcher(object):
    default_config = {
        "weights": None,
        "cuda": True,
        "min_conf": 0.1,
    }

    def __init__(self, config={}):
        self.config = {**self.default_config, **config}
        logging.info("LighterGlue matcher config:")
        logging.info(self.config)

        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        logging.info("creating LighterGlue matcher...")
        self.lighterglue = LighterGlue(weights=self.config.get("weights"))
        self.lighterglue.to(self.device)
        
        # Armazenamento interno para manter compatibilidade com a chamada dividida da VO
        self.good = []
        self.ret_dict = {}

    def match(self, kptdescs):
        """
        Executa a correspondência de features usando o LighterGlue e popula as estruturas
        esperadas pela classe VisualOdometry.
        """
        # Prepara os dados no formato que o LighterGlue espera
        data = {
            'keypoints0': torch.from_numpy(kptdescs['ref']['keypoints']).unsqueeze(0).to(self.device).float(),
            'descriptors0': torch.from_numpy(kptdescs['ref']['descriptors']).unsqueeze(0).to(self.device).float(),
            'image_size0': torch.tensor(kptdescs['ref']['image_size']).unsqueeze(0).to(self.device).long(),
            'keypoints1': torch.from_numpy(kptdescs['cur']['keypoints']).unsqueeze(0).to(self.device).float(),
            'descriptors1': torch.from_numpy(kptdescs['cur']['descriptors']).unsqueeze(0).to(self.device).float(),
            'image_size1': torch.tensor(kptdescs['cur']['image_size']).unsqueeze(0).to(self.device).long(),
        }

        # Executa a inferência do matcher baseado em redes neurais
        logging.debug("matching keypoints with LighterGlue...")
        out = self.lighterglue(data, min_conf=self.config.get("min_conf", 0.1))

        # Extrai os índices dos matches válidos
        idxs = out["matches"][0]

        # Converte os keypoints originais para numpy para fatiamento
        kp_ref = data["keypoints0"][0].cpu().numpy()
        kp_cur = data["keypoints1"][0].cpu().numpy()

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
            # Encapsula no LighterGlueMatch simulando o cv2.DMatch
            # A distância pode ser interpretada inversamente ao score (1.0 - conf)
            match_obj = LighterGlueMatch(queryIdx=int(idx_ref), trainIdx=int(idx_cur), distance=float(1.0 - score))
            self.good.append([match_obj])

        return self.good

    def get_good_keypoints(self, kptdescs=None):
        """Retorna o dicionário processado na última chamada do método match."""
        return self.ret_dict

    def __call__(self, data):
        """Mantém compatibilidade com chamadas diretas no formato antigo."""
        self.match(data)
        return self.get_good_keypoints()
