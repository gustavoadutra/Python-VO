import cv2
import numpy as np
import logging
from utils.tools import plot_keypoints


class HandcraftDetector(object):
    default_config = {
        "type": "SIFT",
        "ORB": {
            "nfeatures": 1000,
            "scaleFactor": 1.2,
            "nLevels": 8,
            "edgeThreshold": 31,
            "firstLevel": 0,
            "WTA_K": 2,
            "patchSize": 31,
            "fastThreshold": 20,
        },
        "SIFT": {
            "nfeatures": 1000,
            "nOctaveLayers": 3,
            "contrastThreshold": 0.04,
            "edgeThreshold": 10,
            "sigma": 1.6,
        },
    }

    # Fator de magnificação da região de suporte (mesmo "mrSize" usado pelo
    # kornia/kornia_moons para converter keypoints OpenCV em LAFs). Para SIFT
    # o valor padrão de toda a stack kornia é 6.0; para ORB, 1.0 (sem magnificação).
    _MR_SIZE = {"SIFT": 6.0, "ORB": 1.0}

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Handcraft detector config: ")
        logging.info(self.config)

        if self.config["type"] == "ORB":
            logging.info("creating ORB detector...")
            self.det = cv2.ORB_create(
                nfeatures=self.config["ORB"]["nfeatures"],
                scaleFactor=self.config["ORB"]["scaleFactor"],
                nlevels=self.config["ORB"]["nLevels"],
                edgeThreshold=self.config["ORB"]["edgeThreshold"],
                firstLevel=self.config["ORB"]["firstLevel"],
                WTA_K=self.config["ORB"]["WTA_K"],
                patchSize=self.config["ORB"]["patchSize"],
                fastThreshold=self.config["ORB"]["fastThreshold"],
            )
        elif self.config["type"] == "SIFT":
            logging.info("creating SIFT detector...")
            self.det = cv2.SIFT_create(
                nfeatures=self.config["SIFT"]["nfeatures"],
                nOctaveLayers=self.config["SIFT"]["nOctaveLayers"],
                contrastThreshold=self.config["SIFT"]["contrastThreshold"],
                edgeThreshold=self.config["SIFT"]["edgeThreshold"],
                sigma=self.config["SIFT"]["sigma"],
            )
        else:
            raise NotImplementedError(
                f"Not implement for feature type: {self.feature_type}"
            )

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        logging.debug("keypoint detecting and computing...")
        kpts_cv, desc = self.det.detectAndCompute(image, None)

        # Normalização L2 obrigatória para o LightGlue
        # para o flannmatch nao faz diferenca
        if self.config["type"] == "SIFT":
            norm = np.linalg.norm(desc, axis=1, keepdims=True)
            desc = np.divide(desc, norm, out=np.zeros_like(desc), where=norm!=0)

        mr_size = self._MR_SIZE.get(self.config["type"], 1.0)

        kpts = np.zeros((len(kpts_cv), 2))
        scores = np.zeros((len(kpts_cv)))
        scales = np.zeros((len(kpts_cv)))
        oris = np.zeros((len(kpts_cv)))
        for i, p in enumerate(kpts_cv):
            kpts[i, 0] = p.pt[0]
            kpts[i, 1] = p.pt[1]
            scores[i] = p.response
            scales[i] = mr_size * p.size
            ori = np.deg2rad(-p.angle)
            if ori < 0:
                ori += 2.0 * np.pi
            oris[i] = ori

        return {
            "image_size": np.array([image.shape[1], image.shape[0]]),
            "keypoints": kpts,
            "scores": scores,
            "descriptors": desc,
            "scales": scales,
            "oris": oris,
        }


