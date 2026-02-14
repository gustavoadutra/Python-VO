from pathlib import Path
import logging
import numpy as np
import torch
import cv2
from utils.tools import *
from Detectors.accelerated_features.modules.xfeat import XFeat


class XfeatDetector(object):

    def __init__(self, config={}):
        self.default_config = {
            "cuda": True,
            "weights": None,
            "top_k": 4096,
            "detection_threshold": 0.05,
        }

        self.config = {**self.default_config, **config}
        logging.info("XFeat detector config:")
        logging.info(self.config)

        self.device = "cuda" if torch.cuda.is_available() and self.config["cuda"] else "cpu"

        logging.info("creating XFeat detector...")
        print(self.config.get("weights"))
        self.xfeat = XFeat(
            top_k=self.config.get("top_k"),
            detection_threshold=self.config.get("detection_threshold"),
        ).to(self.device)

    def __call__(self, image):
        logging.debug("detecting keypoints with XFeat...")

        # XFeat accepts either numpy arrays or torch tensors; ensure we pass numpy HxW or HxWxC
        # convert to grayscale numpy if needed (already done above)
        img_in = image

        # Image treatment before passing to XFeat
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
        image_tensor: torch.Tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)  # (H, W, C) -> (1, C, H, W)
        # Run detection (returns list of dicts for batch) - pass a single image
        
        out = self.xfeat.detectAndCompute(image_tensor)
        # detectAndCompute returns a list (one entry per batch)
        pred = out[0]

        # pred contains torch Tensors for 'keypoints' (N,2), 'scores' (N,), 'descriptors' (N, C)
        kps = pred["keypoints"].cpu().detach().numpy()
        scores = pred["scores"].cpu().detach().numpy() if "scores" in pred else np.ones(len(kps))
        descs = pred["descriptors"].cpu().detach().numpy()
        ret_dict = {
            "image_size": np.array([image.shape[1], image.shape[0]]),  # [width, height]
            "torch": pred,
            "keypoints": kps,
            "scores": scores,
            "descriptors": descs,
        }

        return ret_dict


if __name__ == "__main__":
    img = cv2.imread("../test_imgs/sequences/00/image_0/000000.png")

    detector = XfeatDetector()
    kptdescs = detector(img)

    img = plot_keypoints(img, kptdescs["keypoints"], kptdescs["scores"])
    cv2.imshow("Xfeat", img)
    cv2.waitKey()
