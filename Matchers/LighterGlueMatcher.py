import logging
from pathlib import Path
import torch
import numpy as np
from utils.tools import *
from Detectors.accelerated_features.modules.lighterglue import LighterGlue


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

    def __call__(self, data):

        # Prepare data for LighterGlue
        data = {
            'keypoints0': torch.from_numpy(data['ref']['keypoints']).unsqueeze(0).to(self.device).float(),
            'descriptors0': torch.from_numpy(data['ref']['descriptors']).unsqueeze(0).to(self.device).float(),
            'image_size0': torch.tensor(data['ref']['image_size']).unsqueeze(0).to(self.device).long(),
            'keypoints1': torch.from_numpy(data['cur']['keypoints']).unsqueeze(0).to(self.device).float(),
            'descriptors1': torch.from_numpy(data['cur']['descriptors']).unsqueeze(0).to(self.device).float(),
            'image_size1': torch.tensor(data['cur']['image_size']).unsqueeze(0).to(self.device).long(),
        }

        # Run lighterglue
        logging.debug("matching keypoints with LighterGlue...")
        out = self.lighterglue(data, min_conf=self.config.get("min_conf", 0.1))

        # Extract matches indices
        idxs = out["matches"][0]

        # Convert to numpy keypoint arrays
        kp_ref = data["keypoints0"][0].cpu().numpy()
        kp_cur = data["keypoints1"][0].cpu().numpy()

        if idxs.numel() == 0:
            return {"ref_keypoints": np.zeros((0, 2)), "cur_keypoints": np.zeros((0, 2)), "match_score": np.array([])}

        idxs_np = idxs.cpu().numpy()
        mkpts0 = kp_ref[idxs_np[:, 0]]
        mkpts1 = kp_cur[idxs_np[:, 1]]

        # Extract confidence scores (scores are per match, not per keypoint)
        scores = out.get("scores", None)
        if scores is not None:
            match_scores = scores[0].cpu().numpy()
        else:
            match_scores = np.ones(len(idxs_np))

        return {"ref_keypoints": mkpts0, "cur_keypoints": mkpts1, "match_score": match_scores}


if __name__ == "__main__":
    from DataLoader.SequenceImageLoader import SequenceImageLoader
    from Detectors.XfeatDetector import XfeatDetector

    loader = SequenceImageLoader()
    detector = XfeatDetector({"cuda": 0})
    matcher = LighterGlueMatcher({"cuda": 0})

    kptdescs = {}
    imgs = {}
    for i, img in enumerate(loader):
        imgs["cur"] = img
        kptdescs["cur"] = detector(img)
        if i >= 1:
            matches = matcher(kptdescs)
            img = plot_matches(imgs['ref'], imgs['cur'], matches['ref_keypoints'][0:200], matches['cur_keypoints'][0:200], matches['match_score'][0:200], layout='lr')
            import cv2

            cv2.imshow("track", img)
            if cv2.waitKey() == 27:
                break

        kptdescs["ref"], imgs["ref"] = kptdescs["cur"], imgs["cur"]
