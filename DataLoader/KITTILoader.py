import cv2
import numpy as np
import glob
import logging
import os

from utils.PinholeCamera import PinholeCamera

class KITTILoader(object):
    default_config = {"root_path": "../test_imgs", "sequence": "00", "start": 0}

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("KITTI Dataset config: ")
        logging.info(self.config)

        if self.config["sequence"] in ["00", "01", "02"]:
            self.cam = PinholeCamera(
                1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157
            )
        elif self.config["sequence"] in ["03"]:
            self.cam = PinholeCamera(
                1242.0, 375.0, 721.5377, 721.5377, 609.5593, 172.854
            )
        elif self.config["sequence"] in ["04", "05", "06", "07", "08", "09", "10"]:
            self.cam = PinholeCamera(
                1226.0, 370.0, 707.0912, 707.0912, 601.8873, 183.1104
            )
        else:
            raise ValueError(f"Unknown sequence number: {self.config['sequence']}")

        # 1. Carrega as poses (Ground Truth)
        self.pose_path = os.path.join(
            self.config["root_path"], "poses", self.config["sequence"] + ".txt"
        )
        self.gt_poses = []
        with open(self.pose_path) as f:
            lines = f.readlines()
            for line in lines:
                ss = line.strip().split()
                pose = np.zeros((1, len(ss)))
                for i in range(len(ss)):
                    pose[0, i] = float(ss[i])
                pose.resize([3, 4])
                self.gt_poses.append(pose)

        # Carrega os Timestamps do arquivo times.txt
        self.times_path = os.path.join(
            self.config["root_path"], "sequences", self.config["sequence"], "times.txt"
        )
        self.times = []
        if os.path.exists(self.times_path):
            with open(self.times_path) as f:
                self.times = [float(line.strip()) for line in f.readlines()]
        else:
            logging.warning(f"Arquivo times.txt não encontrado em {self.times_path}! Usando índices falsos.")
            # Fallback caso o arquivo não exista: usa o índice como segundo (0.0, 1.0, 2.0...)
            self.times = [float(i) for i in range(len(self.gt_poses))]

        # Verifica quantidade de imagens
        self.img_id = self.config["start"]
        img_pattern = os.path.join(
            self.config["root_path"], "sequences", self.config["sequence"], "image_0", "*.png"
        )
        self.img_N = len(glob.glob(pathname=img_pattern))

    def get_cur_pose(self):
        return self.gt_poses[self.img_id - 1]

    def __getitem__(self, item):
        file_name = os.path.join(
            self.config["root_path"], "sequences", self.config["sequence"], "image_0", str(item).zfill(6) + ".png"
        )
        img = cv2.imread(file_name)
        return img

    def __iter__(self):
        return self

    def __next__(self):
        if self.img_id < self.img_N:
            file_name = os.path.join(
                self.config["root_path"], "sequences", self.config["sequence"], "image_0", str(self.img_id).zfill(6) + ".png"
            )
            img = cv2.imread(file_name)
            self.img_id += 1
            return img
        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]