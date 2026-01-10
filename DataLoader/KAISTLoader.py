import cv2
import numpy as np
import glob
import logging
import pandas as pd
from pathlib import Path
from pandas._libs.tslibs import timestamps
from tqdm import tqdm

from utils.PinholeCamera import PinholeCamera


class ComplexUrbanDatasetLoader(object):
    default_config = {
        "root_path": "/run/media/aki/OhShit/dataset_hdd/urban27",
        "sequence": "urban27",
        "start": 0,
        "camera": "stereo_left",
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Complex Urban Dataset config: ")
        logging.info(self.config)

        self.dataset_path = Path(self.config["root_path"])
        self.sequence_path = self.dataset_path / self.config["sequence"]

        if self.config["camera"] == "stereo_left":
            self.img_folder = self.sequence_path / "stereo_left"
        else:
            self.img_folder = self.sequence_path / "stereo_right"
        # Hardcoded for urban27
        if PinholeCamera is not None:
            self.cam = PinholeCamera(
                width=1280,
                height=560,
                fx=816.9037899,
                fy=811.5680383,
                cx=608.5072628,
                cy=263.4759976,
                k1=-0.0561430,
                k2=0.1395256,
                p1=-0.0012156,
                p2=-0.0009728,
                k3=-0.0808782,
            )

        # Ground truth
        self.gt_poses = []

        # Load timestamps file
        img_files = sorted(list(self.img_folder.glob("*.png")))
        self.timestamps = np.array([int(p.stem) for p in img_files])

        # Load vrs gps file
        vrs_gps_path = self.sequence_path / "sensor_data/vrs_gps.csv"
        gps_df = pd.read_csv(vrs_gps_path, header=None)

        # Default values for urban27
        gps_data = gps_df.values

        gps_timestamps = gps_data[:, 0]

        # For each image get pose based in the timestamp
        for ts in self.timestamps:
            # Encontrar índice do timestamp mais próximo
            idx = np.argmin(np.abs(gps_timestamps - ts))
            row = gps_data[idx]

            # Build 4x4 pose matrix
            pose = np.eye(4)

            pose[0, 3] = row[3]  # UTM X -> Camera X (Right)
            pose[1, 3] = -row[5]  # Altitude -> Camera Y (Down) - Note the negative sign
            pose[2, 3] = row[4]  # UTM Y -> Camera Z (Forward)

            # Rotação = Heading
            # Verifica heading_valid (coluna 13)
            if row[12] == 1:
                heading = np.radians(row[13])
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)

                pose[:3, :3] = np.array(
                    [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]]
                )

            self.gt_poses.append(pose[:4, :4])

        all_poses = np.array(self.gt_poses)

        if len(all_poses) > 0:
            # 2. Get the very first pose as the 'Origin'
            first_pose = all_poses[0]
            first_pose_inv = np.linalg.inv(first_pose)

            self.gt_poses = []
            for i in range(len(all_poses)):
                # 3. Calculate relative movement from start
                rel_pose = first_pose_inv @ all_poses[i]

                # 4. Convert to KITTI format: Keep only 3 rows and 4 columns
                # KITTI uses 3x4: [R | t]
                self.gt_poses.append(rel_pose[:3, :4])

                self.img_id = self.config["start"]

        search_path = str(self.img_folder / "*.png")
        self.img_files = sorted(glob.glob(search_path))
        self.img_N = len(self.img_files)

    def get_cur_pose(self):
        """Retorna a pose relativa ao primeiro frame no formato KITTI (3x4)"""
        if 0 <= self.img_id - 1 < len(self.gt_poses):
            return self.gt_poses[self.img_id - 1] - self.gt_poses[0]

        return np.eye(4)[:3, :]  # Fallbak

    def __getitem__(self, item):
        """Acesso aleatório por índice (similar ao KITTI Loader)"""
        if item >= len(self.img_files):
            raise IndexError("Index out of bounds")

        file_name = self.img_files[item]
        img = cv2.imread(file_name)  # Retorna BGR
        # Se precisar de grayscale explicitamente:
        # img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        return img

    def __iter__(self):
        """Reseta o iterador"""
        return self

    def __next__(self):
        """Lógica de iteração principal"""
        if self.img_id < self.img_N:
            file_name = self.img_files[self.img_id]
            img = cv2.imread(file_name)

            self.img_id += 1
            return img

        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
