import cv2
import numpy as np
import glob
import logging
import pandas as pd
from pathlib import Path

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

        # Raw Intrinsics (Input)
        # Camera matrix without transformations
        self.K_raw = np.array(
            [
                [816.9037899, 0.505101667, 608.5072628],
                [0.0, 811.5680383, 263.4759976],
                [0.0, 0.0, 1.0],
            ]
        )
        # Distortion Coefficients
        self.D = np.array([-0.056143, 0.139526, -0.001216, -0.000973, -0.080878])
        # Rectification Matrix
        self.R = np.array(
            [
                [0.999969, 0.000362, -0.007812],
                [-0.000344, 0.999997, 0.002300],
                [0.007813, -0.002298, 0.999967],
            ]
        )
        # Projection Matrix (Output - This is what your VO will use)
        self.P = np.array(
            [
                [775.372356, 0.0, 619.473091, 0.0],
                [0.0, 775.372356, 257.180490, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )

        # 3. Pre-compute the Undistortion/Rectification Maps
        # This is the "Blueprint" for fixing the images. We calculate it ONCE.
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.D,
            self.R,
            self.P[:3, :3],  # We only need the 3x3 part of P
            (1280, 560),  # Image Size
            cv2.CV_32F,
        )

        # 4. Initialize Camera Object for the VO
        # Note: We use the values from P (Projected), NOT K_raw!
        if PinholeCamera is not None:
            self.cam = PinholeCamera(
                width=1280,
                height=560,
                fx=self.P[0, 0],  # 775.37...
                fy=self.P[1, 1],  # 775.37...
                cx=self.P[0, 2],  # 619.47...
                cy=self.P[1, 2],  # 257.18...
                k1=0,
                k2=0,
                p1=0,
                p2=0,
                k3=0,  # Distortion is now 0!
            )
        # Ground truth
        self.gt_poses = []

        # Load timestamps file
        img_files = sorted(list(self.img_folder.glob("*.png")))
        self.timestamps = np.array([int(p.stem) for p in img_files])

        # Load vrs gps file
        vrs_gps_path = self.sequence_path / "sensor_data/vrs_gps.csv"
        gps_df = pd.read_csv(vrs_gps_path, header=None)
        gps_df = gps_df.apply(pd.to_numeric, errors="coerce")
        gps_data = gps_df.values

        # Extrair dados brutos do GPS
        gps_ts_raw = gps_data[:, 0]  # Timestamps do GPS
        gps_x_raw = gps_data[:, 3]  # UTM X
        gps_y_raw = gps_data[:, 4]  # UTM Y
        gps_z_raw = gps_data[:, 5]  # Altitude

        # Timestamps das Imagens (já carregados anteriormente)
        cam_ts = self.timestamps

        # --- CORREÇÃO: INTERPOLAÇÃO ---
        # Cria posições virtuais de GPS sincronizadas com cada frame da câmera
        interp_x = np.interp(cam_ts, gps_ts_raw, gps_x_raw)
        interp_y = np.interp(cam_ts, gps_ts_raw, gps_y_raw)
        interp_z = np.interp(cam_ts, gps_ts_raw, gps_z_raw)

        self.gt_poses = []

        # Agora iteramos pelos índices, pois já temos os dados sincronizados
        for i in range(len(cam_ts)):
            pose = np.eye(4)

            # Usamos os valores interpolados
            # Nota: Mantive a conversão de eixos que você usava (X->X, Z->Y, Y->Z)
            # Confirme se para o KAIST isso ainda faz sentido visualmente
            pose[0, 3] = interp_x[i]  # Câmera X
            pose[1, 3] = -interp_z[i]  # Câmera Y (Altitude invertida)
            pose[2, 3] = interp_y[i]  # Câmera Z (Frente)

            # --- TRATAMENTO DE ROTAÇÃO (HEADING) ---
            # Rotação é mais chato interpolar linearmente por causa do salto 360->0
            # Vamos pegar o mais próximo APENAS para a rotação, ou interpolar com cuidado
            # Para simplificar, vou usar o 'nearest' para rotação, mas o 'interp' para posição
            idx_nearest = np.argmin(np.abs(gps_ts_raw - cam_ts[i]))
            row_raw = gps_data[idx_nearest]

            if row_raw[12] == 1:  # heading_valid
                heading = np.radians(row_raw[13])
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)

                # Ajuste a matriz de rotação conforme a orientação do seu VO
                pose[:3, :3] = np.array(
                    [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]]
                )

            self.gt_poses.append(pose)

        print(self.gt_poses)
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
            return self.gt_poses[self.img_id - 1]

        return np.eye(4)[:3, :]  # Fallbak

    def __getitem__(self, item):
        if item >= len(self.img_files):
            raise IndexError("Index out of bounds")
        file_name = self.img_files[item]

        img = cv2.imread(file_name)  # Retorna BGR
        # Se precisar de grayscale explicitamente:
        # img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        # img_rectified = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)

        return img

    def __iter__(self):
        """Reseta o iterador"""
        return self

    def __next__(self):
        """Lógica de iteração principal"""
        if self.img_id < self.img_N:
            file_name = self.img_files[self.img_id]
            img = cv2.imread(file_name)

            # This makes the image match the Projection Matrix
            img_rectified = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
            self.img_id += 1
            return img_rectified

        raise StopIteration()

    def __len__(self):
        return self.img_N - self.config["start"]
