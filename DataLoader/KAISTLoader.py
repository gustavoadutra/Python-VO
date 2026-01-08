import cv2
import numpy as np
<<<<<<< HEAD
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Iterator
import matplotlib.pyplot as plt

from utils.PinholeCamera import PinholeCamera


class ComplexUrbanDatasetLoader:
    # DataLoader for Complex Urban Dataset

=======
import glob
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Assumindo que você tem este arquivo conforme seu exemplo original
# Caso não tenha, o código funcionará, mas self.cam não será instanciado
try:
    from utils.PinholeCamera import PinholeCamera
except ImportError:
    logging.warning("utils.PinholeCamera not found. self.cam will be None.")
    PinholeCamera = None


class ComplexUrbanDatasetLoader(object):
    # Configuração padrão alinhada com o estilo KITTI
>>>>>>> d9d7c5c (Added files)
    default_config = {
        "root_path": "/run/media/aki/OhShit/dataset_hdd/urban27",
        "sequence": "urban27",
        "start": 0,
<<<<<<< HEAD
        "camera": "stereo_left",
    }

    def __init__(self, config: Dict = {}):
        self.config = {**self.default_config, **config}

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Initializing Complex Urban Dataset with config: {self.config}")

        self._setup_paths()
        self._init_calibration()
        self._load_data()

        # Iteration state
        self.img_id = self.config["start"]
        self.img_N = len(self.img_files)

    def _setup_paths(self):
        self.dataset_path = Path(self.config["root_path"])
        self.sequence_path = self.dataset_path / self.config["sequence"]

        target_cam = self.config["camera"]
        if target_cam not in ["stereo_left", "stereo_right"]:
            raise ValueError(f"Invalid camera selection: {target_cam}")

        self.img_folder = self.sequence_path / target_cam

    def _init_calibration(self):
        calib_file = self.sequence_path / "calibration" / "left.yaml"

        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found at: {calib_file}")

        fs = cv2.FileStorage(str(calib_file), cv2.FILE_STORAGE_READ)

        if not fs.isOpened():
            raise ValueError(f"Could not open calibration file: {calib_file}")

        self.width = int(fs.getNode("image_width").real())
        self.height = int(fs.getNode("image_height").real())
        self.K_raw = fs.getNode("camera_matrix").mat()
        self.D = fs.getNode("distortion_coefficients").mat()
        self.R = fs.getNode("rectification_matrix").mat()
        self.P = fs.getNode("projection_matrix").mat()

        fs.release()

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.D,
            self.R,
            self.P[:3, :3],
            (self.width, self.height),
            cv2.CV_32F,
        )

        if PinholeCamera is not None:
            self.cam = PinholeCamera(
                width=self.width,
                height=self.height,
                fx=self.P[0, 0],
                fy=self.P[1, 1],
                cx=self.P[0, 2],
                cy=self.P[1, 2],
            )

    def _load_data(self):
        # Load timestamps from CSV or fallback to image folder
        stamp_file_path = self.sequence_path / "sensor_data/stereo_stamp.csv"
        if stamp_file_path.exists():
            df_stamps = pd.read_csv(stamp_file_path, header=None)
            self.timestamps_vo = pd.to_numeric(df_stamps.iloc[:, 0]).values.astype(
                np.int64
            )
        else:
            print(f"Warning: {stamp_file_path} not found. Loading timestamps from images.")
            image_paths = list(self.img_folder.glob("*.png"))
            if not image_paths:
                raise FileNotFoundError(f"No images in: {self.img_folder}")
            try:
                ts_list = [int(p.stem) for p in image_paths]
            except ValueError:
                raise ValueError("Image filenames must be numeric.")
            self.timestamps_vo = np.array(sorted(ts_list), dtype=np.int64)

        # Convert nanoseconds to seconds for encoder sync
        self.times = self.timestamps_vo / 1e9

        self.img_files = []
        for ts in self.timestamps_vo:
            img_p = self.img_folder / f"{ts}.png"
            self.img_files.append(str(img_p))

        logging.info(f"Loaded {len(self.img_files)} image entries.")

        # Load GPS for ground truth poses
        vrs_gps_path = self.sequence_path / "sensor_data/vrs_gps.csv"
        if not vrs_gps_path.exists():
            logging.error("GPS file not found.")
            self.gt_poses = []
            return

        gps_df = pd.read_csv(vrs_gps_path, header=None, sep=",")
        needed_cols = [0, 3, 4, 5, 12, 13]
        for col in needed_cols:
            gps_df[col] = pd.to_numeric(gps_df[col])

        gps_data = gps_df.values
        self.gps_ts_raw = gps_data[:, 0].astype(np.float64)
        gps_x_raw = gps_data[:, 3].astype(np.float64)
        gps_y_raw = gps_data[:, 4].astype(np.float64)
        gps_z_raw = gps_data[:, 5].astype(np.float64)

        if len(self.gps_ts_raw) == 0:
            self.gt_poses = []
            return
        
        target_ts = self.timestamps_vo.astype(np.float64)
        interp_x = np.interp(target_ts, self.gps_ts_raw, gps_x_raw)
        interp_y = np.interp(target_ts, self.gps_ts_raw, gps_y_raw)
        interp_z = np.interp(target_ts, self.gps_ts_raw, gps_z_raw)

        raw_poses = []
        for i in range(len(self.timestamps_vo)):
            pose = np.eye(4)
            pose[0, 3] = interp_x[i]
            pose[1, 3] = interp_z[i]
            pose[2, 3] = interp_y[i]

            idx_nearest = np.argmin(np.abs(self.gps_ts_raw - target_ts[i]))
            row_raw = gps_data[idx_nearest]
            if row_raw[12] == 1:  # Valid heading
                heading = np.radians(row_raw[13])
                cos_h, sin_h = np.cos(heading), np.sin(heading)
                pose[:3, :3] = np.array(
                    [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]]
                )
            raw_poses.append(pose)

        # Convert to relative poses from first frame
        self.gt_poses = []
        if len(raw_poses) > 0:
            first_pose_inv = np.linalg.inv(raw_poses[0])
            for p in raw_poses:
                self.gt_poses.append((first_pose_inv @ p)[:3, :4])

    def get_cur_pose(self) -> np.ndarray:
        idx = self.img_id - 1
        if 0 <= idx < len(self.gt_poses):
            return self.gt_poses[idx]
        return np.eye(4)[:3, :]

    def __len__(self) -> int:
        return self.img_N - self.config["start"]

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> np.ndarray:
        if self.img_id < self.img_N:
            file_path = self.img_files[self.img_id]

            if not Path(file_path).exists():
                logging.warning(f"Image file missing: {file_path}")
                img = np.zeros((560, 1280, 3), dtype=np.uint8)
            else:
                img = cv2.imread(file_path)
                if img is None:
                    img = np.zeros((560, 1280, 3), dtype=np.uint8)

            img_rectified = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR)
            # img_rectified = cv2.rotate(img, cv2.ROTATE_180)
            self.img_id += 1
            return img_rectified

        raise StopIteration()

    def __getitem__(self, index: int) -> np.ndarray:
        if index >= len(self.img_files):
            raise IndexError("Index out of bounds")
        file_path = self.img_files[index]
        return cv2.imread(file_path)
=======
        "camera": "stereo_left",  # Opção extra para escolher câmera esquerda/direita
    }

    def __init__(self, config={}):
        self.config = self.default_config
        self.config = {**self.config, **config}
        logging.info("Complex Urban Dataset config: ")
        logging.info(self.config)

        # Configuração de caminhos usando pathlib para robustez, mas convertendo para string onde necessário
        self.dataset_path = Path(self.config["root_path"])
        self.sequence_path = self.dataset_path / self.config["sequence"]

        if self.config["camera"] == "stereo_left":
            self.img_folder = self.sequence_path / "stereo_left"
        else:
            self.img_folder = self.sequence_path / "stereo_right"

        # ------------------------------------------------------------------
        # 1. Configuração da Câmera (Intrínsecos)
        # ------------------------------------------------------------------
        # Valores hardcoded baseados no dataset KAIST Urban
        # Assinatura estimada do PinholeCamera: (width, height, fx, fy, cx, cy)
        if PinholeCamera is not None:
            self.cam = PinholeCamera(
                1280.0,  # Width
                560.0,  # Height
                847.416,  # fx
                847.416,  # fy
                635.556,  # cx
                518.018,  # cy
            )

        # Guardamos os intrínsecos raw também caso precise sem a classe utilitária
        self.K = np.array([[847.416, 0, 635.556], [0, 847.416, 518.018], [0, 0, 1]])

        # ------------------------------------------------------------------
        # 2. Carregamento e Sincronização de Poses (Ground Truth)
        # ------------------------------------------------------------------
        self.gt_poses = []

        # Carregar timestamps das imagens
        timestamp_file = self.img_folder / "timestamps.txt"
        if timestamp_file.exists():
            self.timestamps = np.loadtxt(timestamp_file)
        else:
            # Fallback: tentar ler dos nomes dos arquivos
            img_files = sorted(list(self.img_folder.glob("*.png")))
            self.timestamps = np.array([int(p.stem) for p in img_files])

        # Carregar GPS (VRS) para calcular as poses
        vrs_gps_path = self.sequence_path / "sensor_data/vrs_gps.csv"
        print("vrs_gps_path", vrs_gps_path)

        if vrs_gps_path.exists():
            # Processamento do CSV (adaptado do seu código original)
            try:
                gps_df = pd.read_csv(vrs_gps_path, header=None)
                # Define colunas críticas para extração
                # Assumindo formato V2 (18 colunas) ou V1 (17 colunas), o X/Y UTM são colunas 3 e 4 (índice base 0)
                # Col 0: Timestamp, Col 3: X_UTM, Col 4: Y_UTM, Col 5: Altitude
                # Col 13: Heading Valid, Col 14: Magnetic Heading

                gps_data = gps_df.values
                print(gps_df.values)
                gps_timestamps = gps_data[:, 0]

                # Para cada imagem, encontrar o GPS mais próximo e montar a pose
                for ts in self.timestamps:
                    # Encontrar índice do timestamp mais próximo
                    idx = np.argmin(np.abs(gps_timestamps - ts))
                    row = gps_data[idx]

                    # Montar matriz de pose 4x4
                    pose = np.eye(4)
                    pose[0, 3] = row[3]  # x_utm
                    pose[1, 3] = row[4]  # y_utm
                    pose[2, 3] = row[5]  # altitude

                    # Rotação (Heading)
                    # Verifica heading_valid (coluna 13)
                    if row[13] == 1:
                        heading = np.radians(row[14])
                        cos_h = np.cos(heading)
                        sin_h = np.sin(heading)
                        # Rotação em torno do eixo Z (Yaw)
                        pose[:3, :3] = np.array(
                            [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]]
                        )

                    self.gt_poses.append(pose)

            except Exception as e:
                logging.error(f"Error loading GPS data: {e}")
                # Preencher com identidades se falhar para não quebrar o loop
                self.gt_poses = [np.eye(4) for _ in self.timestamps]
        else:
            logging.warning("GPS file not found. Poses will be Identity.")
            self.gt_poses = [np.eye(4) for _ in self.timestamps]

        # ------------------------------------------------------------------
        # 3. Configuração de Imagens
        # ------------------------------------------------------------------
        self.img_id = self.config["start"]

        # Usando glob para listar arquivos, similar ao KITTI Loader
        # O KAIST usa timestamps no nome, então sorted() é crucial
        search_path = str(self.img_folder / "*.png")
        self.img_files = sorted(glob.glob(search_path))
        self.img_N = len(self.img_files)

    def get_cur_pose(self):
        """Retorna a pose GT correspondente à imagem atual do iterador"""
        # Ajuste de índice pois img_id avança após o yield
        if 0 <= self.img_id - 1 < len(self.gt_poses):
            return self.gt_poses[self.img_id - 1] - self.gt_poses[0]
        return np.eye(4)

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


if __name__ == "__main__":
    # Exemplo de uso idêntico ao KITTILoader
    # Ajuste o root_path conforme sua máquina
    dataset_config = {
        "root_path": "/run/media/aki/OhShit/dataset_hdd/urban27",
        "sequence": "urban27",
    }

    loader = ComplexUrbanDatasetLoader(dataset_config)

    print(f"Total frames: {len(loader)}")

    for img in tqdm(loader):
        # Exibe texto na imagem
        cv2.putText(
            img,
            "Press any key but Esc to continue, press Esc to exit",
            (10, 30),
            cv2.FONT_HERSHEY_PLAIN,
            1,
            (0, 0, 255),
            1,
            8,
        )

        cv2.imshow("img", img)

        # Teste rápido: pegar a pose atual
        pose = loader.get_cur_pose()
        # print(f"Current Translation: {pose[0,3]:.2f}, {pose[1,3]:.2f}")

        if cv2.waitKey(0) == 27:  # WaitKey(0) espera input do usuário (passo a passo)
            break

    cv2.destroyAllWindows()
>>>>>>> d9d7c5c (Added files)
