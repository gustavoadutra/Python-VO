import cv2
import numpy as np
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
    default_config = {
        "root_path": "/run/media/aki/OhShit/dataset_hdd/urban27",
        "sequence": "urban27",
        "start": 0,
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
