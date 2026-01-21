import cv2
import numpy as np
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Iterator

from utils.PinholeCamera import PinholeCamera


class ComplexUrbanDatasetLoader:
    """
    DataLoader for the Complex Urban Dataset.
    """

    default_config = {
        "root_path": "/run/media/aki/OhShit/dataset_hdd/urban27",
        "sequence": "urban27",
        "start": 0,
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

        self.K_raw = fs.getNode("camera_matrix").mat()
        self.D = fs.getNode("distortion_coefficients").mat()
        self.R = fs.getNode("rectification_matrix").mat()
        self.P = fs.getNode("projection_matrix").mat()

        fs.release()

        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K_raw, self.D, self.R, self.P[:3, :3], (1280, 560), cv2.CV_32F
        )

        if PinholeCamera is not None:
            self.cam = PinholeCamera(
                width=1280,
                height=560,
                fx=self.P[0, 0],
                fy=self.P[1, 1],
                cx=self.P[0, 2],
                cy=self.P[1, 2],
            )

    def _load_data(self):
        """
        Loads the image list and creates 'self.times' for synchronization.
        """
        # Load Image Timestamps from CSV
        stamp_file_path = self.sequence_path / "sensor_data/stereo_stamp.csv"

        if not stamp_file_path.exists():
            raise FileNotFoundError(f"Timestamp file not found: {stamp_file_path}")

        # Load timestamps (Nanoseconds)
        df_stamps = pd.read_csv(stamp_file_path, header=None)
        self.timestamps = pd.to_numeric(df_stamps.iloc[:, 0]).values.astype(np.int64)
        print("timestamps", self.timestamps[0])
        # === CRITICAL ADDITION FOR WHEEL ODOMETRY SYNC ===
        # Convert nanoseconds to seconds so main.py can sync with encoder.csv
        self.times = self.timestamps / 1e9
        # =================================================
        print("time", self.times)
        self.img_files = []
        for ts in self.timestamps:
            img_p = self.img_folder / f"{ts}.png"
            self.img_files.append(str(img_p))

        logging.info(f"Loaded {len(self.img_files)} image entries.")

        # Load GPS Data for Ground Truth (Optional validation)
        vrs_gps_path = self.sequence_path / "sensor_data/vrs_gps.csv"
        if not vrs_gps_path.exists():
            logging.error("GPS file not found. GT poses will be empty.")
            self.gt_poses = []
            return

        gps_df = pd.read_csv(vrs_gps_path, header=None)
        needed_cols = [0, 3, 4, 5, 12, 13]
        for col in needed_cols:
            gps_df[col] = pd.to_numeric(gps_df[col])

        gps_data = gps_df.values
        gps_ts_raw = gps_data[:, 0].astype(np.float64)
        gps_x_raw = gps_data[:, 3].astype(np.float64)
        gps_y_raw = gps_data[:, 4].astype(np.float64)
        gps_z_raw = gps_data[:, 5].astype(np.float64)

        if len(gps_ts_raw) == 0:
            self.gt_poses = []
            return

        target_ts = self.timestamps.astype(np.float64)

        interp_x = np.interp(target_ts, gps_ts_raw, gps_x_raw)
        interp_y = np.interp(target_ts, gps_ts_raw, gps_y_raw)
        interp_z = np.interp(target_ts, gps_ts_raw, gps_z_raw)

        raw_poses = []
        for i in range(len(self.timestamps)):
            pose = np.eye(4)
            pose[0, 3] = interp_x[i]
            pose[1, 3] = interp_z[i]
            pose[2, 3] = interp_y[i]

            idx_nearest = np.argmin(np.abs(gps_ts_raw - target_ts[i]))
            row_raw = gps_data[idx_nearest]

            if row_raw[12] == 1:
                heading = np.radians(row_raw[13])
                cos_h = np.cos(heading)
                sin_h = np.sin(heading)
                pose[:3, :3] = np.array(
                    [[cos_h, -sin_h, 0], [sin_h, cos_h, 0], [0, 0, 1]]
                )
            raw_poses.append(pose)

        self.gt_poses = []
        if len(raw_poses) > 0:
            first_pose = raw_poses[0]
            first_pose_inv = np.linalg.inv(first_pose)
            for p in raw_poses:
                rel_pose = first_pose_inv @ p
                self.gt_poses.append(rel_pose[:3, :4])

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
            self.img_id += 1
            return img_rectified

        raise StopIteration()

    def __getitem__(self, index: int) -> np.ndarray:
        if index >= len(self.img_files):
            raise IndexError("Index out of bounds")
        file_path = self.img_files[index]
        return cv2.imread(file_path)
