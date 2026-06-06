import os
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


class WheelOdometry(object):
    """Wheel Odometry with differential drive kinematics and encoder calibration."""

    def __init__(self, config: Dict = {}):
        """Initialize with config dict containing 'root_path' and 'sequence' keys."""

        encoder_param_path = (
            Path(config["root_path"])
            / config["sequence"]
            / "calibration"
            / "EncoderParameter.txt"
        )

        csv_path = (
            Path(config["root_path"])
            / config["sequence"]
            / "sensor_data"
            / "encoder.csv"
        )

        # Mode indicator: True for direct position (cusco), False for encoder ticks (kaist)
        self.use_direct_position = config.get("use_direct_position", False)

        # Default Parameters (Prius approximations) if file not found
        self.ticks_per_rev = 980
        self.radius_left = 0.06
        self.radius_right = 0.06
        self.base_line = 0.335

        # Always initialize conversion factors with defaults
        self._update_conversion_factors()

        # If parameter file is provided, load it to override defaults
        if encoder_param_path.exists() and not self.use_direct_position:
            self.load_calibration(encoder_param_path)

        # Internal state
        self.index = 0
        self.prev_ticks = None
        self.cur_theta = 0.0
        self.cur_R = np.identity(3)
        self.cur_t = np.zeros((3, 1))
        self.prev_position = None
        self.v = 0.0
        self.w = 0.0

        # Data storage
        self.df = None
        if csv_path:
            if self.use_direct_position:
                self.load_direct_position_csv(csv_path)
            else:
                self.load_kaist_csv(csv_path)

    def load_calibration(self, param_file):
        """Load calibration parameters from EncoderParameter.txt."""
        print(f"[INFO] Loading calibration from {param_file}...")
        try:
            with open(param_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if "Encoder resolution" in line:
                    self.ticks_per_rev = float(line.split(":")[1])
                elif "left wheel diameter" in line:
                    # File gives Diameter, we need Radius (Dia / 2)
                    self.radius_left = float(line.split(":")[1]) / 2.0
                elif "right wheel diameter" in line:
                    self.radius_right = float(line.split(":")[1]) / 2.0
                elif "wheel base" in line:
                    self.base_line = float(line.split(":")[1])

            self._update_conversion_factors()

        except Exception as e:
            print(f"[ERROR] Failed to load calibration: {e}")
            print("[WARN] Using default parameters.")

    def _update_conversion_factors(self):
        """Calculate tick-to-meter conversion factors for asymmetric wheels."""
        self.tick_to_meter_left = (2 * np.pi * self.radius_left) / self.ticks_per_rev
        self.tick_to_meter_right = (2 * np.pi * self.radius_right) / self.ticks_per_rev

        print("[DEBUG] Calibration Loaded:")
        print(f"  - Radius L: {self.radius_left:.5f}m")
        print(f"  - Radius R: {self.radius_right:.5f}m")
        print(f"  - Base: {self.base_line:.5f}m")
        print(f"  - Resolution: {self.ticks_per_rev}")

    def load_kaist_csv(self, csv_path):
        """Load encoder CSV with [timestamp, left, right] columns."""
        print(f"[INFO] Loading encoder data from: {csv_path}")
        try:
            self.df = pd.read_csv(
                csv_path, header=None, names=["timestamp", "left", "right"]
            )
            self.df["timestamp"] = self.df["timestamp"] / 1e9
            print(f"[INFO] Loaded {len(self.df)} encoder entries.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find encoder file at: {csv_path}")

    def load_direct_position_csv(self, csv_path):
        """Load CSV with direct position data [timestamp, image_filename, x, y, z] (cusco format)."""
        print(f"[INFO] Loading direct position data from: {csv_path}")
        try:
            self.df = pd.read_csv(csv_path)
            # Ensure columns exist
            required_cols = ["timestamp", "x", "y", "z"]
            if not all(col in self.df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            print(f"[INFO] Loaded {len(self.df)} position entries.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find position file at: {csv_path}")

    def get_interpolated_ticks(self, target_time):
        """Linearly interpolate ticks at target_time."""
        if self.df is None:
            return 0, 0

        idx = np.searchsorted(self.df["timestamp"], target_time)

        if idx == 0:
            return self.df.iloc[0]["left"], self.df.iloc[0]["right"]
        if idx >= len(self.df):
            return self.df.iloc[-1]["left"], self.df.iloc[-1]["right"]

        row_prev, row_next = self.df.iloc[idx - 1], self.df.iloc[idx]
        t1, t2 = row_prev["timestamp"], row_next["timestamp"]

        if t2 - t1 == 0:
            return row_prev["left"], row_prev["right"]

        alpha = (target_time - t1) / (t2 - t1)
        interp_left = row_prev["left"] + alpha * (row_next["left"] - row_prev["left"])
        interp_right = row_prev["right"] + alpha * (
            row_next["right"] - row_prev["right"]
        )
        return interp_left, interp_right

    def get_tick_deltas(self, t1, t2):
        """Get proper tick differences between two timestamps via interpolation."""
        if self.df is None:
            return 0, 0
        
        if t1 >= t2:
            return 0, 0
        
        # Find encoder samples in the time window [t1, t2]
        idx_start = np.searchsorted(self.df["timestamp"], t1, side='right')
        idx_end = np.searchsorted(self.df["timestamp"], t2, side='left')
        
        d_left = 0.0
        d_right = 0.0
        
        # Get interpolated values at t1 (to establish baseline)
        left_at_t1, right_at_t1 = self.get_interpolated_ticks(t1)
        
        # Sum all intermediate encoder readings
        if idx_start < len(self.df) and idx_start <= idx_end:
            # Add delta from t1 to first encoder sample at idx_start
            if idx_start > 0:
                row_at_t1_lower = self.df.iloc[idx_start - 1]
                row_at_t1_upper = self.df.iloc[idx_start]
                
                # Interpolate the first full reading to get exact tick value at t1 boundary
                t_lower = row_at_t1_lower["timestamp"]
                t_upper = row_at_t1_upper["timestamp"]
                
                if t_upper > t_lower:
                    alpha = (t1 - t_lower) / (t_upper - t_lower)
                    left_boundary = row_at_t1_lower["left"] + alpha * (row_at_t1_upper["left"] - row_at_t1_lower["left"])
                    right_boundary = row_at_t1_lower["right"] + alpha * (row_at_t1_upper["right"] - row_at_t1_lower["right"])
                else:
                    left_boundary = row_at_t1_lower["left"]
                    right_boundary = row_at_t1_lower["right"]
                
                # Delta from interpolated t1 to first real encoder sample
                d_left += row_at_t1_upper["left"] - left_boundary
                d_right += row_at_t1_upper["right"] - right_boundary
            
            # Sum all complete intermediate readings
            if idx_start < idx_end:
                # Ensure i + 1 never exceeds valid dataframe indices
                for i in range(idx_start, min(idx_end, len(self.df) - 1)):
                    d_left += self.df.iloc[i + 1]["left"] - self.df.iloc[i]["left"]
                    d_right += self.df.iloc[i + 1]["right"] - self.df.iloc[i]["right"]
            
            # Add delta from last encoder sample to t2
            if idx_end < len(self.df):
                row_at_t2_lower = self.df.iloc[idx_end]
                if idx_end + 1 < len(self.df):
                    row_at_t2_upper = self.df.iloc[idx_end + 1]
                    t_lower = row_at_t2_lower["timestamp"]
                    t_upper = row_at_t2_upper["timestamp"]
                    
                    if t_upper > t_lower:
                        alpha = (t2 - t_lower) / (t_upper - t_lower)
                        left_boundary = row_at_t2_lower["left"] + alpha * (row_at_t2_upper["left"] - row_at_t2_lower["left"])
                        right_boundary = row_at_t2_lower["right"] + alpha * (row_at_t2_upper["right"] - row_at_t2_lower["right"])
                    else:
                        left_boundary = row_at_t2_lower["left"]
                        right_boundary = row_at_t2_lower["right"]
                    
                    d_left += left_boundary - row_at_t2_lower["left"]
                    d_right += right_boundary - row_at_t2_lower["right"]
        
        return d_left, d_right

    def update(self, left_tick=None, right_tick=None, dt=None, prev_timestamp=None, cur_timestamp=None):
        """Update pose using differential drive kinematics or direct position. Return (theta, R, t, w, v)."""
        # Mode 3: Direct position (cusco dataset)
        if self.use_direct_position and cur_timestamp is not None:
            return self._update_from_direct_position(cur_timestamp)
        
        # Mode 2: Timestamp-based (new, recommended)
        if prev_timestamp is not None and cur_timestamp is not None:
            dt = cur_timestamp - prev_timestamp
            if dt <= 0:
                self.v = 0.0
                self.w = 0.0
                return self.cur_theta, self.cur_R, self.cur_t, self.w, self.v
            
            # Get proper tick deltas from raw encoder data (solves interpolation errors)
            d_left_ticks, d_right_ticks = self.get_tick_deltas(prev_timestamp, cur_timestamp)
        
        # Mode 1: Legacy absolute tick input
        elif left_tick is not None and right_tick is not None and dt is not None:
            if self.index == 0:
                self.prev_ticks = (left_tick, right_tick)
                d_left_ticks = 0.0
                d_right_ticks = 0.0
            else:
                d_left_ticks = left_tick - self.prev_ticks[0]
                d_right_ticks = right_tick - self.prev_ticks[1]
                self.prev_ticks = (left_tick, right_tick)
        else:
            raise ValueError(
                "Must provide either:\n"
                "  - Mode 1 (legacy): left_tick, right_tick, dt\n"
                "  - Mode 2 (recommended): prev_timestamp, cur_timestamp\n"
                "  - Mode 3 (cusco direct position): cur_timestamp"
            )
        
        # Handle first frame
        if self.index == 0:
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
            self.cur_theta = 0.0
            self.v = 0.0
            self.w = 0.0
        else:
            # Use asymmetric conversion factors
            d_left = d_left_ticks * self.tick_to_meter_left
            d_right = d_right_ticks * self.tick_to_meter_right

            # Differential Drive Math
            dist_center = (d_right + d_left) / 2.0
            d_theta = (d_right - d_left) / self.base_line

            # Move along current heading
            dx = dist_center * np.cos(d_theta / 2.0)
            dy = dist_center * np.sin(d_theta / 2.0)

            # Update Pose
            dt_rel = np.array([[dx], [dy], [0.0]])
            c, s = np.cos(d_theta), np.sin(d_theta)
            dR_rel = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

            self.cur_t = self.cur_t + self.cur_R.dot(dt_rel)
            self.cur_R = self.cur_R.dot(dR_rel)
            # Updates yaw angle
            self.cur_theta += d_theta
            
            # Linear velocity (m/s) and Angular velocity (rad/s)
            if dt > 0:
                self.v = dist_center / dt
                self.w = d_theta / dt
            else:
                self.v = 0.0
                self.w = 0.0

        self.index += 1
        return self.cur_theta, self.cur_R, self.cur_t, self.w, self.v

    def _update_from_direct_position(self, cur_timestamp):
        """Update pose from direct position data (cusco format)."""
        if self.df is None:
            raise RuntimeError("No position data loaded")
        
        # Find the closest row to cur_timestamp
        idx = np.argmin(np.abs(self.df["timestamp"] - cur_timestamp))
        cur_row = self.df.iloc[idx]
        
        cur_x = cur_row["x"]
        cur_y = cur_row["y"]
        cur_z = cur_row["z"]
        
        # Handle first frame
        if self.index == 0:
            self.cur_t = np.array([[cur_x], [cur_y], [cur_z]])
            self.prev_position = np.array([cur_x, cur_y, cur_z])
            self.cur_R = np.identity(3)
            self.cur_theta = 0.0
            self.v = 0.0
            self.w = 0.0
        else:
            # Calculate displacement from previous position
            cur_pos = np.array([cur_x, cur_y, cur_z])
            displacement = cur_pos - self.prev_position
            
            # Update position
            self.cur_t = cur_pos.reshape((3, 1))
            
            # Calculate angle from displacement
            dx = displacement[0]
            dy = displacement[1]
            
            # Calculate new heading angle
            if np.linalg.norm(displacement[:2]) > 1e-6:
                new_theta = np.arctan2(dy, dx)
                d_theta = new_theta - self.cur_theta
                
                # Normalize angle to [-pi, pi]
                d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
                
                self.cur_theta = new_theta
                
                # Update rotation matrix (2D rotation for yaw)
                c, s = np.cos(self.cur_theta), np.sin(self.cur_theta)
                self.cur_R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            else:
                self.v = 0.0
                self.w = 0.0
            
            self.prev_position = cur_pos
        
        self.index += 1
        return self.cur_theta, self.cur_R, self.cur_t, self.w, self.v
