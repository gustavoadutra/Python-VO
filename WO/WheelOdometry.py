import os
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


class WheelOdometry(object):
    """
    Wheel Odometry with KAIST CSV support and Asymmetric Wheel Calibration.
    
    Implements differential drive kinematics with support for:
    - Asymmetric left/right wheel radius calibration
    - KAIST encoder CSV loading and timestamp interpolation
    - Pose tracking with rotation matrices and translation vectors
    - Linear and angular velocity estimation
    """

    def __init__(self, config: Dict = {}):
        """
        Initialize Wheel Odometry system.
        
        Loads calibration parameters from EncoderParameter.txt and encoder data from encoder.csv.
        Falls back to default Prius approximation parameters if files are not found.
        
        Args:
            config (Dict): Configuration dictionary with keys:
                - 'root_path': Root directory of dataset
                - 'sequence': Sequence identifier (e.g., 'dataset_20260126_084725')
        """
        """
        encoder_param_path = (
            Path(config["root_path"])
            / config["sequence"]
            / "calibration"
            / "EncoderParameter.txt"
        )
        """
        encoder_param_path = None  # Set to None to skip loading calibration file (use defaults)


        csv_path = (
            Path(config["root_path"])
            / config["sequence"]
            / "sensor_data"
            / "encoder.csv"
        )

        # Default Parameters (Prius approximations) if file not found
        self.ticks_per_rev = 980
        self.radius_left = 0.06
        self.radius_right = 0.06
        self.base_line = 0.335

        # If parameter file is provided, load it immediately
        if encoder_param_path:
            self.load_calibration(encoder_param_path)
        else:
            self._update_conversion_factors()

        # Internal state
        self.index = 0
        self.prev_ticks = None
        self.cur_theta = 0.0
        self.cur_R = np.identity(3)
        self.cur_t = np.zeros((3, 1))

        # Data storage
        self.df = None
        if csv_path:
            self.load_kaist_csv(csv_path)

    def load_calibration(self, param_file):
        """
        Parses EncoderParameter.txt to set precise calibration values.
        
        Reads encoder resolution, left/right wheel diameters, and wheel base from the 
        calibration file. Automatically converts diameters to radii. Updates conversion 
        factors after loading.
        
        Args:
            param_file (str or Path): Path to EncoderParameter.txt calibration file
            
        Raises:
            Exception: If file cannot be opened or parsed; falls back to default parameters
        """
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
        """
        Calculates tick-to-meter conversion factors for each wheel.
        
        Computes the distance traveled per encoder tick for left and right wheels
        using: conversion_factor = (2 * pi * radius) / ticks_per_revolution
        
        Allows asymmetric wheel calibration for left and right wheels.
        Prints calibration parameters to console for verification.
        """
        self.tick_to_meter_left = (2 * np.pi * self.radius_left) / self.ticks_per_rev
        self.tick_to_meter_right = (2 * np.pi * self.radius_right) / self.ticks_per_rev

        print("[DEBUG] Calibration Loaded:")
        print(f"  - Radius L: {self.radius_left:.5f}m")
        print(f"  - Radius R: {self.radius_right:.5f}m")
        print(f"  - Base: {self.base_line:.5f}m")
        print(f"  - Resolution: {self.ticks_per_rev}")

    def load_kaist_csv(self, csv_path):
        """
        Loads the encoder data CSV in KAIST format.
        
        Reads a CSV file with columns: timestamp (nanoseconds), left ticks, right ticks.
        Converts timestamps from nanoseconds to seconds for internal use.
        
        Args:
            csv_path (str or Path): Path to encoder.csv file with columns [timestamp, left, right]
            
        Raises:
            FileNotFoundError: If CSV file does not exist at the specified path
        """
        print(f"[INFO] Loading encoder data from: {csv_path}")
        try:
            self.df = pd.read_csv(
                csv_path, header=None, names=["timestamp", "left", "right"]
            )
            self.df["timestamp"] = self.df["timestamp"] / 1e9
            print(f"[INFO] Loaded {len(self.df)} encoder entries.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find encoder file at: {csv_path}")

    def get_interpolated_ticks(self, target_time):
        """
        Synchronizes encoder ticks to a specific image timestamp via linear interpolation.
        
        Uses binary search to find the encoder readings closest to the target timestamp,
        then linearly interpolates between them to get ticks at the exact target time.
        
        Args:
            target_time (float): Target timestamp in seconds
            
        Returns:
            tuple: (left_ticks, right_ticks) interpolated at target_time
        """
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
        """
        Calculates proper tick differences between two timestamps using raw sensor data.
        
        This method solves the interpolation-error problem by:
        1. Finding all raw encoder readings strictly between t1 and t2
        2. Summing their actual tick differences (preserves true increments)
        3. Interpolating only the fractional parts at the boundaries
        
        This is more accurate than interpolating absolute values independently
        and then differencing them, which causes errors to compound (especially
        problematic for turning angle calculations).
        
        Args:
            t1 (float): Start timestamp in seconds
            t2 (float): End timestamp in seconds
            
        Returns:
            tuple: (delta_left_ticks, delta_right_ticks) between t1 and t2
        """
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
                for i in range(idx_start, idx_end):
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
        """
        Calculates pose update using Differential Drive Kinematics.
        
        Supports two input modes:
        
        Mode 1 (Legacy): Absolute tick counts
            update(left_tick=100, right_tick=105, dt=0.05)
            
        Mode 2 (NEW - Recommended): Timestamps with proper interpolation
            update(prev_timestamp=10.5, cur_timestamp=10.6)
            Automatically calculates proper tick deltas from raw encoder data,
            avoiding interpolation errors in turning calculations.
        
        Args:
            left_tick (float, optional): Left wheel encoder ticks (Mode 1)
            right_tick (float, optional): Right wheel encoder ticks (Mode 1)
            dt (float, optional): Time delta since last update in seconds (Mode 1)
            prev_timestamp (float, optional): Previous frame timestamp in seconds (Mode 2)
            cur_timestamp (float, optional): Current frame timestamp in seconds (Mode 2)
            
        Returns:
            tuple: (cur_theta, cur_R, cur_t, w, v) where:
                - cur_theta: Current heading angle in radians
                - cur_R: 3x3 rotation matrix (current orientation)
                - cur_t: 3x1 translation vector (current position)
                - w: Angular velocity (yaw rate) in rad/s
                - v: Linear velocity in m/s
        """
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
                "  - Mode 2 (recommended): prev_timestamp, cur_timestamp"
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
