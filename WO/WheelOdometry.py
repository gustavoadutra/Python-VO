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
        encoder_param_path = (
            Path(config["root_path"])
            / config["sequence"]
            / "calibration"
            / "EncoderParameter.txt"
        )
        print(encoder_param_path)
        csv_path = (
            Path(config["root_path"])
            / config["sequence"]
            / "sensor_data"
            / "encoder.csv"
        )

        # Default Parameters (Prius approximations) if file not found
        self.ticks_per_rev = 4096.0
        self.radius_left = 0.311
        self.radius_right = 0.311
        self.base_line = 1.52

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

    def update(self, left_tick, right_tick, dt):
        """
        Calculates pose update using Differential Drive Kinematics.
        
        Updates the internal pose state (rotation matrix, translation vector, heading angle)
        based on wheel encoder ticks and elapsed time. Also estimates linear and angular velocities.
        
        Args:
            left_tick (float): Left wheel encoder ticks count
            right_tick (float): Right wheel encoder ticks count
            dt (float): Time delta since last update in seconds
            
        Returns:
            tuple: (cur_theta, cur_R, cur_t, w, v) where:
                - cur_theta: Current heading angle in radians
                - cur_R: 3x3 rotation matrix (current orientation)
                - cur_t: 3x1 translation vector (current position)
                - w: Angular velocity (yaw rate) in rad/s
                - v: Linear velocity in m/s
        """
        if self.index == 0:
            self.prev_ticks = (left_tick, right_tick)
            self.cur_R = np.identity(3)
            self.cur_t = np.zeros((3, 1))
            self.cur_theta = 0.0
            self.v = 0.0
            self.w = 0.0
        else:
            d_left_ticks = left_tick - self.prev_ticks[0]
            d_right_ticks = right_tick - self.prev_ticks[1]
            self.prev_ticks = (left_tick, right_tick)

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
            self.cur_theta += d_theta
            
            # Linear velocity (m/s) and Angular velocity (rad/s)
            if dt > 0:
                self.v = dist_center / dt
                self.w = d_theta / dt
            else:
                self.v = 0.0
                self.w = 0.0

        self.index += 1
        return self.cur_theta, self.cur_R, self.cur_t, self.w, self.v, self.cur_theta
