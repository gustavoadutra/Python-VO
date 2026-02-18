
import logging
from typing import Dict

import numpy as np


class LinearKalmanFilter(object):
    """
    Linear Kalman Filter for state estimation (pose fusion of VO and WO).
    
    State vector: [x, y, theta]
    Uses constant-velocity motion model with linear measurement model.
    """
    default_config = {
        "measurement_noise_vo": 0.05,  # Lower = trust VO more
        "measurement_noise_wo": 0.1,   # Higher = trust WO less
    }

    def __init__(self, config: Dict = {}):
        """
        Initializes the Linear Kalman Filter with default parameters.
        """
        self.config = {**self.default_config, **config}

        logging.basicConfig(level=logging.INFO)
        logging.info(f"Initializing Linear Kalman Filter with config: {self.config}")

        # State vector [x, y, theta]
        self.state = np.zeros((3, 1))
        # State covariance matrix
        self.P = np.eye(3)
        # Process noise covariance (added during prediction)
        self.Q = np.eye(3) * 0.01
        # Measurement noise covariance
        self.R = np.eye(3) * 0.1
        # Accumulated displacement for state update
        self.last_vo_measurement = None

    def initialize(self, initial_state):
        """
        Initializes the filter state with first measurement.
        
        Args:
            initial_state: [x, y, theta] initial pose (3×1 array)
        """
        self.state = np.array(initial_state).reshape(3, 1)
        self.P = np.eye(3) * 0.5  # Start with moderate uncertainty
        logging.info(f"EKF initialized with state: {self.state.T}")

    def predict(self, dt, u=None):
        """
        Predicts the next state using linear constant-position motion model.
        
        Args:
            dt: Time step between frames (seconds)
            u: Optional control input (not used in constant-position model)
        
        Motion model: x_k = F @ x_{k-1}
        where F = I (constant position assumption)
        """
        # Linear motion model: F = identity (no dynamics, state persists)
        # For constant-velocity model, F would be [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # and state would include velocities
        F = np.eye(3)
        
        # State prediction 
        # For constant position: x stays same
        # If you add velocity to state, this becomes: x_new = F @ x + B @ u
        self.state = F @ self.state
        
        # Predict covariance: P_pred = F * P * F^T + Q
        # This increases uncertainty over time
        self.P = F @ self.P @ F.T + self.Q

    def update(self, vo_data, wo_data, R_noise_vo=None, R_noise_wo=None):
        """
        Updates the state estimate based on blended VO and WO measurements.
        
        Args:
            vo_data: dict with keys 'R' (3x3), 't' (3x1), from Visual Odometry
            wo_data: dict with keys 'R' (3x3), 't' (3x1), 'yaw' from Wheel Odometry
            R_noise_vo: VO measurement noise (uses config default if None)
            R_noise_wo: WO measurement noise (uses config default if None)
            
        Returns:
            tuple: (R_ekf, t_ekf) - Rotation matrix and translation vector
        """
        # Use config defaults if not provided
        if R_noise_vo is None:
            R_noise_vo = self.config["measurement_noise_vo"]
        if R_noise_wo is None:
            R_noise_wo = self.config["measurement_noise_wo"]
        
        # Extract measurements
        vo_measurement = vo_data["t"]  # 3x1 array [x, y, z]
        
        # Handle WO measurement - use yaw if available, otherwise use translation
        if wo_data and wo_data["t"] is not None:
            wo_measurement = wo_data["t"]  # 3x1 array
            # Use yaw from WO if available
            if "yaw" in wo_data and wo_data["yaw"] is not None:
                wo_measurement = np.array([[wo_measurement[0, 0]], 
                                          [wo_measurement[1, 0]], 
                                          [wo_data["yaw"]]])
        else:
            wo_measurement = self.state.copy()  # Use current state if no WO
            R_noise_wo = float('inf')  # No trust in missing measurement
        
        # Weight measurements inversely by their noise
        vo_weight = 1 / R_noise_vo
        wo_weight = 1 / R_noise_wo if R_noise_wo != float('inf') else 0
        total_weight = vo_weight + wo_weight
        
        if total_weight > 0:
            # Blend measurements (weighted average)
            blended_measurement = (vo_weight * vo_measurement + wo_weight * wo_measurement) / total_weight
        else:
            blended_measurement = vo_measurement
        
        # Blended noise (harmonic mean of noise values)
        blended_noise = 1 / (total_weight + 1e-10)
        
        # Perform the measurement update
        self._measurement_update(blended_measurement, blended_noise)
        
        # Generate output matrices from state
        R_ekf = vo_data.get("R", np.eye(3))  # Use VO rotation
        t_ekf = self.state.copy()  # Use filtered state for translation
        
        return R_ekf, t_ekf

    def _measurement_update(self, measurement, noise):
        """
        Internal method to perform the linear measurement update step.
        
        Measurement model (linear): z = H @ x + v
        where H = I (direct measurement of state), v is measurement noise
        
        Args:
            measurement: [x, y, theta] measurement (3×1 array)
            noise: measurement noise scalar
        """
        # Linear measurement matrix H (identity - directly measure [x, y, theta])
        H = np.eye(3)
        
        # Measurement noise covariance (diagonal matrix)
        R = np.eye(3) * noise
        
        # Kalman Gain: K = P * H^T / (H * P * H^T + R)
        # Tells us how much to trust measurement vs. prediction
        S = H @ self.P @ H.T + R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Innovation (measurement residual): y = z - H @ x_pred
        innovation = measurement - H @ self.state
        
        # Update state estimate: x = x_pred + K @ y
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K @ H) @ P_pred
        # Measurement reduces uncertainty
        self.P = (np.eye(3) - K @ H) @ self.P
    
    def get_state(self):
        """
        Returns the current state estimate [x, y, theta].
        
        Returns:
            tuple: (x, y, theta) as floats
        """
        return float(self.state[0]), float(self.state[1]), float(self.state[2])