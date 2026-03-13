import numpy as np
import logging
from typing import Dict, Union

class ExtendedKalmanFilter:
    """
    Kalman Filter for fusing Visual Odometry (VO) and Wheel Odometry (WO).
    
    State vector: [x, z]^T  (2D position on the ground plane)
    Motion Model: Discrete unicycle model with linear and angular velocities from WO
    Measurement Model: Direct observation of [x, z] from VO (Linear)
    """
    default_config = {
        "measurement_noise_vo": 0.5,  # Covariance for VO measurements
        "process_noise_pos": 0.01,    # Process noise for position
    }

    def __init__(self, config: Dict = {}):
        self.config = {**self.default_config, **config}
        logging.basicConfig(level=logging.INFO)
        
        # State vector [x, z, yaw]
        self.state = np.zeros((3, 1))
        
        # State covariance matrix P (3x3)
        self.P = np.eye(3)
        
        # Process noise covariance Q
        self.Q = np.eye(3) * self.config["process_noise_pos"]

        self.R = np.eye(3) * self.config["measurement_noise_vo"]

    def initialize(self):
        """Initializes the filter state [x, z, yaw]."""
        # initial_state should be a list or array of at least 3 elements
        self.state = np.zeros((3, 1))  # [x, z, yaw]
        # State covariance matrix P (3x3)
        self.P = np.eye(3) * 0.1
        logging.info(f"Filter initialized with state: {self.state.T}")

    def predict(self, v_wo: float, yaw_wo: float, dt: float):
        # v_wo is linear velocity
        # yaw_ is yaw angle 

        dx = v_wo * np.cos(yaw_wo) * dt
        dy = v_wo * np.sin(yaw_wo) * dt
        dyaw = 0
        
        control_input = np.array([[dx], [dy], [dyaw]])

        # 2. State Prediction: x = Fx + Bu
        # Jacobian matrix
        F = np.array([[1, 0, -v_wo * np.sin(yaw_wo) * dt],
                      [0, 1, v_wo * np.cos(yaw_wo) * dt],
                      [0, 0, 1]])
        
        self.state = self.state + control_input
        print(f"Predicted state: {self.state.T}")

        # 2. Covariance Prediction
        # P = F * P * F.T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, vo_data: Union[Dict, np.ndarray]):
        """
        Sequentially updates state with VO and WO measurements.
        Handles both Dictionary inputs (from your main loop) or direct Arrays.
        """
        
        # --- Helper to extract translation vector ---
        def extract_t(data):
            if isinstance(data, dict):
                return data.get("t")
            return data

        t_vo = extract_t(vo_data)
        t_vo = np.array([[t_vo[0, 0]], [t_vo[2, 0]], [t_vo[1, 0]]])

        self._measurement_update(t_vo)

        # Construct outputs to match previous interface
        # We assume Identity rotation since we aren't tracking it
        R_out = np.eye(3)
        if isinstance(vo_data, dict) and "R" in vo_data:
             
            R_out = vo_data["R"]

        # Reconstruct 3x1 vector for compatibility with your plotter [x, 0, z]
        t_out = np.zeros((3, 1))
        t_out[0, 0] = self.state[0, 0]
        t_out[1, 0] = 0 # y is ignored
        t_out[2, 0] = self.state[1, 0]
        
        return R_out, t_out

    def _measurement_update(self, z):
        """
        Generic Linear Kalman Update Step.
        Args:
            z: Measurement vector [x, z]
            R: Measurement noise covariance matrix
        """
        # H is Identity (3x3) because we measure [x, z, yaw ] directly
        H = np.eye(3)

        # 1. Innovation (Residual)
        # y = z - H * x_pred

        y = z - H @ self.state
        
        # 2. Innovation Covariance
        # S = H * P * H.T + R
        S = H @ self.P @ H.T + self.R

        # 3. Kalman Gain
        # K = P * H.T * S^-1
        # try:
        K = self.P @ H.T @ np.linalg.inv(S)
        # except np.linalg.LinAlgError:


        # 4. State Update
        # x = x + K * y
        self.state = self.state + K @ y

        # 5. Covariance Update
        # P = (I - K * H) * P
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        """Returns x, z"""
        return float(self.state[0]), float(self.state[1])