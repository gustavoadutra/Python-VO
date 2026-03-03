import numpy as np
import logging
from typing import Dict, Optional, Tuple, Union

class ExtendedKalmanFilter:
    """
    Kalman Filter for fusing Visual Odometry (VO) and Wheel Odometry (WO).
    
    State vector: [x, z]  (2D position on the ground plane)
    Motion Model: Constant Position (Identity) + Noise
    Measurement Model: Direct state observation (Linear)
    """
    default_config = {
        "measurement_noise_vo": 0.05,  # Covariance for VO
        "measurement_noise_wo": 0.1,   # Covariance for WO
        "process_noise_pos": 0.01,     # Uncertainty added per step
    }

    def __init__(self, config: Dict = {}):
        self.config = {**self.default_config, **config}
        logging.basicConfig(level=logging.INFO)
        
        # State vector [x, z]
        self.state = np.zeros((2, 1))
        
        # State covariance matrix P (2x2)
        self.P = np.eye(2)
        
        # Process noise covariance Q (2x2)
        # Tuning parameter: how much uncertainty motion adds per step
        self.Q = np.eye(2) * self.config["process_noise_pos"]

    def initialize(self, initial_state):
        """Initializes the filter state [x, z]."""
        # initial_state should be a list or array of at least 2 elements
        self.state = np.array([initial_state[0], initial_state[1]]).reshape(2, 1)
        self.P = np.eye(2) * 0.1
        logging.info(f"Filter initialized with state: {self.state.T}")

    def predict(self, dt: float, u: Optional[Tuple[float, float]] = None):
        """
        Prediction Step.
        Since Theta is removed, we cannot project velocity (v) into x/z components 
        without external orientation. 
        
        We assume a 'Static' or 'Constant Velocity' model where the best prediction 
        of the next position is the current position plus process noise 
        (unless explicit vx, vz velocities are provided).
        """
        # F is Identity (2x2)
        # x_k = x_{k-1}
        # z_k = z_{k-1}
        F = np.eye(2)

        # 1. State Prediction
        # If u contains (vx, vz), we could do: self.state += u * dt
        # For now, we assume Identity transition
        self.state = F @ self.state

        # 2. Covariance Prediction
        # P = F * P * F.T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, vo_data: Union[Dict, np.ndarray], wo_data: Union[Dict, np.ndarray]):
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
        t_wo = extract_t(wo_data)

        # --- Update with Visual Odometry (VO) ---
        if t_vo is not None:
            # Measurement vector [x, z]
            # Mapping index 0 -> x, index 2 -> z (depth)
            z_meas = np.array([[t_vo[0, 0]], [t_vo[2, 0]]])

            # Noise Matrix
            R_cov = np.eye(2) * self.config["measurement_noise_vo"]
            
            self._measurement_update(z_meas, R_cov)

        # --- Update with Wheel Odometry (WO) ---
        if t_wo is not None:
            # Measurement vector [x, z]
            z_meas = np.array([[t_wo[0, 0]], [t_wo[2, 0]]])

            # Noise Matrix (Usually higher drift)
            R_cov = np.eye(2) * self.config["measurement_noise_wo"]
            
            self._measurement_update(z_meas, R_cov)

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

    def _measurement_update(self, z, R):
        """
        Generic Linear Kalman Update Step.
        Args:
            z: Measurement vector [x, z]
            R: Measurement noise covariance matrix
        """
        # H is Identity (2x2) because we measure [x, z] directly
        H = np.eye(2)

        # 1. Innovation (Residual)
        # y = z - H * x_pred
        y = z - H @ self.state
        
        # 2. Innovation Covariance
        # S = H * P * H.T + R
        S = H @ self.P @ H.T + R

        # 3. Kalman Gain
        # K = P * H.T * S^-1
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logging.warning("Singular matrix in update, skipping step.")
            return

        # 4. State Update
        # x = x + K * y
        self.state = self.state + K @ y

        # 5. Covariance Update
        # P = (I - K * H) * P
        self.P = (np.eye(2) - K @ H) @ self.P

    def get_state(self):
        """Returns x, z"""
        return float(self.state[0]), float(self.state[1])