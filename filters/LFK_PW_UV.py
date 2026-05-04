import numpy as np
import logging
from typing import Dict, Union

class KalmanFilter:
    """
    Linear Kalman Filter for sensor fusion of Visual Odometry (VO) and Wheel Odometry (WO).
    
    This is a LINEAR Kalman Filter (not extended), as both the motion model and measurement
    model are purely linear transformations with identity matrices (F = I, H = I).
    
    STATE VECTOR CONFIGURATION:
    - State representation: [x, z, yaw]^T (2D position and orientation on ground plane)
    - x: Horizontal displacement (forward/backward along visual odometry X-axis)
    - z: Lateral displacement (sideways, corresponds to VO Z-axis for ground plane)
    - yaw: Rotation angle (heading) around vertical axis
    
    SENSOR FUSION STRATEGY:
    - Prediction step (Motion Model): Linear accumulation of VO Cartesian displacements
      Motion equation: x_k = x_{k-1} + delta_VO_k (no nonlinear kinematics)
      State transition matrix F = I (identity) - direct addition of incremental motion
      Process noise Q represents VO prediction uncertainty and drift
    
    - Update step (Measurement Model): Direct linear measurement from Wheel Odometry
      Measurement equation: z_k = H * x_k where H = I (identity)
      Directly observes [x, z, yaw] from WO with no nonlinear sensor transformation
      Measurement noise R represents WO sensor uncertainty
    
    DESIGN RATIONALE:
    This linear formulation is appropriate because:
    1. VO provides Cartesian displacements already in the state space (no frame conversion needed)
    2. WO measurements are direct pose observations (no sensor-to-state transformation required)
    3. Ground-plane 2D model without complex differential drive kinematics
    4. Linear model prevents over-parameterization while maintaining sensor fusion benefits
    
    WO typically provides stable measurements for small distances and corrects VO's accumulative
    drift, while VO's rapid update rate enables continuous motion tracking.
    """
    default_config = {
        "measurement_noise_wo": 0.9,   # Measurement noise covariance for Wheel Odometry (R)
                                       # Higher values = lower trust in WO measurements
        "process_noise_pos": 0.2,    # Process noise covariance for Visual Odometry (Q)
                                       # Higher values = lower trust in VO predictions
    }

    def __init__(self, config: Dict = {}):
        """Initialize the Linear Kalman Filter with configuration parameters."""
        self.config = {**self.default_config, **config}
        logging.basicConfig(level=logging.INFO)
        
        # Kalman filter state vector: [x_position, z_position, yaw_angle]^T (3x1 vector)
        self.state = np.zeros((3, 1))
        
        # State covariance matrix P (3x3): Uncertainty in state estimate
        # Initialized to identity matrix - assumes initial high uncertainty
        self.P = np.eye(3)
        
        # Process noise covariance Q (3x3): Accounts for VO prediction uncertainty
        # Diagonal matrix where each diagonal element represents trust in VO predictions
        # Q_val smaller = higher trust in VO model; Q_val larger = model is less reliable
        self.Q = np.eye(3) * self.config["process_noise_pos"]

        # Measurement noise covariance R (3x3): Accounts for WO measurement uncertainty
        # Diagonal matrix where each diagonal element represents noise in WO sensors
        # R_val smaller = higher trust in WO; R_val larger = sensors are noisier
        self.R = np.eye(3) * self.config["measurement_noise_wo"]

        # Previous VO translation vector for computing frame-to-frame displacement (delta)
        # Used in predict() to calculate incremental motion between frames
        self.prev_t_vo = None

    def initialize(self):
        """Initializes the filter state [x, z, yaw]."""
        self.state = np.zeros((3, 1))
        self.P = np.eye(3) * 0.1
        self.prev_t_vo = None
        logging.info(f"Filter initialized with state: {self.state.T}")

    def predict(self, vo_data: Union[Dict, np.ndarray]):
        """
        Prediction step of the Kalman filter using Visual Odometry (VO).
        
        Computes the incremental displacement (delta) since the last frame and updates
        the state estimate. This acts as the motion model in the filter.
        
        Args:
            vo_data: Either a dictionary with key 't' containing translation, or directly
                    a 3x1 translation vector from Visual Odometry
        """
        # Helper function to extract translation vector from flexible input formats
        def extract_t(data):
            """Extract translation vector from dict or ndarray format."""
            if isinstance(data, dict):
                return data.get("t")
            return data

        t_vo_raw = extract_t(vo_data)
        
        # Coordinate frame mapping: VO uses camera frame [X, Y, Z] 
        # Convert to state frame [x, z, yaw] used by ground-plane 2D model
        # VO frame X (forward) -> state x (forward)
        # VO frame Z (lateral) -> state z (lateral)  
        # VO frame Y (vertical) -> state yaw (orientation) is not used
        t_vo = np.array([[t_vo_raw[0, 0]], [t_vo_raw[2, 0]], [0]])

        # On first frame, initialize previous position without making a prediction
        # Cannot compute meaningful delta without prior position reference
        if self.prev_t_vo is None:
            self.prev_t_vo = t_vo
            return

        # STEP 1: Control Input (u) - Incremental motion from previous to current frame
        # This represents the dead-reckoning measurement from Visual Odometry
        delta_vo = t_vo - self.prev_t_vo

        # STEP 2: State Prediction using linear motion model: x_k = F * x_{k-1} + u_k
        # Since we accumulate Cartesian displacements directly in the global frame,
        # the state transition matrix F is simply the identity matrix (no nonlinear dynamics)
        # This assumes constant velocity or at least small time steps
        F = np.eye(3)
        
        # Update state: add the incremental motion to current estimate
        self.state = self.state + delta_vo

        # STEP 3: Covariance Prediction: P_k = F * P_{k-1} * F^T + Q
        # Propagate uncertainty through the motion model
        # Q represents how much we expect the model to be wrong (VO drift/noise)
        self.P = F @ self.P @ F.T + self.Q

        # Store current position for next iteration's delta computation
        self.prev_t_vo = t_vo

    def update(self, t_wo: np.ndarray, yaw_wo: float):
        """
        Measurement update step of the Kalman filter using Wheel Odometry (WO).
        
        Corrects the predicted state estimate based on direct observations from wheel sensors.
        This step reduces accumulated drift in the position estimate.
        
        Args:
            t_wo: Global accumulated translation from wheel odometry [x, y]^T (3x1 vector)
            yaw_wo: Current yaw angle (rotation) from wheel odometry (scalar in radians) IGNORED
            
        Returns:
            Tuple of (R_out, t_out): Rotation matrix and translation vector in VO frame format
        """
        # Coordinate frame mapping: WO provides measurement in vehicle frame [x_vehicle, y_vehicle]
        # Map to state frame [x, z, yaw] for consistency with prediction step
        # WO x (forward) -> state x (forward)
        # WO y (lateral) -> state z (lateral in depth) - note: y_wo corresponds to z in state
        # WO yaw -> state yaw (orientation) - direct mapping but ignored in this 2D model
        z_meas = np.array([
            [t_wo[0, 0]],  # x component from wheel odometry
            [t_wo[1, 0]],  # y component from WO (mapped to z in state frame)
            [0]       # yaw/heading angle from wheel odometry (IGNORED)
        ])

        # Perform the linear Kalman update with measurement from WO
        self._measurement_update(z_meas)

        # Reconstruct output in the format expected by the trajectory plotter
        # The plotter uses VO frame format: [3x3 rotation matrix, 3x1 translation vector]
        R_out = np.eye(3)  # Rotation: Identity (no rotation change in this 2D model)
        t_out = np.zeros((3, 1))  # Translation vector in output frame
        t_out[0, 0] = self.state[0, 0]  # x position from filtered state
        t_out[1, 0] = 0                  # y ignored in 2D ground plane model
        t_out[2, 0] = self.state[1, 0]  # z position from filtered state
        
        return R_out, t_out

    def _measurement_update(self, z):
        """
        Perform linear Kalman filter measurement update step.
        
        Updates the state estimate and covariance based on the measurement innovation
        (difference between observed and predicted measurements).
        
        Standard Kalman update equations:
        - Compute innovation (measurement residual)
        - Calculate Kalman gain to weight innovation vs. prediction confidence
        - Update state estimate and covariance matrix
        
        Args:
            z: Measurement vector [x_meas, z_meas, yaw_meas]^T (3x1)
        """
        # Measurement matrix H: relates state to measurements
        # Identity (3x3) because we directly measure [x, z, yaw] from WO
        # (No nonlinear transformation needed; measurements are in state space)
        H = np.eye(3)

        # STEP 1: Calculate Innovation (measurement residual)
        # Difference between actual measurement and predicted measurement
        # Positive values mean measurement is higher than prediction
        y = z - H @ self.state
        
        # STEP 2: Calculate Innovation Covariance (S matrix)
        # Represents the uncertainty in the measurement residual
        # Combines prediction uncertainty (P) and measurement noise (R)
        S = H @ self.P @ H.T + self.R

        # STEP 3: Calculate Kalman Gain (K matrix)
        # Determines how much weight to give the measurement vs. the prediction
        # Small S (confident measurements) -> large K (trust measurement more)
        # Large S (uncertain measurements) -> small K (trust prediction more)
        K = self.P @ H.T @ np.linalg.inv(S)

        # STEP 4: Update State Estimate
        # Blend prediction with measurement-based correction
        # Correction is weighted by Kalman gain: higher gain = larger correction
        self.state = self.state + K @ y

        # STEP 5: Update Covariance Matrix (uncertainty reduction)
        # Joseph form of covariance update for numerical stability
        # As measurements come in, uncertainty should decrease
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        """
        Extract current filtered position estimate from state vector.
        
        Returns:
            Tuple of (x, z): Current x and z position estimates (floats)
        """
        return float(self.state[0]), float(self.state[1])