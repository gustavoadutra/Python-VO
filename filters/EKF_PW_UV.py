import numpy as np
import logging
from typing import Dict, Union

class ExtendedKalmanFilter:
    """
    Kalman Filter for fusing Visual Odometry (VO) and Wheel Odometry (WO).
    
    CONFIGURAÇÃO INVERTIDA:
    State vector: [x, z, yaw]^T  (2D position on the ground plane)
    Motion Model (Predict): Driven by Cartesian displacement from Visual Odometry (VO)
    Measurement Model (Update): Direct observation of [x, z, yaw] from Wheel Odometry (WO)
    """
    default_config = {
        "measurement_noise_wo": 0.5,  # Agora a medição é do WO
        "process_noise_pos": 0.01,    # Agora o processo (predição) é do VO
    }

    def __init__(self, config: Dict = {}):
        self.config = {**self.default_config, **config}
        logging.basicConfig(level=logging.INFO)
        
        # State vector [x, z, yaw]
        self.state = np.zeros((3, 1))
        
        # State covariance matrix P (3x3)
        self.P = np.eye(3)
        
        # Process noise covariance Q (VO)
        self.Q = np.eye(3) * self.config["process_noise_pos"]

        # Measurement noise covariance R (WO)
        self.R = np.eye(3) * self.config["measurement_noise_wo"]

        # Variáveis para calcular o Delta do VO na predição
        self.prev_t_vo = None

    def initialize(self):
        """Initializes the filter state [x, z, yaw]."""
        self.state = np.zeros((3, 1))
        self.P = np.eye(3) * 0.1
        self.prev_t_vo = None
        logging.info(f"Filter initialized with state: {self.state.T}")

    def predict(self, vo_data: Union[Dict, np.ndarray]):
        """
        Prediction step using Visual Odometry (VO).
        Calcula o deslocamento (delta) desde a última medição e atualiza o estado.
        """
        # --- Helper to extract translation vector ---
        def extract_t(data):
            if isinstance(data, dict):
                return data.get("t")
            return data

        t_vo_raw = extract_t(vo_data)
        
        # Mapeando os eixos do VO para o formato do estado [x, z, yaw/y]
        t_vo = np.array([[t_vo_raw[0, 0]], [t_vo_raw[2, 0]], [t_vo_raw[1, 0]]])

        # Se for o primeiro frame, apenas salva e não prevê (não há delta ainda)
        if self.prev_t_vo is None:
            self.prev_t_vo = t_vo
            return

        # 1. Input de Controle (u): Deslocamento entre o frame anterior e o atual
        delta_vo = t_vo - self.prev_t_vo
        
        # 2. State Prediction: x = Fx + u
        # Como estamos somando deltas cartesianos globais diretamente, a Jacobiana (F) 
        # é simplesmente a matriz Identidade.
        F = np.eye(3)
        
        self.state = self.state + delta_vo
        # print(f"Predicted state: {self.state.T}")

        # 3. Covariance Prediction: P = F * P * F.T + Q
        self.P = F @ self.P @ F.T + self.Q

        # Atualiza o frame anterior
        self.prev_t_vo = t_vo

    def update(self, t_wo: np.ndarray, yaw_wo: float):
        """
        Measurement update using Wheel Odometry (WO).
        Recebe a pose global acumulada da odometria de roda.
        """
        # Mapeando as coordenadas do WO para o vetor de medição z
        # Assumindo que o WO retorna [x, y, 0], onde o 'y' do WO corresponde ao 'z' (profundidade) do VO.
        z_meas = np.array([
            [t_wo[0, 0]],  # x_wo
            [t_wo[1, 0]],  # y_wo (mapeado para z)
            [yaw_wo]       # yaw_wo
        ])

        self._measurement_update(z_meas)

        # Reconstruir saídas para o plotter
        R_out = np.eye(3)
        t_out = np.zeros((3, 1))
        t_out[0, 0] = self.state[0, 0]
        t_out[1, 0] = 0 # y é ignorado no seu plotter 2D
        t_out[2, 0] = self.state[1, 0]
        
        return R_out, t_out

    def _measurement_update(self, z):
        """
        Generic Linear Kalman Update Step.
        """
        # H is Identity (3x3) because we measure [x, z, yaw] directly from WO
        H = np.eye(3)

        # 1. Innovation (Residual)
        y = z - H @ self.state
        
        # 2. Innovation Covariance
        S = H @ self.P @ H.T + self.R

        # 3. Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. State Update
        self.state = self.state + K @ y

        # 5. Covariance Update
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        """Returns x, z"""
        return float(self.state[0]), float(self.state[1])