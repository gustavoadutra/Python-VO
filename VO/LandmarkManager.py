import numpy as np

class LandmarkManager:
    def __init__(self):
        self.landmarks_3d = {}  # lm_id -> (X, Y, Z) global
        self.observations = {}  # lm_id -> [(frame_idx, u, v), ...]
        self.descriptors = {}   # lm_id -> descriptor (opcional para rastreio direto, essencial se for fazer loop closure)
        self.landmark_id_counter = 0

    def add_landmark(self, pt_3d_global, descriptor, frame_idx, u, v):
        """Registra um landmark inédito no referencial GLOBAL."""
        lm_id = self.landmark_id_counter
        self.landmarks_3d[lm_id] = tuple(pt_3d_global)
        self.descriptors[lm_id] = descriptor
        self.observations[lm_id] = [(frame_idx, float(u), float(v))]
        
        self.landmark_id_counter += 1
        return lm_id

    def add_observation(self, lm_id, frame_idx, u, v):
        """Adiciona uma nova observação 2D a um landmark existente."""
        # dado um determinado ponto em outra imagem vai adicionar ao landmark a nova observação 
        if lm_id in self.observations:
            self.observations[lm_id].append((frame_idx, float(u), float(v)))