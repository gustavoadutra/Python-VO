import numpy as np
import cv2
from .LandmarkManager import LandmarkManager

class VisualOdometry(object):
    """
    A robust frame-by-frame monocular visual odometry.
    Handles frames with no keypoints or matching failures gracefully.
    """
    def __init__(self, detector, matcher, cam):
        self.detector = detector
        self.matcher = matcher
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        
        self.index = 0
        self.kptdescs = {}
        
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # --- NOVAS ESTRUTURAS DE RASTREAMENTO ---
        self.landmark_manager = LandmarkManager()
        
        # Mapeia: índice_no_ref_keypoints -> landmark_id
        # Isso diz quem é quem no frame imediatamente anterior
        self.ref_idx_to_landmark_id = {} 
        
        # Guardar os matches atuais para o Bundle Adjustment
        self.current_observations = []


    def update(self, image, absolute_scale=1.0):
        # 1. Extração
        kptdesc = self.detector(image)
        if kptdesc is None:
            self.kptdescs["cur"] = {"keypoints": [], "descriptors": [], "scores": []}
        else:
            self.kptdescs["cur"] = kptdesc

        if kptdesc is None or len(kptdesc.get("keypoints", [])) < 8:
            self.index += 1
            return self.cur_R, self.cur_t, None, None

        if self.index == 0 or "ref" not in self.kptdescs:
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.index += 1
            return self.cur_R, self.cur_t, None, None

        try:
            # 2. Matching com o seu FrameByFrameMatcher
            good_matches = self.matcher.match(self.kptdescs) # Retorna lista de [DMatch]
            matched_dict = self.matcher.get_good_keypoints(self.kptdescs)
            
            ref_pts = matched_dict["ref_keypoints"]
            cur_pts = matched_dict["cur_keypoints"]

            if len(ref_pts) < 8:
                raise ValueError("Not enough matches")

            # 3. Recuperação de Pose
            E, mask = cv2.findEssentialMat(cur_pts, ref_pts, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, cur_pts, ref_pts, focal=self.focal, pp=self.pp)

            inlier_mask = mask.flatten().astype(bool)
            
            # Matrizes de projeção para triangulação (Referencial local da câmera anterior)
            P1 = np.array([[self.focal, 0, self.pp[0], 0],
                           [0, self.focal, self.pp[1], 0],
                           [0, 0, 1, 0]], dtype=float)
            P2 = np.hstack((R, absolute_scale * t))
            P2 = np.array([[self.focal, 0, self.pp[0]],
                           [0, self.focal, self.pp[1]],
                           [0, 0, 1]], dtype=float) @ P2

            if absolute_scale > 0:
                # 1. Salvar o estado anterior explícitamente!
                prev_R = self.cur_R.copy()
                prev_t = self.cur_t.copy()

                # Acumular a pose GLOBAL
                rel_t = absolute_scale * t
                self.cur_t = self.cur_t + self.cur_R.dot(rel_t)
                self.cur_R = R.dot(self.cur_R)

                # --- 4. DATA ASSOCIATION & TRIANGULAÇÃO ---
                new_ref_idx_to_landmark_id = {}
                self.current_observations = [] # Limpa as observações deste frame

                # Iterar apenas sobre os inliers que o RANSAC aprovou
                for idx, is_inlier in enumerate(inlier_mask):
                    if not is_inlier:
                        continue
                    
                    # Pegar o objeto de match original do seu FrameByFrameMatcher
                    # Nota: O seu matcher retorna uma lista de listas: [[match1], [match2]]
                    match = good_matches[idx][0] 
                    q_idx = match.queryIdx # Índice no frame anterior
                    t_idx = match.trainIdx # Índice no frame atual
                    
                    u_cur, v_cur = cur_pts[idx]
                    
                    # A MÁGICA: Este ponto já existe?
                    if q_idx in self.ref_idx_to_landmark_id:
                        # SIM! É um landmark conhecido.
                        lm_id = self.ref_idx_to_landmark_id[q_idx]
                        self.landmark_manager.add_observation(lm_id, self.index, u_cur, v_cur)
                        
                        # Passar o ID para o próximo frame
                        new_ref_idx_to_landmark_id[t_idx] = lm_id
                        self.current_observations.append((lm_id, u_cur, v_cur))
                    else:
                        # NÃO! É uma nova quina/ponto encontrado.
                        # Triangula (retorna coordenadas em relação à câmera anterior)
                        pt4d = cv2.triangulatePoints(P1, P2, ref_pts[idx].reshape(2,1), cur_pts[idx].reshape(2,1))
                        pt3d_local = (pt4d[:3, 0] / pt4d[3, 0]).reshape(3, 1)

                        # Filtro de profundidade de segurança
                        if 0.01 < pt3d_local[2, 0] < 100:
                            # TRANFORMAÇÃO PARA COORDENADAS GLOBAIS
                            # Multiplica pela rotação global ANTERIOR e soma a translação global ANTERIOR
                            # (Pois P1 estava na câmera anterior)
                            pt3d_global = prev_R.dot(pt3d_local) + prev_t
                            prev_t = self.cur_t - prev_R.dot(rel_t) 
                            
                            pt3d_global = prev_R.dot(pt3d_local) + prev_t
                            
                            descriptor = self.kptdescs["cur"]["descriptors"][t_idx]
                            
                            lm_id = self.landmark_manager.add_landmark(
                                pt3d_global.flatten(), descriptor, self.index, u_cur, v_cur
                            )
                            
                            new_ref_idx_to_landmark_id[t_idx] = lm_id
                            self.current_observations.append((lm_id, u_cur, v_cur))

                # Atualiza o mapeamento para a próxima iteração
                self.ref_idx_to_landmark_id = new_ref_idx_to_landmark_id

            self.kptdescs["ref"] = self.kptdescs["cur"]

        except Exception as e:
            print(f"Frame {self.index} Error: {e}")
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.ref_idx_to_landmark_id = {} # Reseta o rastreamento local em caso de falha

        self.index += 1
        return self.cur_R, self.cur_t, rel_t if absolute_scale > 0 else None, R if absolute_scale > 0 else None

    def get_observations_for_ba(self):
        """
        Get observations (2D keypoint measurements) and landmarks for Bundle Adjustment.
        
        Returns:
            Tuple (observations, landmark_initials) where:
            - observations: List of (landmark_id, u_pixel, v_pixel)
            - landmark_initials: Dict mapping landmark_id -> (x, y, z) in world frame
        """
        observations = []
        landmark_initials = {}
        
        # Iteramos diretamente sobre a lista construída no método update()
        for lm_id, u, v in self.current_observations:
            observations.append((lm_id, float(u), float(v)))
            
            # Buscamos a posição 3D global inicial no LandmarkManager
            if lm_id in self.landmark_manager.landmarks_3d:
                x, y, z = self.landmark_manager.landmarks_3d[lm_id]
                landmark_initials[lm_id] = (float(x), float(y), float(z))
        
        return observations, landmark_initials


class AbsoluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose
        scale = 1.0
        
        if self.count != 0 and self.prev_pose is not None:
            
            # Distance formula between previous and current GT position
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                * (self.cur_pose[0, 3] - self.prev_pose[0, 3])
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                * (self.cur_pose[1, 3] - self.prev_pose[1, 3])
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3])
                * (self.cur_pose[2, 3] - self.prev_pose[2, 3])
            )
            
        self.count += 1
        self.prev_pose = self.cur_pose.copy()
        return scale