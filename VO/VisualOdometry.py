import numpy as np
import cv2
from .LandmarkManager import LandmarkManager

class VisualOdometry(object):
    """
    A robust frame-by-frame monocular visual odometry.
    Handles frames with no keypoints or matching failures gracefully.
    """
    def __init__(self, detector, matcher, cam, enable_pnp=True, config=None):
        if config is None:
            config = {}

        self.detector = detector
        self.matcher = matcher
        self.focal = (cam.fx, cam.fy)
        self.pp = (cam.cx, cam.cy)

        self.index = 0
        self.kptdescs = {}

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        self.landmark_manager = LandmarkManager()
        self.ref_idx_to_landmark_id = {}
        self.current_observations = []

        # Matriz para a imagem ja calibrada
        self.K = np.array([[self.focal[0], 0, self.pp[0]],
                           [0, self.focal[1], self.pp[1]],
                           [0, 0, 1]], dtype=float)

        self.ba_active = False
        self.enable_pnp_conf = enable_pnp 
        
        # minimo de keypoints retornados pelo descritor
        self.min_keypoints = config.get('min_keypoints', 20)
        self.num_ref_keypoints = config.get("min_ref_keypoints", 0)
        # minimo de pontos 3d para fazer triangulacao
        self.min_3d_points = config.get('min_3d_points', 10)
        # minima escala a odometria retornar t e R
        self.min_absolute_scale = config.get('min_absolute_scale', 0.0001)
        # minima e maxima profundidade calculado pela triangulacao
        self.min_depth = config.get('min_depth', 0.00001)
        self.max_depth = config.get('max_depth', 100.0)
        # minimo de paralax para criar novo frame em pixels
        self.min_parallax = config.get('min_parallax', 10)
        # quantos pontos do ref ainda aparecem no cur
        self.min_track_rate = config.get('min_track_rate', 10)
        # minimo geral de inliers
        self.min_inliers = config.get('min_inliers', 0)

    def set_ba_active(self, active: bool):
        self.ba_active = active
        if not active:
            self.current_observations = []

    def _track_landmark(self, lm_id, t_idx, u_cur, v_cur, new_ref_idx):
        """Método auxiliar para atualizar mapas e registrar observações do BA."""
        new_ref_idx[t_idx] = lm_id
        if self.ba_active:
            self.landmark_manager.add_observation(lm_id, self.index, u_cur, v_cur)
            self.current_observations.append((lm_id, u_cur, v_cur))

    def update(self, image, absolute_scale=1.0):
        # Extração dos pontos
        kptdesc = self.detector(image)
        is_keyframe = False

        # Verifica se o detector retornou keypoints suficientes
        if kptdesc is None or len(kptdesc.get("keypoints", [])) < self.min_keypoints:
            print(f"[VO WARNING] Frame {self.index} com {len(kptdesc.get('keypoints', []))} keypoints.")
            self.kptdescs["cur"] = {"keypoints": [], "descriptors": [], "scores": []}
            self.index += 1
            return self.cur_R, self.cur_t, None, None
        else:
            self.kptdescs["cur"] = kptdesc

        # Se for o primeiro descritor
        if self.index == 0 or "ref" not in self.kptdescs:
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.index += 1
            return self.cur_R, self.cur_t, None, None

        t_rel = None
        R_rel = None

        try:
            # Matching 
            good_matches = self.matcher.match(self.kptdescs)
            matched_dict = self.matcher.get_good_keypoints(self.kptdescs)

            ref_pts = matched_dict["ref_keypoints"]
            cur_pts = matched_dict["cur_keypoints"]
            # DEPOIS — só atualiza se for um keyframe
            # robô parado ou pouco movimento: mantém o mesmo keyframe como ref
            # o PnP ainda funciona normalmente pois os landmarks já existem
            if self.enable_pnp_conf:
                is_keyframe = self._should_create_keyframe(ref_pts, cur_pts)

            object_points_3d = []
            image_points_2d = []
            inlier_matches_idx = []

            if self.enable_pnp_conf:
                for idx in range(len(ref_pts)):
                    match = good_matches[idx][0]
                    q_idx = match.queryIdx

                    if q_idx in self.ref_idx_to_landmark_id:
                        lm_id = self.ref_idx_to_landmark_id[q_idx]
                        if lm_id in self.landmark_manager.landmarks_3d:
                            pt3d = self.landmark_manager.landmarks_3d[lm_id]
                            object_points_3d.append(pt3d)
                            image_points_2d.append(cur_pts[idx])
                            inlier_matches_idx.append(idx)

                object_points_3d = np.array(object_points_3d, dtype=np.float32)
                image_points_2d = np.array(image_points_2d, dtype=np.float32)

            use_pnp = False
            
            # Novo dicionário temporário para rastrear as correspondências do frame ATUAL
            new_ref_idx_to_landmark_id = {}
            if self.ba_active:
                self.current_observations = []

            # =========================================================
            # CAMINHO 1: PnP — usa landmarks 3D já triangulados
            # =========================================================
            if self.enable_pnp_conf and len(object_points_3d) >= self.min_3d_points:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points_3d,
                    imagePoints=image_points_2d,
                    cameraMatrix=self.K,
                    distCoeffs=None,
                    flags=cv2.SOLVEPNP_EPNP,
                    reprojectionError=2.0
                )
                print(f"Frame {self.index}: PnP inliers = {len(inliers)} / {len(object_points_3d)}")
                if success and inliers is not None and len(inliers) > self.min_inliers:
                    use_pnp = True
                    R_cam2world, _ = cv2.Rodrigues(rvec)
                    prev_R = self.cur_R.copy()
                    prev_t = self.cur_t.copy()

                    self.cur_R = R_cam2world.T
                    self.cur_t = -self.cur_R.dot(tvec)

                    R_rel = prev_R.T.dot(self.cur_R)
                    t_rel = prev_R.T.dot(self.cur_t - prev_t)

                    for i in inliers.flatten():
                        idx_original = inlier_matches_idx[i]
                        match = good_matches[idx_original][0]
                        u_cur, v_cur = cur_pts[idx_original]
                        lm_id = self.ref_idx_to_landmark_id[match.queryIdx]
                        
                        self._track_landmark(lm_id, match.trainIdx, u_cur, v_cur, new_ref_idx_to_landmark_id)
                    
                    print(f"Frame {self.index} landmark")


            # =========================================================
            # CAMINHO 2: Essential Matrix — fallback
            # =========================================================
            if not use_pnp:
                if self.enable_pnp_conf:
                    print(f"[VO WARNING] Frame {self.index}: PnP falhou ou pontos insuficientes. Usando Essential Matrix.")
                # Usa ransac para filtrar os pontos 
                # prob eh a probabilidade encontrar a matriz perfeita
                # threshold e a distancia de erro em pixel de onde ele deveria estar
                # E eh a matriz essencial calculada, mask diz se eh inlier
                E, mask = cv2.findEssentialMat(
                    ref_pts, cur_pts, cameraMatrix=self.K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                # A matriz essencial retorna 4 posicoes possiveis de T e R
                # teste de Cheirality Check para descobrir qual eh a certa
                # o vetor de t eh normalizado, logo nao tem escala
                _, R, t, mask = cv2.recoverPose(E, ref_pts, cur_pts, cameraMatrix=self.K)

                inlier_mask = mask.flatten().astype(bool)
                print(f"Frame {self.index}: {np.sum(inlier_mask)} inliers")

                if absolute_scale > self.min_absolute_scale:
                    R_rel = R.T
                    t_rel = -R.T.dot(absolute_scale * t)

                    prev_R = self.cur_R.copy()
                    prev_t = self.cur_t.copy()

                    self.cur_R = prev_R.dot(R_rel)
                    self.cur_t = prev_t + prev_R.dot(t_rel)
                    # Triangulação e atualização de landmarks
                    if self.ba_active or self.enable_pnp_conf:
                        # Matrizes de projecao
                        # Descreve como um ponto 3D no mundo eh projetado para virar um pixel
                        # Cola as matrizes intrinsecas e extrinsecas
                        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                        P2 = self.K @ np.hstack((R, absolute_scale * t))
                                                
                        # Pegamos os índices exatos onde a máscara é verdadeira (inliers)
                        inlier_indices = np.where(inlier_mask)[0]
                        
                        if len(inlier_indices) > self.min_inliers:
                            # Filtra apenas os pontos válidos para triangulação de uma só vez
                            valid_ref_pts = ref_pts[inlier_indices]
                            valid_cur_pts = cur_pts[inlier_indices]

                            # Triangula todos os pontos válidos de uma só vez (Fora do laço!)
                            pts4d_all = cv2.triangulatePoints(P1, P2, valid_ref_pts.T, valid_cur_pts.T)
                            
                            # Converte de coordenadas homogêneas (4D) para euclidianas (3D) dividindo por W
                            pts3d_local_all = pts4d_all[:3, :] / pts4d_all[3, :]

                            # Agora iteramos apenas para checar profundidade e salvar os landmarks
                            for i, idx in enumerate(inlier_indices):
                                match = good_matches[idx][0]
                                q_idx = match.queryIdx
                                t_idx = match.trainIdx
                                u_cur, v_cur = cur_pts[idx]

                                if q_idx in self.ref_idx_to_landmark_id:
                                    # Ponto já conhecido, apenas atualiza o tracking
                                    lm_id = self.ref_idx_to_landmark_id[q_idx]
                                    self._track_landmark(lm_id, t_idx, u_cur, v_cur, new_ref_idx_to_landmark_id)
                                else:
                                    # Ponto novo: pegamos o ponto 3D correspondente do array triangulado
                                    # i é o índice no array de inliers; idx é o índice original nas features
                                    pt3d_local = pts3d_local_all[:, i].reshape(3, 1)

                                    # Calculo de profundidade para filtragem do ponto triangulado
                                    depth = pt3d_local[2, 0]
                                    if not (self.min_depth < depth < self.max_depth):
                                        if depth <= self.min_depth:
                                            print(f"[VO WARNING] Profundidade abaixo do limite: {depth:.4f}. Ignorando.")
                                        else:
                                            print(f"[VO WARNING] Profundidade acima do limite: {depth:.4f}. Ignorando.")
                                        continue

                                    # Converte o ponto local recém-criado para as coordenadas globais
                                    pt3d_global = prev_R.dot(pt3d_local) + prev_t
                                    descriptor = self.kptdescs["cur"]["descriptors"][t_idx]
                                    
                                    # Adiciona no gerenciador de landmarks
                                    lm_id = self.landmark_manager.add_landmark(
                                        pt3d_global.flatten(), descriptor, self.index, u_cur, v_cur
                                    )

                                    # Atualiza o dicionário de rastreio para o próximo frame
                                    self._track_landmark(lm_id, t_idx, u_cur, v_cur, new_ref_idx_to_landmark_id)

            if is_keyframe:
                # Transfere landmarks visíveis no cur para o novo mapa de referência
                self.kptdescs["ref"] = self.kptdescs["cur"]
                self.num_ref_keypoints = len(self.kptdescs["cur"]["keypoints"])
                self.ref_idx_to_landmark_id = new_ref_idx_to_landmark_id
                print(f"Frame {self.index}: novo keyframe criado. Landmarks transferidos: {len(self.ref_idx_to_landmark_id)}")
            elif not self.enable_pnp_conf:
                self.kptdescs["ref"] = self.kptdescs["cur"]

        except Exception as e:
            print(f"Frame {self.index} Error: {e}")
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.num_ref_keypoints = len(self.kptdescs["cur"]["keypoints"])
            self.ref_idx_to_landmark_id = {}

        self.index += 1
        return self.cur_R, self.cur_t, t_rel, R_rel

    def get_observations_for_ba(self):
        observations = []
        landmark_initials = {}

        for lm_id, u, v in self.current_observations:
            observations.append((lm_id, float(u), float(v)))

            if lm_id in self.landmark_manager.landmarks_3d:
                x, y, z = self.landmark_manager.landmarks_3d[lm_id]
                landmark_initials[lm_id] = (float(x), float(y), float(z))

        return observations, landmark_initials
    
    def _should_create_keyframe(self, ref_pts, cur_pts):
        if len(ref_pts) == 0 or len(cur_pts) == 0:
            return False
        
        # Paralaxe média entre os pontos matchados
        flow = cur_pts - ref_pts
        mean_parallax = np.mean(np.linalg.norm(flow, axis=1))
        
        # Taxa de rastreamento (quantos pontos do ref ainda aparecem no cur)
        tracking_rate = len(cur_pts) / max(self.num_ref_keypoints, 1)
        
        parallax_ok  = mean_parallax > self.min_parallax
        tracking_low = tracking_rate < self.min_track_rate 
        
        return parallax_ok or tracking_low

class AbsoluteScaleComputer(object):
    def __init__(self):
        self.prev_pose = None
        self.cur_pose = None
        self.count = 0

    def update(self, pose):
        self.cur_pose = pose
        scale = 1.0

        if self.count != 0 and self.prev_pose is not None:
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) ** 2
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) ** 2
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) ** 2
            )

        self.count += 1
        self.prev_pose = self.cur_pose.copy()
        return scale