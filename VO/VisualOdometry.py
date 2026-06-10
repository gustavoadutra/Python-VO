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
        self.focal = (cam.fx, cam.fy)
        self.pp = (cam.cx, cam.cy)
        self.dist_coeffs = cam.dist_coeffs if hasattr(cam, 'dist_coeffs') else None
        print(f"Camera Intrinsics: fx={self.focal[0]}, fy={self.focal[1]}, cx={self.pp[0]}, cy={self.pp[1]}")

        self.index = 0
        self.kptdescs = {}

        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # --- ESTRUTURAS DE RASTREAMENTO ---
        self.landmark_manager = LandmarkManager()

        # Mapeia: índice_no_ref_keypoints -> landmark_id
        # Isso diz quem é quem no frame imediatamente anterior
        self.ref_idx_to_landmark_id = {}

        # Observações do frame atual para o Bundle Adjustment
        # Só é populado quando ba_active=True
        self.current_observations = []

        # Matriz Intrínseca (K)
        self.K = np.array([[self.focal[0], 0, self.pp[0]],
                           [0, self.focal[1], self.pp[1]],
                           [0, 0, 1]], dtype=float)

        # Controle do Bundle Adjustment
        # Setado pelo main via set_ba_active() junto com a criação do ba_obj
        self.ba_active = False

    def set_ba_active(self, active: bool):
        """
        Liga ou desliga o Bundle Adjustment.
        Ao desligar, limpa as observações pendentes para não vazar dados velhos.
        O mapa 3D (landmark_manager / ref_idx_to_landmark_id) continua sendo
        mantido independentemente — ele é necessário para o PnP funcionar.
        """
        self.ba_active = active
        if not active:
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

        t_rel = None
        R_rel = None

        try:
            # 2. Matching
            good_matches = self.matcher.match(self.kptdescs)
            matched_dict = self.matcher.get_good_keypoints(self.kptdescs)

            ref_pts = matched_dict["ref_keypoints"]
            cur_pts = matched_dict["cur_keypoints"]

            # --- Coleta pontos 3D conhecidos para tentar o PnP ---
            object_points_3d = []
            image_points_2d = []
            inlier_matches_idx = []

            for idx in range(len(ref_pts)):
                match = good_matches[idx][0]
                q_idx = match.queryIdx  # índice no frame anterior

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

            # =========================================================
            # CAMINHO 1: PnP — usa landmarks 3D já triangulados
            # =========================================================
            if len(object_points_3d) >= 15:

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points_3d,
                    imagePoints=image_points_2d,
                    cameraMatrix=self.K,
                    distCoeffs=None,        # imagem já retificada
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=2.0
                )

                if success and inliers is not None and len(inliers) >= 10:
                    use_pnp = True
                    print(f"Frame {self.index}: PnP inliers = {len(inliers)} / {len(object_points_3d)}")

                    # --- Pose global (única vez) ---
                    R_cam2world, _ = cv2.Rodrigues(rvec)
                    prev_R = self.cur_R.copy()
                    prev_t = self.cur_t.copy()

                    self.cur_R = R_cam2world.T
                    self.cur_t = -self.cur_R.dot(tvec)

                    # Pose relativa para o BA / Filtro de Kalman
                    R_rel = prev_R.T.dot(self.cur_R)
                    t_rel = prev_R.T.dot(self.cur_t - prev_t)

                    # --- Data association: sempre atualiza o mapa ---
                    new_ref_idx_to_landmark_id = {}
                    if self.ba_active:
                        self.current_observations = []

                    for i in inliers.flatten():
                        idx_original = inlier_matches_idx[i]
                        match = good_matches[idx_original][0]
                        t_idx = match.trainIdx
                        lm_id = self.ref_idx_to_landmark_id[match.queryIdx]
                        u_cur, v_cur = cur_pts[idx_original]

                        # Mapa 3D: sempre atualiza para o próximo frame ter PnP disponível
                        new_ref_idx_to_landmark_id[t_idx] = lm_id

                        # Observações para o BA: só quando ativo
                        if self.ba_active:
                            self.landmark_manager.add_observation(lm_id, self.index, u_cur, v_cur)
                            self.current_observations.append((lm_id, u_cur, v_cur))

                    self.ref_idx_to_landmark_id = new_ref_idx_to_landmark_id

            # =========================================================
            # CAMINHO 2: Essential Matrix — fallback quando PnP falha
            # =========================================================
            if not use_pnp:
                print(f"Frame {self.index}: PnP falhou ou pontos insuficientes. Usando Essential Matrix.")

                E, mask = cv2.findEssentialMat(
                    ref_pts, cur_pts, cameraMatrix=self.K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                _, R, t, mask = cv2.recoverPose(E, ref_pts, cur_pts, cameraMatrix=self.K)

                inlier_mask = mask.flatten().astype(bool)

                if absolute_scale > 0:
                    # Pose relativa (câmera atual no referencial da câmera anterior)
                    # R e t do OpenCV mapeiam X_cur = R*X_ref + t, então a inversa é:
                    R_rel = R.T
                    t_rel = -R.T.dot(absolute_scale * t)

                    prev_R = self.cur_R.copy()
                    prev_t = self.cur_t.copy()

                    self.cur_R = prev_R.dot(R_rel)
                    self.cur_t = prev_t + prev_R.dot(t_rel)

                    # --- Data association & triangulação: sempre atualiza o mapa ---
                    new_ref_idx_to_landmark_id = {}
                    if self.ba_active:
                        self.current_observations = []

                    # Matrizes de projeção no referencial local da câmera anterior
                    P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = self.K @ np.hstack((R, absolute_scale * t))

                    inliers_count = np.sum(inlier_mask)
                    print(f"Frame {self.index}: {inliers_count} inliers")

                    for idx, is_inlier in enumerate(inlier_mask):
                        if not is_inlier:
                            continue

                        match = good_matches[idx][0]
                        q_idx = match.queryIdx
                        t_idx = match.trainIdx
                        u_cur, v_cur = cur_pts[idx]

                        if q_idx in self.ref_idx_to_landmark_id:
                            lm_id = self.ref_idx_to_landmark_id[q_idx]

                            # Mapa 3D: sempre
                            new_ref_idx_to_landmark_id[t_idx] = lm_id

                            # BA: só quando ativo
                            if self.ba_active:
                                self.landmark_manager.add_observation(lm_id, self.index, u_cur, v_cur)
                                self.current_observations.append((lm_id, u_cur, v_cur))

                        else:
                            # Novo ponto: triangula sempre (necessário para o PnP futuro)
                            pt4d = cv2.triangulatePoints(
                                P1, P2,
                                ref_pts[idx].reshape(2, 1),
                                cur_pts[idx].reshape(2, 1)
                            )
                            pt3d_local = (pt4d[:3, 0] / pt4d[3, 0]).reshape(3, 1)

                            # Filtra profundidades inválidas
                            if not (0.01 < pt3d_local[2, 0] < 100):
                                print(f"[VO WARNING] Profundidade inválida: {pt3d_local[2, 0]:.4f}. Ignorando.")
                                continue

                            pt3d_global = prev_R.dot(pt3d_local) + prev_t
                            descriptor = self.kptdescs["cur"]["descriptors"][t_idx]
                            lm_id = self.landmark_manager.add_landmark(
                                pt3d_global.flatten(), descriptor, self.index, u_cur, v_cur
                            )

                            # Mapa 3D: sempre
                            new_ref_idx_to_landmark_id[t_idx] = lm_id

                            # BA: só quando ativo
                            if self.ba_active:
                                self.current_observations.append((lm_id, u_cur, v_cur))

                    self.ref_idx_to_landmark_id = new_ref_idx_to_landmark_id

            self.kptdescs["ref"] = self.kptdescs["cur"]

        except Exception as e:
            print(f"Frame {self.index} Error: {e}")
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.ref_idx_to_landmark_id = {}

        self.index += 1
        return self.cur_R, self.cur_t, t_rel, R_rel

    def get_observations_for_ba(self):
        """
        Retorna as observações 2D e posições 3D iniciais dos landmarks
        para o Bundle Adjustment do frame atual.

        Returns:
            observations: List of (landmark_id, u_pixel, v_pixel)
            landmark_initials: Dict mapping landmark_id -> (x, y, z) no referencial global
        """
        observations = []
        landmark_initials = {}

        for lm_id, u, v in self.current_observations:
            observations.append((lm_id, float(u), float(v)))

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
            scale = np.sqrt(
                (self.cur_pose[0, 3] - self.prev_pose[0, 3]) ** 2
                + (self.cur_pose[1, 3] - self.prev_pose[1, 3]) ** 2
                + (self.cur_pose[2, 3] - self.prev_pose[2, 3]) ** 2
            )

        self.count += 1
        self.prev_pose = self.cur_pose.copy()
        return scale