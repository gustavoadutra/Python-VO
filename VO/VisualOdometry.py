import numpy as np
import cv2
from .LandmarkManager import LandmarkManager

class VisualOdometry(object):
    """
    A robust frame-by-frame monocular visual odometry.
    Handles frames with no keypoints or matching failures gracefully.
    """
    def __init__(self, detector, matcher, cam, enable_pnp=True):
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

        self.landmark_manager = LandmarkManager()
        self.ref_idx_to_landmark_id = {}
        self.current_observations = []

        self.K = np.array([[self.focal[0], 0, self.pp[0]],
                           [0, self.focal[1], self.pp[1]],
                           [0, 0, 1]], dtype=float)

        self.ba_active = False
        self.enable_pnp_conf = enable_pnp

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

            object_points_3d = []
            image_points_2d = []
            inlier_matches_idx = []

            # Só percorre e cria arrays 3D se o PnP estiver ativado
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

            # =========================================================
            # CAMINHO 1: PnP — usa landmarks 3D já triangulados
            # =========================================================
            if self.enable_pnp_conf and len(object_points_3d) >= 15:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=object_points_3d,
                    imagePoints=image_points_2d,
                    cameraMatrix=self.K,
                    distCoeffs=None,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=2.0
                )

                if success and inliers is not None and len(inliers) >= 10:
                    use_pnp = True
                    print(f"Frame {self.index}: PnP inliers = {len(inliers)} / {len(object_points_3d)}")

                    R_cam2world, _ = cv2.Rodrigues(rvec)
                    prev_R = self.cur_R.copy()
                    prev_t = self.cur_t.copy()

                    self.cur_R = R_cam2world.T
                    self.cur_t = -self.cur_R.dot(tvec)

                    R_rel = prev_R.T.dot(self.cur_R)
                    t_rel = prev_R.T.dot(self.cur_t - prev_t)

                    new_ref_idx_to_landmark_id = {}
                    if self.ba_active:
                        self.current_observations = []

                    for i in inliers.flatten():
                        idx_original = inlier_matches_idx[i]
                        match = good_matches[idx_original][0]
                        u_cur, v_cur = cur_pts[idx_original]
                        lm_id = self.ref_idx_to_landmark_id[match.queryIdx]
                        
                        self._track_landmark(lm_id, match.trainIdx, u_cur, v_cur, new_ref_idx_to_landmark_id)

                    self.ref_idx_to_landmark_id = new_ref_idx_to_landmark_id

            # =========================================================
            # CAMINHO 2: Essential Matrix — fallback
            # =========================================================

            if not use_pnp:
                if self.enable_pnp_conf:
                    print(f"Frame {self.index}: PnP falhou ou pontos insuficientes. Usando Essential Matrix.")

                E, mask = cv2.findEssentialMat(
                    ref_pts, cur_pts, cameraMatrix=self.K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
                _, R, t, mask = cv2.recoverPose(E, ref_pts, cur_pts, cameraMatrix=self.K)

                inlier_mask = mask.flatten().astype(bool)

                if absolute_scale > 0:
                    R_rel = R.T
                    t_rel = -R.T.dot(absolute_scale * t)

                    prev_R = self.cur_R.copy()
                    prev_t = self.cur_t.copy()

                    self.cur_R = prev_R.dot(R_rel)
                    self.cur_t = prev_t + prev_R.dot(t_rel)

                    new_ref_idx_to_landmark_id = {}
                    if self.ba_active:
                        self.current_observations = []

                    P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
                    P2 = self.K @ np.hstack((R, absolute_scale * t))

                    print(f"Frame {self.index}: {np.sum(inlier_mask)} inliers")
                    if self.ba_active or self.enable_pnp_conf:
                        for idx, is_inlier in enumerate(inlier_mask):
                            if not is_inlier:
                                continue

                            match = good_matches[idx][0]
                            q_idx = match.queryIdx
                            t_idx = match.trainIdx
                            u_cur, v_cur = cur_pts[idx]

                            if q_idx in self.ref_idx_to_landmark_id:
                                # Ponto conhecido
                                lm_id = self.ref_idx_to_landmark_id[q_idx]
                                self._track_landmark(lm_id, t_idx, u_cur, v_cur, new_ref_idx_to_landmark_id)
                            
                            # OTIMIZAÇÃO: Só triangula se o PnP estiver ativado ou BA ligado
                            elif self.enable_pnp_conf or self.ba_active:
                                pt4d = cv2.triangulatePoints(
                                    P1, P2,
                                    ref_pts[idx].reshape(2, 1),
                                    cur_pts[idx].reshape(2, 1)
                                )
                                pt3d_local = (pt4d[:3, 0] / pt4d[3, 0]).reshape(3, 1)

                                if not (0.01 < pt3d_local[2, 0] < 100):
                                    print(f"[VO WARNING] Profundidade inválida: {pt3d_local[2, 0]:.4f}. Ignorando.")
                                    continue

                                pt3d_global = prev_R.dot(pt3d_local) + prev_t
                                descriptor = self.kptdescs["cur"]["descriptors"][t_idx]
                                lm_id = self.landmark_manager.add_landmark(
                                    pt3d_global.flatten(), descriptor, self.index, u_cur, v_cur
                                )

                                self._track_landmark(lm_id, t_idx, u_cur, v_cur, new_ref_idx_to_landmark_id)

                        self.ref_idx_to_landmark_id = new_ref_idx_to_landmark_id

            self.kptdescs["ref"] = self.kptdescs["cur"]

        except Exception as e:
            print(f"Frame {self.index} Error: {e}")
            self.kptdescs["ref"] = self.kptdescs["cur"]
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