import numpy as np
import cv2

class VisualOdometry(object):
    """
    A robust frame-by-frame monocular visual odometry.
    Handles frames with no keypoints or matching failures gracefully.
    """

    def __init__(self, detector, matcher, cam):
        """
        :param detector: a feature detector (e.g., SIFT, ORB)
        :param matcher: a keypoints matcher (e.g., FLANN, BF)
        :param cam: camera parameters object with fx, fy, cx, cy
        """
        self.detector = detector
        self.matcher = matcher

        # Camera parameters
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)

        # Counters and Data
        self.index = 0
        self.kptdescs = {} # Stores "cur" and "ref"

        # State (Absolute Pose)
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))

        # Relative motion for filter use
        self.relative_motion = np.zeros((3, 1))
        self.relative_translation = np.zeros((3, 1))
        self.relative_rotation = np.eye(3)
        
        # Bundle Adjustment: Store matched keypoints and triangulated landmarks
        self.matched_keypoints_cur = None  # Current frame 2D points
        self.matched_keypoints_ref = None  # Reference frame 2D points
        self.landmark_id_counter = 0       # Counter for assigning unique landmark IDs
        self.landmarks_3d = {}             # Map: landmark_id -> (x, y, z)

    def triangulate_landmarks(self, ref_keypoints, cur_keypoints, R_ref_to_cur, t_ref_to_cur):
        """
        Triangulate 3D landmarks from matched 2D keypoints in ref and cur frames.
        
        Args:
            ref_keypoints: Nx2 array of keypoints in reference frame
            cur_keypoints: Nx2 array of keypoints in current frame
            R_ref_to_cur: 3x3 rotation matrix from ref to cur
            t_ref_to_cur: 3x1 translation vector from ref to cur
            
        Returns:
            Dict mapping landmark_id -> (x, y, z) for successfully triangulated points
        """
        # SHOULDN'T USE THE LAST POSITION MATRIX?
        # way to see if a new point is truly new
        # get two matched keypoints and reconstruct them in 3D space
        # Create projection matrices
        # Reference camera at origin: P1 = K[I|0]
        K = np.array([
            [self.focal, 0, self.pp[0]],
            [0, self.focal, self.pp[1]],
            [0, 0, 1]
        ], dtype=float)
        # concatenate rotation and translation for the reference
        # it's the matrix that projects 2D points in the reference frame to 3D space
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        
        # Current camera: P2 = K[R|t]
        P2 = K @ np.hstack([R_ref_to_cur, t_ref_to_cur])
        
        landmarks = {}
        # for every reference keypoint
        for i in range(len(ref_keypoints)):
            try:
                # Normalize keypoints
                ref_pt = ref_keypoints[i].astype(float)
                cur_pt = cur_keypoints[i].astype(float)
                
                # Use cv2.triangulatePoints
                # fourth component is the scale factor (homogeneous coordinate)
                points_4d = cv2.triangulatePoints(P1, P2, ref_pt.reshape(2, 1), cur_pt.reshape(2, 1))
                
                # Convert from homogeneous to 3D
                if points_4d[3, 0] != 0:
                    pt_3d = points_4d[:3, 0] / points_4d[3, 0]
                    
                    # Only keep points in front of both cameras and with reasonable depth
                    if pt_3d[2] > 0.1 and pt_3d[2] < 100:  # Depth constraint
                        lm_id = self.landmark_id_counter # controls landmark id assignment
                        landmarks[lm_id] = tuple(pt_3d)
                        self.landmark_id_counter += 1
            except Exception as e:
                # Skip problematic points
                continue
        
        return landmarks

    def update(self, image, absolute_scale=1.0):
        """
        Update with a new image and compute the pose.
        :param image: input BGR/Gray image
        :param absolute_scale: ground truth scale for monocular VO
        """
        # 1. Feature Detection
        kptdesc = self.detector(image)
        
        # Ensure 'cur' key exists for external plotting/logging even on failure
        if kptdesc is None:
            self.kptdescs["cur"] = {"keypoints": [], "descriptors": [], "scores": []}
        else:
            self.kptdescs["cur"] = kptdesc

        # 2. Safety Check: If no keypoints found, skip this frame
        if kptdesc is None or len(kptdesc.get("keypoints", [])) < 8:
            print(f"Frame {self.index}: Too few keypoints. Skipping.")
            self.index += 1
            return self.cur_R, self.cur_t, self.relative_motion, self.relative_rotation

        # 3. Initialization: First successful frame becomes the first reference
        if self.index == 0 or "ref" not in self.kptdescs:
            self.kptdescs["ref"] = self.kptdescs["cur"]
            self.index += 1
            return self.cur_R, self.cur_t, None, None

        # 4. Feature Matching
        try:
            matches = self.matcher(self.kptdescs)
            
            # Ensure we have enough matches for the Essential Matrix (5-point min, 8-point rec)
            if matches is None or len(matches.get("cur_keypoints", [])) < 8:
                raise ValueError("Not enough matches between frames")

            # 5. Essential Matrix & Pose Recovery
            E, mask = cv2.findEssentialMat(
                matches["cur_keypoints"],
                matches["ref_keypoints"],
                focal=self.focal,
                pp=self.pp,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            
            if E is None or E.shape != (3, 3):
                raise ValueError("Essential Matrix calculation failed")

            _, R, t, mask = cv2.recoverPose(
                E,
                matches["cur_keypoints"],
                matches["ref_keypoints"],
                focal=self.focal,
                pp=self.pp,
            )

            # 6. Accumulate Pose
            # Scale is applied to the translation vector t (unit vector)
            # To preserve the correct relative transformation for BA, the
            # translation should remain in the previous camera frame.
            if absolute_scale > 0:
                self.relative_translation = absolute_scale * t
                self.relative_rotation = R
                self.relative_motion = self.cur_R.dot(self.relative_translation)

                self.cur_t = self.cur_t + self.relative_motion
                self.cur_R = R.dot(self.cur_R)
                
                # 6b. Triangulate landmarks for Bundle Adjustment
                # Extract inlier keypoints using the mask from recoverPose
                inlier_mask = mask.flatten().astype(bool)
                inlier_ref_kpts = matches["ref_keypoints"][inlier_mask]
                inlier_cur_kpts = matches["cur_keypoints"][inlier_mask]
                
                # Triangulate from reference frame perspective
                self.landmarks_3d = self.triangulate_landmarks(
                    inlier_ref_kpts, inlier_cur_kpts, R, absolute_scale * t
                )
                
                # Store matched keypoints in current frame for observations
                self.matched_keypoints_cur = inlier_cur_kpts
                self.matched_keypoints_ref = inlier_ref_kpts

            # 7. Success! Update reference for the next frame
            self.kptdescs["ref"] = self.kptdescs["cur"]

        except Exception as e:
            # Handle Matcher/OpenCV errors (like the 'index out of range' error)
            print(f"Frame {self.index} Error: {e}. Attempting reset.")
            # We set this current frame as the new reference so the NEXT frame 
            # can try to match against it (Resetting the baseline)
            self.kptdescs["ref"] = self.kptdescs["cur"]

        self.index += 1
        return self.cur_R, self.cur_t, self.relative_translation, self.relative_rotation

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
        
        if (self.matched_keypoints_cur is not None and 
            len(self.landmarks_3d) > 0):
            
            for lm_id, (x, y, z) in self.landmarks_3d.items():
                if lm_id < len(self.matched_keypoints_cur):
                    u, v = self.matched_keypoints_cur[lm_id]
                    observations.append((lm_id, float(u), float(v)))
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