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
        self.relative_rotation = np.eye(3)

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
            # rotated by current orientation
            if absolute_scale > 0:
                self.relative_motion = absolute_scale * self.cur_R.dot(t)
                self.relative_rotation = R
                
                self.cur_t = self.cur_t + self.relative_motion
                self.cur_R = R.dot(self.cur_R)

            # 7. Success! Update reference for the next frame
            self.kptdescs["ref"] = self.kptdescs["cur"]

        except Exception as e:
            # Handle Matcher/OpenCV errors (like the 'index out of range' error)
            print(f"Frame {self.index} Error: {e}. Attempting reset.")
            # We set this current frame as the new reference so the NEXT frame 
            # can try to match against it (Resetting the baseline)
            self.kptdescs["ref"] = self.kptdescs["cur"]

        self.index += 1
        return self.cur_R, self.cur_t, self.relative_motion, self.relative_rotation

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