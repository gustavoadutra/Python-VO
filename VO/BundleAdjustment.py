import numpy as np
import gtsam

class GTSAMBundleAdjuster(object):
    """
    Incremental Bundle Adjuster using GTSAM iSAM2.
    Optimizes both camera poses and 3D landmark positions.
    """

    def __init__(self, config=None):
        if config is None:
            config = {}

        # 1. Initialize iSAM2
        # iSAM2 manages the windowing and sparsity dynamically
        parameters = gtsam.ISAM2Params()
        # Says if the optimizer should go back and adjust past variables because the drift is too high
        # High means
        parameters.setRelinearizeThreshold(0.1)
        parameters.relinearizeSkip = 1
        self.isam2 = gtsam.ISAM2(parameters)

        # 2. Camera Calibration
        # Must be provided from the config file via the dataset loader
        # To optimizer understand how 3D points project to 2D, 
        fx = float(config["fx"])
        fy = float(config["fy"])
        cx = float(config["cx"])
        cy = float(config["cy"])
        self.calibration = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        # 3. Tuned Noise Models
        # [Roll, Pitch, Yaw, X, Y, Z] - Notice translation (XYZ) has higher uncertainty than rotation
        self.noise_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1], dtype=float)
        )
        self.noise_odom = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2], dtype=float)
        )
        # Projection noise (measured in pixels)
        pixel_noise = float(config.get("pixel_noise", 1.0))
        self.noise_proj = gtsam.noiseModel.Isotropic.Sigma(2, pixel_noise)

        self.current_key = 0
        self.seen_landmarks = set() # Track which 3D points are already in the graph
        self.last_pose = None

    def _pose3_from_rt(self, R, t):
        # creates the pose 3d for the graph
        if isinstance(t, np.ndarray):
            t = np.asarray(t).reshape(3, 1)
            point = gtsam.Point3(float(t[0, 0]), float(t[1, 0]), float(t[2, 0]))
        else:
            point = gtsam.Point3(float(t[0]), float(t[1]), float(t[2]))
        return gtsam.Pose3(gtsam.Rot3(R), point)

    def update(self, absolute_pose, relative_rotation=None, relative_translation=None, 
               observations=None, landmark_initials=None):
        """
        Add a new keyframe and landmarks, then incrementally optimize.

        Args:
            absolute_pose: tuple(R, t) current absolute pose estimate from VO.
            relative_rotation: 3x3 rotation from previous to current frame (optional).
            relative_translation: 3x1 translation from previous to current frame (optional).
            observations: List of tuples (landmark_id, u_pixel, v_pixel).
            landmark_initials: Dict mapping landmark_id -> (x, y, z) for newly seen landmarks.
        Returns:
            Tuple (R_opt, t_opt) for the optimized current pose.
        """
        if absolute_pose is None:
            raise ValueError("absolute_pose must be provided.")
        
        # Default empty lists/dicts if not provided
        observations = observations or []
        landmark_initials = landmark_initials or {}

        # iSAM2 requires us to only pass *new* factors and *new* values on each step
        # Saves pose prior, odometry constrains and projection factors for the current keyframe
        new_factors = gtsam.NonlinearFactorGraph()
        # current camera pose and any new landmarks observed in this frame
        new_values = gtsam.Values()

        # Add current pose to new values
        R_vo, t_vo = absolute_pose
        # get current pose 3d
        current_pose = self._pose3_from_rt(R_vo, t_vo)
        # creates the symbol for the current pose
        pose_symbol = gtsam.symbol('x', self.current_key)
        print(f"Adding pose symbol: {pose_symbol} for keyframe {self.current_key}")
        # add the new pose and its symbol
        new_values.insert(pose_symbol, current_pose)

        # 1. Pose Graph Factors (Odometry & Prior)
        if self.current_key == 0:
            # Anchor the first frame, with its own noise
            new_factors.add(gtsam.PriorFactorPose3(pose_symbol, current_pose, self.noise_prior))
        else:
            # Add odometry constraint from the previous frame
            if relative_rotation is not None and relative_translation is not None:
                # get the previous pose symbol
                prev_symbol = gtsam.symbol('x', self.current_key - 1)
                # creates the relative pose from the VO estimate
                rel_pose = self._pose3_from_rt(relative_rotation, relative_translation)
                print(f"Adding odometry factor: {prev_symbol} -> {pose_symbol}")
                # saves the new factor between the previous and current pose with the relative pose and noise model
                new_factors.add(
                    gtsam.BetweenFactorPose3(prev_symbol, pose_symbol, rel_pose, self.noise_odom)
                )

        # 2. Bundle Adjustment Factors (3D landmarks projected to 2D)
        # for each observation
        # u is the horizontal pixel coordinate, v is the vertical pixel coordinate
        for lm_id, u, v in observations:
            # creates the symbol for the landmark
            lm_symbol = gtsam.symbol('l', lm_id)
            # creates the measurement for the 2d point in the image
            measurement = gtsam.Point2(u, v)
            
            # Add projection factor for this observation
            print(f"Adding projection factor: {pose_symbol} -> {lm_symbol}")
            new_factors.add(
                gtsam.GenericProjectionFactorCal3_S2(
                    measurement, self.noise_proj, pose_symbol, lm_symbol, self.calibration
                )
            )
            print(f"Added projection factor for landmark {lm_id} observed at pixel ({u}, {v})")
            # If this is the first time we've seen this landmark
            # provide an initial 3D guess
            if lm_id not in self.seen_landmarks:
                if lm_id not in landmark_initials:
                    raise ValueError(f"Initial 3D position missing for new landmark {lm_id}")
                # get 3d position for the new landmark
                lx, ly, lz = landmark_initials[lm_id]
                # creates the 3d point and it's simbol and adds it to the new values
                new_values.insert(lm_symbol, gtsam.Point3(lx, ly, lz))
                # mark as seen so we don't re-insert it in future frames
                print(f"Adding new landmark symbol: {lm_symbol} for landmark ID {lm_id} at position ({lx}, {ly}, {lz})")
                self.seen_landmarks.add(lm_id)

        # 3. Update iSAM2 and calculate the estimate
        self.isam2.update(new_factors, new_values)
        
        # calculateEstimate() gives us the fully optimized graph thus far
        result = self.isam2.calculateEstimate()
        
        optimized_pose = result.atPose3(pose_symbol)
        self.last_pose = optimized_pose
        self.current_key += 1

        return optimized_pose.rotation().matrix(), np.array(
            optimized_pose.translation()
        ).reshape(3, 1)

    def get_last_pose(self):
        return self.last_pose