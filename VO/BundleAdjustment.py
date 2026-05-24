import numpy as np
from collections import deque
import gtsam



class GTSAMBundleAdjuster(object):
    """Sliding-window bundle adjuster using GTSAM pose graph optimization."""

    def __init__(self, config=None, window_size=7):
        if config is None:
            config = {}

        if gtsam is None:
            raise ImportError(
                "GTSAM is not installed. Install it using `pip install gtsam` or `conda install -c conda-forge gtsam` to use --ba."
            )

        self.window_size = int(config.get("window_size", window_size))
        self.prior_sigma = float(config.get("prior_sigma", 1.0))
        self.odometry_sigma = float(config.get("odometry_sigma", 0.1))

        self.keyframes = deque(maxlen=self.window_size)
        self.relative_poses = deque(maxlen=max(0, self.window_size - 1))
        self.current_key = 0
        self.last_pose = None

        self.noise_prior = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.prior_sigma] * 6, dtype=float)
        )
        self.noise_odom = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([self.odometry_sigma] * 6, dtype=float)
        )

    def _pose3_from_rt(self, R, t):
        if isinstance(t, np.ndarray):
            t = np.asarray(t).reshape(3, 1)
            point = gtsam.Point3(float(t[0, 0]), float(t[1, 0]), float(t[2, 0]))
        else:
            point = gtsam.Point3(float(t[0]), float(t[1]), float(t[2]))

        return gtsam.Pose3(gtsam.Rot3(R), point)

    def update(self, relative_rotation, relative_translation, absolute_pose=None):
        """Add a new keyframe and optimize the current sliding window.

        Args:
            relative_rotation: 3x3 rotation matrix from previous to current frame
            relative_translation: 3x1 translation vector from previous to current frame
            absolute_pose: tuple(R, t) with the current absolute pose estimate from VO
        Returns:
            Tuple (R_opt, t_opt) for the optimized current pose.
        """
        if absolute_pose is None:
            raise ValueError("absolute_pose must be provided to GTSAMBundleAdjuster.update()")

        R_vo, t_vo = absolute_pose
        current_pose = self._pose3_from_rt(R_vo, t_vo)

        self.keyframes.append((self.current_key, current_pose))

        if relative_rotation is not None and relative_translation is not None and len(self.keyframes) > 1:
            rel_pose = self._pose3_from_rt(relative_rotation, relative_translation)
            self.relative_poses.append(rel_pose)

        # Build a fresh graph for the current window only
        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        for idx, (key, pose) in enumerate(self.keyframes):
            symbol = gtsam.symbol('x', key)
            initial.insert(symbol, pose)

            if idx == 0:
                graph.add(gtsam.PriorFactorPose3(symbol, pose, self.noise_prior))
            else:
                prev_key, _ = self.keyframes[idx - 1]
                prev_symbol = gtsam.symbol('x', prev_key)
                rel_pose = self.relative_poses[idx - 1]
                graph.add(
                    gtsam.BetweenFactorPose3(prev_symbol, symbol, rel_pose, self.noise_odom)
                )

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
        result = optimizer.optimize()

        last_key, _ = self.keyframes[-1]
        last_symbol = gtsam.symbol('x', last_key)
        optimized_pose = result.atPose3(last_symbol)
        self.last_pose = optimized_pose
        self.current_key += 1

        return optimized_pose.rotation().matrix(), np.array(
            optimized_pose.translation()
        ).reshape(3, 1)

    def get_last_pose(self):
        return self.last_pose
