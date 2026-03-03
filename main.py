import numpy as np
import cv2
import argparse
import yaml
import logging

from utils.tools import plot_keypoints

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer
from WO.WheelOdometry import WheelOdometry
from filters.LKF import LinearKalmanFilter
from filters.EKF import ExtendedKalmanFilter


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(
        img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"]
    )


class TrajPlotter(object):
    def __init__(self, is_robot=False):
        self.errors = []
        self.w, self.h = 800, 800
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.scale = 0.1 if not is_robot else 100  # Adjust scale for robot datasets
        

    def update(self, est_xyz, gt_xyz, wo_xyz=None, ekf_xyz=None):
        """
        Updates the trajectory plot.
        :param est_xyz: Visual Odometry position
        :param gt_xyz: Ground Truth position
        :param wo_xyz: Wheel Odometry position (Optional)
        :param ekf_xyz: EKF position (Optional)
        """
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]

        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)

        error = np.linalg.norm(est - gt)
        self.errors.append(error)
        avg_error = np.mean(np.array(self.errors))

        # Offset: Centers the start point.
        offset_x = self.w // 2
        offset_y = self.h // 2
        draw_x = int((x * self.scale).item()) + offset_x
        draw_y = int((z * self.scale).item()) + offset_y

        true_x = int((gt_x * self.scale).item()) + offset_x
        true_y = int((gt_z * self.scale).item()) + offset_y
        # Draw Visual Odometry (Green)
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)

        # Draw Ground Truth (Red)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 1)

        # Draw Wheel Odometry (Blue) - if available
        if wo_xyz is not None:
            wo_x, wo_z = (
                int(wo_xyz[0] * self.scale) + offset_x,
                int(wo_xyz[1] * self.scale) + offset_y,
            )
            cv2.circle(self.traj, (wo_x, wo_z), 1, (255, 0, 0), 1)

        if ekf_xyz is not None:
            ekf_x, ekf_z = (
                int(ekf_xyz[0] * self.scale) + offset_x,
                int(ekf_xyz[2] * self.scale) + offset_y,
            )
            cv2.circle(self.traj, (ekf_x, ekf_z), 1, (255, 255, 0), 1)

        # Legend and Text
        cv2.rectangle(self.traj, (10, 20), (600, 80), (0, 0, 0), -1)
        text = "AvgError: %2.4fm" % (avg_error)
        cv2.putText(
            self.traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8
        )

        # Legend Colors
        cv2.putText(
            self.traj, "VO (Green)", (20, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1
        )
        cv2.putText(
            self.traj, "GT (Red)", (150, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1
        )
        if wo_xyz is not None:
            cv2.putText(
                self.traj,
                "Wheel (Blue)",
                (250, 60),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
            )
        if ekf_xyz is not None:
            cv2.putText(
                self.traj,
                "EKF (Cyan)",
                (400, 60),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 0),
                1,
            )

        return self.traj


def run(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.Loader)

    loader = create_dataloader(config["dataset"])
    detector = create_detector(config["detector"])
    matcher = create_matcher(config["matcher"])
    
    # Select filter: LKF or EKF
    if args.filter == "lkf":
        filter_obj = LinearKalmanFilter(config.get("filter", {}))
    elif args.filter == "ekf":
        filter_obj = ExtendedKalmanFilter(config.get("filter", {}))
        
    initialized = False

    absscale = AbosluteScaleComputer()

    # Check if this is a robot dataset from config
    is_robot = config["dataset"].get("is_robot", False)
    scale_factor = config["dataset"].get("scale_factor", 50)
    traj_plotter = TrajPlotter(is_robot=is_robot)
    traj_plotter.scale = scale_factor

    # Initialize Wheel Odometry only if the flag is True
    wo = None
    if args.encoder:
        print("[INFO] Encoder Flag Detected: Initializing Wheel Odometry...")
        wo = WheelOdometry(config["dataset"])

    fname = args.config.split("/")[-1].split(".")[0]
    log_fopen = open("results/" + fname + ".txt", mode="a")
    vo = VisualOdometry(detector, matcher, loader.cam)

    for i, img in enumerate(loader):
        gt_pose, img_gt = loader.get_cur_pose()
        t_wo = None  # Default if no wheel odometry

        # 2. Wheel Odometry update
        if wo:
            timestamp = loader.times[i]
            l_tick, r_tick = wo.get_interpolated_ticks(timestamp)
            yaw_wo, R_wo, t_wo_raw = wo.update(l_tick, r_tick)
            
            # Correction for robot and kaist datasets
            t_wo = np.zeros((3, 1))
            if config["dataset"].get("is_kaist", False):
                t_wo[0, 0] = -t_wo_raw[1]  
                t_wo[1, 0] = t_wo_raw[0]  
                t_wo[2, 0] = 0            
            if config["dataset"].get("is_robot", False):
                t_wo[0, 0] = -t_wo_raw[1] 
                t_wo[1, 0] = -t_wo_raw[0]            
                t_wo[2, 0] = t_wo_raw[2]  
            
        # Needed to create the current scale
        wo_pose = np.eye(4)
        if t_wo is not None:
            wo_pose[:3, :3] = R_wo
            wo_pose[:3, 3] = t_wo.flatten()

        current_scale = absscale.update(wo_pose)

        # 3. Visual Odometry update
        R_vo, t_vo = vo.update(img, absolute_scale=0.01)

        # Correcting the order of gt_pose for robot datasets 
        if is_robot:
            gt_pose[0], gt_pose[1] = gt_pose[1], gt_pose[0]

        # 4. Logging (Handling None for t_wo)
        # We use a fallback [0,0,0] if t_wo is None for consistent column count
        wo_log = t_wo if t_wo is not None else np.zeros((3, 1))
        
        if args.filter is not None:
            R_filtered, t_filtered = filter_obj.update(t_vo, wo_log)
        else:
            t_filtered = None

        print(
            i,
            t_vo[0, 0],
            t_vo[1, 0],
            t_vo[2, 0],
            gt_pose[0, 3],
            gt_pose[1, 3],
            gt_pose[2, 3],
            wo_log[0, 0],
            wo_log[1, 0],
            wo_log[2, 0],
            file=log_fopen,
        )

        # 5. Visualization
        img1 = keypoints_plot(img, vo)
        img2 = traj_plotter.update(t_vo, gt_pose[:, 3], wo_xyz=t_wo, ekf_xyz=t_filtered)

        cv2.imshow("keypoints", img1)
        cv2.imshow("trajectory", img2)
        if cv2.waitKey(10) == 27:
            break
 
    cv2.imwrite("results/" + fname + ".png", img2)
    log_fopen.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python_vo")
    parser.add_argument(
        "--config",
        type=str,
        default="params/kitti_superpoint_supergluematch.yaml",
        help="config file",
    )
    # Changed to optional flags (store_true)
    parser.add_argument(
        "--encoder",
        action="store_true",
        help="If set, Wheel Odometry will be used.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["lkf", "ekf"],
        default=None,
        help="Filter to use: 'lkf' for Linear Kalman Filter, 'ekf' for Extended Kalman Filter",
    )
    parser.add_argument(
        "--logging",
        type=str,
        default="INFO",
        help="logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging._nameToLevel[args.logging])
    run(args)
