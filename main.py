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
from WO.WheelOdometry import WheelOdometry  # Ensure this import works


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(
        img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"]
    )


class TrajPlotter(object):
    def __init__(self):
        self.errors = []
        self.traj = np.zeros((600, 600, 3), dtype=np.uint8)

    def update(self, est_xyz, gt_xyz, wo_xyz=None):
        """
        Updates the trajectory plot.
        :param est_xyz: Visual Odometry position
        :param gt_xyz: Ground Truth position
        :param wo_xyz: Wheel Odometry position (Optional)
        """
        x, z = est_xyz[0], est_xyz[2]
        gt_x, gt_z = gt_xyz[0], gt_xyz[2]

        est = np.array([x, z]).reshape(2)
        gt = np.array([gt_x, gt_z]).reshape(2)

        error = np.linalg.norm(est - gt)
        self.errors.append(error)
        avg_error = np.mean(np.array(self.errors))

        # === Coordinate Transformation for Drawing ===
        # Centering the drawing on the canvas (Adjust 290/90 if needed)
        offset_x, offset_y = 290, 90

        draw_x, draw_y = int(x) + offset_x, int(z) + offset_y
        true_x, true_y = int(gt_x) + offset_x, int(gt_z) + offset_y

        # Draw Visual Odometry (Green)
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)

        # Draw Ground Truth (Red)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 1)

        # Draw Wheel Odometry (Blue) - if available
        if wo_xyz is not None:
            wo_x, wo_z = int(wo_xyz[0]) + offset_x, int(wo_xyz[2]) + offset_y
            cv2.circle(self.traj, (wo_x, wo_z), 1, (255, 0, 0), 1)

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

        return self.traj


def run(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.Loader)

    # create dataloader
    loader = create_dataloader(config["dataset"])
    # create detector
    detector = create_detector(config["detector"])
    # create matcher
    matcher = create_matcher(config["matcher"])

    # Initialize generic scale computer (fallback)
    absscale = AbosluteScaleComputer()

    # Initialize Plotter
    traj_plotter = TrajPlotter()

    # === Wheel Odometry Setup ===
    use_wheel_odometry = False
    wo = None

    # We check if an encoder CSV was provided. If so, we assume KAIST workflow.
    if args.encoder_csv:
        print("[INFO] KAIST Workflow Detected: Initializing Wheel Odometry...")
        # KAIST Prius Parameters: Radius ~0.315m, Baseline ~1.58m, Ticks ~8192
        wo = WheelOdometry(wheel_radius=0.315, base_line=1.58, ticks_per_rev=8192)
        wo.load_kaist_csv(args.encoder_csv)
        use_wheel_odometry = True
    else:
        print("[INFO] No encoder CSV found. Using Ground Truth for scale (KITTI Mode).")

    # log file
    fname = args.config.split("/")[-1].split(".")[0]
    log_fopen = open("results/" + fname + ".txt", mode="a")

    vo = VisualOdometry(detector, matcher, loader.cam)

    prev_wo_t = None

    # Main Loop
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()

        current_scale = 1.0
        wo_position = None

        # === 1. Calculate Scale & Position from Wheel Odometry ===
        if use_wheel_odometry:
            # We need the timestamp. Assuming loader has .times list (standard for these datasets)
            try:
                timestamp = loader.times[i]

                # Get interpolated ticks
                l_tick, r_tick = wo.get_interpolated_ticks(timestamp)

                # Update WO
                R_wo, t_wo = wo.update(l_tick, r_tick)
                wo_position = t_wo  # For plotting

                # Calculate Scale (Distance moved since last frame)
                if i == 0:
                    current_scale = 0.1  # robust initialization
                    prev_wo_t = t_wo.copy()
                else:
                    current_scale = np.linalg.norm(t_wo - prev_wo_t)
                    prev_wo_t = t_wo.copy()

            except AttributeError:
                print("[ERROR] Loader does not have 'times' attribute needed for sync.")
                break
        else:
            # Fallback to GT scale (Original Logic)
            current_scale = absscale.update(gt_pose)

        # === 2. Update Visual Odometry ===
        # If using WO, we pass the wheel-derived scale.
        # If not, we pass the GT-derived scale.
        R, t = vo.update(img, absolute_scale=current_scale)

        # === log writer ==============================
        print(
            i,
            t[0, 0],
            t[1, 0],
            t[2, 0],
            gt_pose[0, 3],
            gt_pose[1, 3],
            gt_pose[2, 3],
            file=log_fopen,
        )

        # === drawer ==================================
        img1 = keypoints_plot(img, vo)

        # Pass WO position to plotter if it exists
        img2 = traj_plotter.update(t, gt_pose[:, 3], wo_xyz=wo_position)

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
    # Added argument specifically for KAIST wheel data
    parser.add_argument(
        "--encoder_csv",
        type=str,
        default=None,
        help="Path to KAIST encoder.csv file. If provided, enables Wheel Odometry.",
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
