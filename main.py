import numpy as np
import cv2
import argparse
import yaml
import logging
import os
import csv

from utils.tools import plot_keypoints

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbosluteScaleComputer
from WO.WheelOdometry import WheelOdometry
from filters.LKF import LinearKalmanFilter
from filters.EKF_PW_UV import ExtendedKalmanFilter


def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(
        img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"]
    )


class TrajPlotter(object):
    def __init__(self, is_robot=False):
        self.errors = []
        self.vo_errors = []
        self.wo_errors = []
        self.ekf_errors = []
        self.vo_positions = []
        self.wo_positions = []
        self.ekf_positions = []
        self.gt_positions = []
        self.is_robot = is_robot
        self.w, self.h = 800, 800
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.scale = 0.1 if not is_robot else 200  # Adjust scale for robot datasets
        

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

        # Calculate VO error
        vo_error = np.linalg.norm(est - gt)
        self.errors.append(vo_error)
        self.vo_errors.append(vo_error)
        # Convert to scalars to ensure consistent dimensions
        self.vo_positions.append([float(np.asarray(x).flat[0]), float(np.asarray(z).flat[0])])
        self.gt_positions.append([float(np.asarray(gt_x).flat[0]), float(np.asarray(gt_z).flat[0])])
        avg_error = np.mean(np.array(self.errors))
        avg_vo_error = np.mean(np.array(self.vo_errors))

        # Calculate WO error if available
        avg_wo_error = None
        if wo_xyz is not None:
            wo_est = np.array([wo_xyz[0], wo_xyz[1]]).reshape(2)
            wo_error = np.linalg.norm(wo_est - gt)
            self.wo_errors.append(wo_error)
            self.wo_positions.append([float(np.asarray(wo_xyz[0]).flat[0]), float(np.asarray(wo_xyz[1]).flat[0])])
            avg_wo_error = np.mean(np.array(self.wo_errors))

        # Calculate EKF error if available
        avg_ekf_error = None
        if ekf_xyz is not None:
            ekf_est = np.array([ekf_xyz[0], ekf_xyz[2]]).reshape(2)
            ekf_error = np.linalg.norm(ekf_est - gt)
            self.ekf_errors.append(ekf_error)
            self.ekf_positions.append([float(np.asarray(ekf_xyz[0]).flat[0]), float(np.asarray(ekf_xyz[2]).flat[0])])
            avg_ekf_error = np.mean(np.array(self.ekf_errors))

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

        # Draw error trajectories as lines connecting consecutive error positions
        # VO Error trajectory (Green dotted)
        if len(self.vo_errors) > 1:
            prev_vo = self.vo_positions[-2]
            curr_vo = self.vo_positions[-1]
            prev_gt = self.gt_positions[-2]
            curr_gt = self.gt_positions[-1]
            
            # Draw line from VO to GT (error vector)
            cv2.line(self.traj, 
                    (int(prev_vo[0] * self.scale) + offset_x, 
                     int(prev_vo[1] * self.scale) + offset_y),
                    (int(curr_vo[0] * self.scale) + offset_x, 
                     int(curr_vo[1] * self.scale) + offset_y),
                    (50, 200, 50), 1)  # Darker green for error trajectory

        # WO Error trajectory (Blue dotted)
        if len(self.wo_errors) > 1 and wo_xyz is not None:
            prev_wo = self.wo_positions[-2]
            curr_wo = self.wo_positions[-1]
            cv2.line(self.traj,
                    (int(prev_wo[0] * self.scale) + offset_x,
                     int(prev_wo[1] * self.scale) + offset_y),
                    (int(curr_wo[0] * self.scale) + offset_x,
                     int(curr_wo[1] * self.scale) + offset_y),
                    (150, 100, 50), 1)  # Darker blue for error trajectory

        # EKF Error trajectory (Yellow dotted)
        if len(self.ekf_errors) > 1 and ekf_xyz is not None:
            prev_ekf = self.ekf_positions[-2]
            curr_ekf = self.ekf_positions[-1]
            cv2.line(self.traj,
                    (int(prev_ekf[0] * self.scale) + offset_x,
                     int(prev_ekf[1] * self.scale) + offset_y),
                    (int(curr_ekf[0] * self.scale) + offset_x,
                     int(curr_ekf[1] * self.scale) + offset_y),
                    (150, 150, 100), 1)  # Darker cyan for error trajectory

        # Legend and Text
        cv2.rectangle(self.traj, (10, 20), (600, 120), (0, 0, 0), -1)
        text = "VO Error: %2.4fm" % (avg_vo_error)
        cv2.putText(
            self.traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8
        )
        
        # Display WO error if available
        y_offset = 50
        if avg_wo_error is not None:
            text_wo = "WO Error: %2.4fm" % (avg_wo_error)
            cv2.putText(
                self.traj, text_wo, (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1, 8
            )
            y_offset += 15
        
        # Display EKF error if available
        if avg_ekf_error is not None:
            text_ekf = "EKF Error: %2.4fm" % (avg_ekf_error)
            cv2.putText(
                self.traj, text_ekf, (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, 8
            )

        # Legend Colors
        cv2.putText(
            self.traj, "VO (Green)", (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1
        )
        cv2.putText(
            self.traj, "GT (Red)", (150, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1
        )
        if wo_xyz is not None:
            cv2.putText(
                self.traj,
                "Wheel (Blue)",
                (280, 80),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
            )
        if ekf_xyz is not None:
            cv2.putText(
                self.traj,
                "EKF (Cyan)",
                (400, 80),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 0),
                1,
            )

        return self.traj

    def save_errors_to_csv(self, dataset_name, output_folder="data"):
        """
        Save individual errors to a CSV file.
        :param dataset_name: Name of the dataset
        :param output_folder: Folder to save the CSV (relative to Python-VO directory)
        """
        # Create data folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Create CSV filename
        csv_filename = os.path.join(output_folder, f"{dataset_name}_errors.csv")
        
        # Determine the maximum number of error records
        max_len = max(len(self.vo_errors), len(self.wo_errors), len(self.ekf_errors))
        
        # Write to CSV
        with open(csv_filename, mode='w', newline='') as csv_file:
            fieldnames = ['Frame', 'VO_Error', 'WO_Error', 'EKF_Error']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(max_len):
                row = {
                    'Frame': i,
                    'VO_Error': self.vo_errors[i] if i < len(self.vo_errors) else '',
                    'WO_Error': self.wo_errors[i] if i < len(self.wo_errors) else '',
                    'EKF_Error': self.ekf_errors[i] if i < len(self.ekf_errors) else '',
                }
                writer.writerow(row)
        
        print(f"[INFO] Errors saved to {csv_filename}")


def run(args):
    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.Loader)

    loader = create_dataloader(config["dataset"])
    detector = create_detector(config["detector"])
    matcher = create_matcher(config["matcher"])
    
    # Select filter: LKF or EKF
    if args.filter == "ekf":
        filter_obj = ExtendedKalmanFilter(config.get("filter", {}))
        
    initialized = False

    absscale = AbosluteScaleComputer()

    # Check if this is a robot dataset from config
    is_robot = config["dataset"].get("is_robot", False)
    traj_plotter = TrajPlotter(is_robot=is_robot)

    # Initialize Wheel Odometry only if the flag is True
    wo = None
    if args.encoder:
        print("[INFO] Encoder Flag Detected: Initializing Wheel Odometry...")
        wo = WheelOdometry(config["dataset"])

    fname = args.config.split("/")[-1].split(".")[0]
    log_fopen = open("results/" + fname + ".txt", mode="a")
    
    vo = VisualOdometry(detector, matcher, loader.cam)

    # ========================================================
    # INDEX RTSP IMAGES BEFORE THE LOOP
    # ========================================================
    dataroot = config["dataset"].get("root_path", "") + config["dataset"].get("sequence", "")
    rtsp_dir = os.path.join(dataroot, "rtsp_images_fixed")
    available_rtsp_timestamps = []

    if os.path.exists(rtsp_dir):
        print(f"[INFO] Indexing RTSP images from: {rtsp_dir}")
        for file in os.listdir(rtsp_dir):
            if file.endswith(".png"):
                try:
                    # Extract the nanosecond integer from the filename
                    ts = int(file.replace(".png", ""))
                    available_rtsp_timestamps.append(ts)
                except ValueError:
                    continue
        available_rtsp_timestamps.sort()
        print(f"[INFO] Found {len(available_rtsp_timestamps)} RTSP images.")
    else:
        print(f"[WARNING] RTSP directory not found at {rtsp_dir}")

    # ========================================================
    # SETUP RTSP CAMERA CALIBRATION (For Display Only)
    # ========================================================
    print("[INFO] Initializing RTSP Camera Calibration Maps...")
    rtsp_w, rtsp_h = 1920, 1080  # Real resolution of the RTSP camera

    K_raw = np.array([
        [1078.79585,           0.0,  988.796493],
        [         0.0, 1085.59661,  547.254318],
        [         0.0,           0.0,           1.0]
    ], dtype=np.float64)

    D_raw = np.array([-0.27300998, 0.0579501, 0.0, 0.0, 0.0], dtype=np.float64)

    # alpha=0 crops out the black borders caused by undistorting barrel distortion
    K_new, _ = cv2.getOptimalNewCameraMatrix(K_raw, D_raw, (rtsp_w, rtsp_h), 0, (rtsp_w, rtsp_h))
    map1, map2 = cv2.initUndistortRectifyMap(K_raw, D_raw, None, K_new, (rtsp_w, rtsp_h), cv2.CV_32FC1)

    # ========================================================
    # MAIN LOOP
    # ========================================================
    for i, img in enumerate(loader):
        gt_pose, img_gt = loader.get_cur_pose()
        t_wo = None  # Default if no wheel odometry
        yaw_wo = 0.0 # Default fallback
        
        timestamp = loader.times[i]
        timestamp_prev = loader.times[i - 1] if i > 0 else timestamp

        # 2. Wheel Odometry update
        if wo:
            yaw_wo, R_wo, t_wo_raw, w_wo, v_wo = wo.update(
                prev_timestamp=timestamp_prev, 
                cur_timestamp=timestamp
            )
            
            # Correction for robot and kaist datasets
            t_wo = np.zeros((3, 1))
            if config["dataset"].get("is_kaist", False):
                t_wo[0, 0] = -t_wo_raw[1]  
                t_wo[1, 0] = t_wo_raw[0]  
                t_wo[2, 0] = 0            
            if config["dataset"].get("is_robot", False):
                t_wo[0, 0] = t_wo_raw[0] 
                t_wo[1, 0] = -t_wo_raw[1]            
                t_wo[2, 0] = t_wo_raw[2]  
            
        # Needed to create the current scale
        wo_pose = np.eye(4)
        if t_wo is not None:
            wo_pose[:3, :3] = R_wo
            wo_pose[:3, 3] = t_wo.flatten()

        current_scale = absscale.update(wo_pose)

        # 3. Visual Odometry update
        if is_robot:
            R_vo, t_vo, rm_vo, rr_vo = vo.update(img, absolute_scale=0.01)
        else:
            R_vo, t_vo, rm_vo, rr_vo = vo.update(img, absolute_scale=current_scale)

        # Correcting the order of gt_pose for robot datasets 
        if is_robot:
            gt_pose[0], gt_pose[1] = gt_pose[1], gt_pose[0]

        # 4. Logging (Handling None for t_wo)
        wo_log = t_wo if t_wo is not None else np.zeros((3, 1))

        # initialize filter using first measurement
        if args.filter is not None and not initialized and t_vo is not None:
            filter_obj.initialize()
            initialized = True

        # ========================================================
        # FILTER EXECUTION (VO for Predict, WO for Update)
        # ========================================================
        if args.filter is not None and initialized:
            
            # 1. Predict step uses Visual Odometry (VO)
            filter_obj.predict(t_vo)
            
            # 2. Measurement Update uses Wheel Odometry (WO)
            if wo and t_wo is not None:
                R_filtered, t_filtered = filter_obj.update(t_wo, yaw_wo)
            else:
                # Fallback just to extract the predicted state for plotting
                t_filtered = np.zeros((3, 1))
                x_est, z_est = filter_obj.get_state()
                t_filtered[0, 0] = x_est
                t_filtered[2, 0] = z_est
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

        # ========================================================
        # MATCH AND DISPLAY RTSP IMAGE
        # ========================================================
        if available_rtsp_timestamps:
            # 1. Parse the timestamp as a float first
            raw_ts = float(timestamp)
            
            # 2. Fix the scale (Seconds vs Nanoseconds)
            # If raw_ts is ~1.7e9, it's in seconds. If it's > 1e18, it's in nanoseconds.
            if raw_ts < 1e12: 
                target_ts = int(raw_ts * 1e9) # Convert seconds to nanoseconds
            else:
                target_ts = int(raw_ts)       # Already in nanoseconds

            # 3. Find the closest RTSP timestamp mathematically
            closest_ts = min(available_rtsp_timestamps, key=lambda x: abs(x - target_ts))
            rtsp_path = os.path.join(rtsp_dir, f"{closest_ts}.png")

            if os.path.exists(rtsp_path):
                rtsp_img = cv2.imread(rtsp_path)
                if rtsp_img is not None:
                    
                    # --- NEW: Apply calibration transformation ---
                    rtsp_rectified = cv2.remap(rtsp_img, map1, map2, cv2.INTER_LINEAR)

                    # Resize the FIXED image to fit the screen
                    rtsp_display = cv2.resize(rtsp_rectified, (640, 360))
                    
                    # Calculate time difference in milliseconds for debugging
                    diff_ms = abs(closest_ts - target_ts) / 1e6
                    
                    # Draw the sync difference on the image
                    text = f"Sync Diff: {diff_ms:.2f} ms"
                    color = (0, 255, 0) if diff_ms < 50 else (0, 0, 255) # Green if < 50ms, Red if out of sync
                    cv2.putText(rtsp_display, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) # Black border
                    cv2.putText(rtsp_display, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)     # Colored text

                    cv2.imshow("RTSP (Ground Truth Camera)", rtsp_display)

        if cv2.waitKey(10) == 27:
            break
 
    cv2.imwrite("results/" + fname + ".png", img2)
    log_fopen.close()
    
    # Save errors to CSV
    dataset_name = config["dataset"].get("name", fname)
    traj_plotter.save_errors_to_csv(dataset_name)


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