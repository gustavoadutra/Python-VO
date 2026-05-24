import numpy as np
import cv2
import argparse
import yaml
import logging
import os
import csv

from utils.tools import plot_keypoints
from utils.RSTPHandler import RSTPHandler

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbsoluteScaleComputer
from VO.BundleAdjustment import GTSAMBundleAdjuster
from WO.WheelOdometry import WheelOdometry
from filters.LFK_PW_UV import KalmanFilter
from utils.PlotTrajectory import TrajPlotter, keypoints_plot


def run(args):
    # Initialize variables for Wheel Odometry
    yaw_wo = 0.0
    t_wo = np.zeros((3, 1))
    wo_pose = np.eye(4)

    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.Loader)

    absscale = AbsoluteScaleComputer()

    loader = create_dataloader(config["dataset"])
    detector = create_detector(config["detector"])
    matcher = create_matcher(config["matcher"])

    # Initialize the filter
    filter_obj = None
    if args.filter == "lkf":
        filter_obj = KalmanFilter(config.get("filter", {}))
        filter_obj.initialize()

    # Initialize bundle adjustment if requested
    ba_obj = None
    if args.ba:
        ba_config = config.get("ba", {})
        ba_config["window_size"] = args.ba_window
        # Pass camera calibration from the loader
        ba_config["fx"] = loader.cam.fx
        ba_config["fy"] = loader.cam.fy
        ba_config["cx"] = loader.cam.cx
        ba_config["cy"] = loader.cam.cy
        ba_obj = GTSAMBundleAdjuster(ba_config)

    # Robot and KAIST datasets often have different axis conventions
    is_robot = config["dataset"].get("is_robot", False)
    is_kaist = config["dataset"].get("is_kaist", False)

    traj_plotter = TrajPlotter(is_robot=is_robot)

    # Initialize Wheel Odometry
    wo = None
    if args.encoder:
        print("[INFO] Encoder Flag Detected: Initializing Wheel Odometry...")
        wo = WheelOdometry(config["dataset"])
        print(f"[DEBUG] WO initialized. CSV loaded: {wo.df is not None}")

    fname = args.config.split("/")[-1].split(".")[0]
    log_fopen = open("results/" + fname + ".txt", mode="a")
    
    vo = VisualOdometry(detector, matcher, loader.cam)

    # Initialize RTSP Handler
    rtsp_handler = None
    if args.rtsp:
        rtsp_handler = RSTPHandler(config)

    # Main loop
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()
        
        # Correcting the order of gt_pose for robot datasets 
        if is_robot:
            gt_pose[0], gt_pose[1] = gt_pose[1], gt_pose[0]

        # Wheel Odometry update
        # It's interpolated so no need to worry about missing timestamps
        if wo:
            # Used to synchronize with RTSP frames and velocity from WO
            timestamp = loader.times[i]
            timestamp_prev = loader.times[i - 1] if i > 0 else timestamp
            
            yaw_wo, R_wo, t_wo_raw, w_wo, v_wo = wo.update(
                prev_timestamp=timestamp_prev, 
                cur_timestamp=timestamp
            )
            
            # Correction for robot and kaist datasets
            if is_kaist:
                t_wo[0, 0] = (-t_wo_raw[1]).item()
                t_wo[1, 0] = (t_wo_raw[0]).item()
                t_wo[2, 0] = 0.0
            elif is_robot:
                t_wo[0, 0] = (t_wo_raw[0]).item()
                t_wo[1, 0] = (-t_wo_raw[1]).item()
                t_wo[2, 0] = (t_wo_raw[2]).item()
            else:
                t_wo[0, 0] = 0.0
                t_wo[1, 0] = 0.0
                t_wo[2, 0] = 0.0

            # Needed to create the current scale
            wo_pose[:3, :3] = R_wo
            wo_pose[:3, 3] = t_wo.flatten()

        # Verifies if it's the kitti dataset
        if is_robot or is_kaist:
            current_scale = absscale.update(wo_pose)
        else:
            current_scale = absscale.update(gt_pose)

        # Update Visual Odometry and get the current pose estimation
        R_vo, t_vo, rel_t_vo, rel_r_vo = vo.update(img, absolute_scale=current_scale)

        # Try to optimize the local window with the bundle adjuster
        ba_xyz = None
        if ba_obj and rel_t_vo is not None and rel_r_vo is not None:
            try:
                _, ba_xyz = ba_obj.update(
                    rel_r_vo,
                    rel_t_vo,
                    absolute_pose=(R_vo, t_vo),
                )
            except Exception as e:
                print(f"[BA ERROR] {e}")

        # Logging (Handling None for t_wo)
        wo_log = t_wo if t_wo is not None else np.zeros((3, 1))

        if filter_obj:
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

        # Visualization
        img1 = keypoints_plot(img, vo)
        img2 = traj_plotter.update(
            t_vo, gt_pose[:, 3], wo_xyz=t_wo, filter_xyz=t_filtered, ba_xyz=ba_xyz
        )
        
        cv2.imshow("keypoints", img1)
        trajectory_window = "trajectory_ba" if ba_obj else "trajectory"
        cv2.imshow(trajectory_window, img2)

        # RTSP Visualization
        if rtsp_handler is not None and rtsp_handler.has_rtsp_images():
            rtsp_display, diff_ms, closest_ts = rtsp_handler.get_rtsp_image(timestamp)
            
            if rtsp_display is not None:
                # Draw synchronization info
                rtsp_display = rtsp_handler.draw_sync_info(rtsp_display, diff_ms)
                cv2.imshow("RTSP (Ground Truth Camera)", rtsp_display)

        if cv2.waitKey(10) == 27:
            break
 
    output_image = f"results/{fname}{'_ba' if args.ba else ''}.png"
    cv2.imwrite(output_image, img2)
    log_fopen.close()
    
    # Save errors to CSV with detector and matcher names
    detector_name = config["detector"].get("type", config["detector"].get("name", "unknown"))
    matcher_name = config["matcher"].get("type", config["matcher"].get("name", "unknown"))
    traj_plotter.save_errors_to_csv(config, detector_name=detector_name, matcher_name=matcher_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="python_vo")
    parser.add_argument(
        "--config",
        type=str,
        default="params/kitti_superpoint_supergluematch.yaml",
        help="config file",
    )
    parser.add_argument(
        "--encoder",
        action="store_true",
        help="If set, Wheel Odometry will be used.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        choices=["lkf"],
        default=None,
        help="Filter to use: 'lkf' for Linear Kalman Filter",
    )
    parser.add_argument(
        "--ba",
        action="store_true",
        help="If set, use GTSAM sliding-window bundle adjustment.",
    )
    parser.add_argument(
        "--ba-window",
        type=int,
        default=7,
        help="Sliding window length for GTSAM BA (default: 7 keyframes)",
    )
    parser.add_argument(
        "--rtsp",
        action="store_true",
        help="If set, RTSP images will be displayed.",
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