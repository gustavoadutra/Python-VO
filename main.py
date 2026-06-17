import numpy as np
import cv2
import argparse
import yaml
import logging

from utils.RSTPHandler import RSTPHandler

from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbsoluteScaleComputer
from VO.BundleAdjustment import GTSAMBundleAdjuster
from WO.WheelOdometry import WheelOdometry
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

    vo = VisualOdometry(detector, matcher, loader.cam, enable_pnp=not args.no_pnp)

    # Initialize bundle adjustment if requested and propagate the flag to VO
    ba_obj = None
    if args.ba:
        ba_config = config.get("ba", {})
        ba_config["fx"] = loader.cam.fx
        ba_config["fy"] = loader.cam.fy
        ba_config["cx"] = loader.cam.cx
        ba_config["cy"] = loader.cam.cy
        ba_obj = GTSAMBundleAdjuster(ba_config)
        # Avisa ao VO que deve manter observações e chamar add_observation
        vo.set_ba_active(True)

    # Robot and KAIST datasets often have different axis conventions
    is_robot = config["dataset"].get("is_robot", False)
    is_kaist = config["dataset"].get("is_kaist", False)
    is_cusco = config["dataset"].get("use_direct_position", False)

    traj_plotter = TrajPlotter(is_robot=is_robot)

    # Initialize Wheel Odometry
    wo = None
    if args.encoder:
        print("[INFO] Encoder Flag Detected: Initializing Wheel Odometry...")
        wo = WheelOdometry(config["dataset"])
        print(f"[DEBUG] WO initialized. CSV loaded: {wo.df is not None}")

    fname = args.config.split("/")[-1].split(".")[0]
    
    # Define o sufixo dinâmico baseado nos argumentos passados
    suffix = ""
    if args.ba:
        suffix += "_ba"
    if args.no_pnp:
        suffix += "_nopnp"

    log_filename = f"results/{fname}{suffix}.txt"
    log_fopen = open(log_filename, mode="a")

    # Initialize RTSP Handler
    rtsp_handler = None
    if args.rtsp:
        rtsp_handler = RSTPHandler(config)

    # Main loop
    for i, img in enumerate(loader):
        gt_pose = loader.get_cur_pose()

        # Correcting the order of gt_pose for robot datasets
        if is_cusco:
            gt_pose[0] = -gt_pose[2]
            gt_pose[2] = -gt_pose[1]
        elif is_robot:
            gt_pose[0], gt_pose[1] = gt_pose[1], gt_pose[0]

        # Wheel Odometry update
        if wo:
            timestamp = loader.times[i]
            timestamp_prev = loader.times[i - 1] if i > 0 else timestamp

            yaw_wo, R_wo, t_wo_raw, w_wo, v_wo = wo.update(
                prev_timestamp=timestamp_prev,
                cur_timestamp=timestamp
            )

            if is_kaist:
                t_wo[0, 0] = (-t_wo_raw[1]).item()
                t_wo[1, 0] = (t_wo_raw[0]).item()
                t_wo[2, 0] = 0.0
            elif is_cusco:
                t_wo[0, 0] = (-t_wo_raw[1]).item()
                t_wo[1, 0] = (t_wo_raw[0]).item()
                t_wo[2, 0] = (t_wo_raw[2]).item()
            elif is_robot:
                t_wo[0, 0] = (t_wo_raw[0]).item()
                t_wo[1, 0] = (-t_wo_raw[1]).item()
                t_wo[2, 0] = (t_wo_raw[2]).item()
            else:
                t_wo[0, 0] = 0.0
                t_wo[1, 0] = 0.0
                t_wo[2, 0] = 0.0

            wo_pose[:3, :3] = R_wo
            wo_pose[:3, 3] = t_wo.flatten()

        current_scale = absscale.update(gt_pose)

        # Update Visual Odometry
        R_vo, t_vo, rel_t_vo, rel_r_vo = vo.update(img, absolute_scale=current_scale)

        # Bundle Adjustment (só quando --ba foi passado)
        ba_xyz = None
        if ba_obj and rel_t_vo is not None and rel_r_vo is not None:
            try:
                observations, landmark_initials = vo.get_observations_for_ba()

                _, ba_xyz = ba_obj.update(
                    absolute_pose=(R_vo, t_vo),
                    relative_rotation=rel_r_vo,
                    relative_translation=rel_t_vo,
                    observations=observations,
                    landmark_initials=landmark_initials,
                )
            except Exception as e:
                print(f"[BA ERROR] {e}")

        # Logging
        wo_log = t_wo if t_wo is not None else np.zeros((3, 1))

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
            t_vo, gt_pose[:, 3], wo_xyz=t_wo, ba_xyz=ba_xyz, image=img
        )

        trajectory_window = "trajectory_ba" if ba_obj else "trajectory"
        cv2.imshow(trajectory_window, img2)

        if rtsp_handler is not None and rtsp_handler.has_rtsp_images():
            rtsp_display, diff_ms, closest_ts = rtsp_handler.get_rtsp_image(timestamp)

            if rtsp_display is not None:
                rtsp_display = rtsp_handler.draw_sync_info(rtsp_display, diff_ms)
                cv2.imshow("RTSP (Ground Truth Camera)", rtsp_display)

        if cv2.waitKey(10) == 27:
            break

    # --- NOVO: Usa a variável de sufixo para nomear a imagem também ---
    output_image = f"results/{fname}{suffix}.png"
    cv2.imwrite(output_image, img2)
    log_fopen.close()

    detector_name = config["detector"].get("type", config["detector"].get("name", "unknown"))
    matcher_name = config["matcher"].get("type", config["matcher"].get("name", "unknown"))
    
    # Passando a variável `suffix` para garantir o mesmo padrão de nomenclatura
    traj_plotter.save_errors_to_csv(
        config, 
        detector_name=detector_name, 
        matcher_name=matcher_name, 
        extra_suffix=suffix
    )
    
    traj_plotter.save_positions_to_csv(
        config, 
        detector_name=detector_name, 
        matcher_name=matcher_name, 
        extra_suffix=suffix
    )

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
        "--ba",
        action="store_true",
        help="If set, use GTSAM sliding-window bundle adjustment.",
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
    parser.add_argument(
        "--no-pnp",
        action="store_true",
        help="If set, PnP will be disabled and it will strictly use Essential Matrix.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging._nameToLevel[args.logging])
    run(args)