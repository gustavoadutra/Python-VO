import numpy as np
import cv2
import argparse
import yaml
import time
import os
import csv

from utils.RSTPHandler import RSTPHandler
from DataLoader import create_dataloader
from Detectors import create_detector
from Matchers import create_matcher
from VO.VisualOdometry import VisualOdometry, AbsoluteScaleComputer
from VO.BundleAdjustment import GTSAMBundleAdjuster
from WO.WheelOdometry import WheelOdometry
from utils.PlotTrajectory import TrajPlotter, keypoints_plot

# ==============================================================================
# CLASSE WRAPPER PARA PROFILING DE TEMPO
# Intercepta as chamadas do detector e matcher para medir o tempo sem alterar o VO
# ==============================================================================
class ProfilingWrapper:
    def __init__(self, obj):
        self._obj = obj
        self.total_time = 0.0

    # Intercepta chamadas de métodos normais (ex: obj.metodo())
    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if callable(attr):
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                result = attr(*args, **kwargs)
                self.total_time += time.perf_counter() - start
                return result
            return wrapper
        return attr

    # Intercepta chamadas diretas ao objeto (ex: obj())
    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = self._obj(*args, **kwargs)  # Repassa a chamada para o objeto base
        self.total_time += time.perf_counter() - start
        return result


def run(args):
    # Initialize variables for Wheel Odometry
    t_wo = np.zeros((3, 1))
    wo_pose = np.eye(4)

    with open(args.config, "r") as f:
        config = yaml.load(f, yaml.Loader)

    absscale = AbsoluteScaleComputer()

    loader = create_dataloader(config["dataset"])
    
    # Criamos os objetos base
    base_detector = create_detector(config["detector"])
    base_matcher = create_matcher(config["matcher"])
    
    # Envolvemos com a classe que mede o tempo
    detector = ProfilingWrapper(base_detector)
    matcher = ProfilingWrapper(base_matcher)
    
    pnp_config = config['pnp']

    vo = VisualOdometry(
        detector,
        matcher,
        loader.cam,
        enable_pnp=not args.no_pnp, 
        config=pnp_config)

    # Initialize bundle adjustment if requested and propagate the flag to VO
    ba_obj = None
    if args.ba:
        ba_config = config.get("ba", {})
        print(ba_config)
        ba_obj = GTSAMBundleAdjuster(loader.cam, ba_config)
        vo.set_ba_active(True)

    # Robot and KAIST datasets often have different axis conventions
    is_robot = config["dataset"].get("is_robot", False)
    is_kaist = config["dataset"].get("is_kaist", False)
    is_cusco = config["dataset"].get("use_direct_position", False)
    config_scale = config["dataset"].get("scale", 0.75)

    traj_plotter = TrajPlotter(scale=config_scale)

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

    # Initialize RTSP Handler
    rtsp_handler = None
    if args.rtsp:
        rtsp_handler = RSTPHandler(config)

    frames_processados = 0

    # Main loop
    for i, img in enumerate(loader):
        # LIMITADOR DE FRAMES (AMOSTRAGEM)
        if args.max_frames > 0 and i >= args.max_frames:
            print(f"\n[INFO] Limite de {args.max_frames} frames atingido. Encerrando amostragem...")
            break
            
        frames_processados += 1
        gt_pose = loader.get_cur_pose()

        # Correcting the order of gt_pose for robot datasets
        t_gt_orig = gt_pose[:3, 3].copy()
        if is_cusco:
            gt_pose[0, 3] = t_gt_orig[2]
            gt_pose[1, 3] = 0
            gt_pose[2, 3] = t_gt_orig[1]
        elif is_robot:
            gt_pose[0, 3] = t_gt_orig[1]
            gt_pose[1, 3] = 0
            gt_pose[2, 3] = t_gt_orig[0]
        elif is_kaist:
            gt_pose[0, 3] = -t_gt_orig[0]
            gt_pose[1, 3] = 0
            gt_pose[2, 3] = t_gt_orig[2]

        timestamp = loader.times[i]
        timestamp_prev = loader.times[i - 1] if i > 0 else timestamp

        # Wheel Odometry update
        if wo:
            yaw_wo, R_wo, t_wo_raw, w_wo, v_wo = wo.update(
                prev_timestamp=timestamp_prev,
                cur_timestamp=timestamp
            )
            if is_kaist:
                t_wo[0, 0] = (t_wo_raw[1]).item()
                t_wo[1, 0] = 0
                t_wo[2, 0] = (t_wo_raw[0]).item()
            elif is_cusco:
                t_wo[0, 0] = (-t_wo_raw[1]).item()
                t_wo[1, 0] = 0
                t_wo[2, 0] = (t_wo_raw[0]).item()
            elif is_robot:
                t_wo[0, 0] = (t_wo_raw[0]).item()
                t_wo[1, 0] = 0
                t_wo[2, 0] = (t_wo_raw[1]).item()
            else:
                t_wo[0, 0] = 0.0
                t_wo[1, 0] = 0.0
                t_wo[2, 0] = 0.0

            wo_pose[:3, :3] = R_wo
            wo_pose[:3, 3] = t_wo.flatten()

        if is_cusco:
            current_scale = 0.01
        else:
            current_scale = 1

        # Update Visual Odometry
        R_vo, t_vo, rel_t_vo, rel_r_vo = vo.update(img, current_scale)

        # Bundle Adjustment
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

        # Visualization
        img1 = keypoints_plot(img, vo)
        img2 = traj_plotter.update(
            timestamp=timestamp, 
            R_vo=R_vo, 
            t_vo=t_vo, 
            gt_pose=gt_pose, 
            wo_pose=wo_pose if wo else None, 
            ba_xyz=ba_xyz, 
            image=img
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

    # Identificação externa do Extrator e Matcher obtida das configurações
    detector_name = config["detector"].get("type", config["detector"].get("name", "unknown"))
    matcher_name = config["matcher"].get("type", config["matcher"].get("name", "unknown"))

    # ==============================================================================
    # PRINT DOS RESULTADOS DE TEMPO & SALVAMENTO EM LOG (CSV)
    # ==============================================================================
    if frames_processados > 0:
        avg_extract = detector.total_time / frames_processados
        avg_match = matcher.total_time / frames_processados
        total_sum = avg_extract + avg_match

        print("\n" + "="*55)
        print("📊 RELATÓRIO DE TEMPO MÉDIO POR FRAME (Amostragem)")
        print("="*55)
        print(f"🔹 Dataset/Config:          {fname}")
        print(f"🔹 Frames Processados:      {frames_processados}")
        print(f"🔹 Extração de Descritor:   {avg_extract:.5f} segundos")
        print(f"🔹 Match:                   {avg_match:.5f} segundos")
        print("-" * 55)
        print(f"🚀 SOMA TOTAL (Ext + Match): {total_sum:.5f} segundos")
        print("="*55 + "\n")

        # Configuração do arquivo de Log
        log_dir = "results"
        log_file = os.path.join(log_dir, "benchmark_log.csv")
        os.makedirs(log_dir, exist_ok=True)

        # 1. Verifica se a combinação (Dataset, Detector, Matcher) já existe no arquivo
        combination_exists = False
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            with open(log_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None) # Pula o cabeçalho se houver
                for row in reader:
                    if len(row) >= 3:
                        # row[0]: Dataset, row[1]: Detector, row[2]: Matcher
                        if row[0] == fname and row[1] == detector_name and row[2] == matcher_name:
                            combination_exists = True
                            break

        # 2. Se não existir, faz o append dos resultados
        if not combination_exists:
            file_is_empty = not os.path.exists(log_file) or os.path.getsize(log_file) == 0
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if file_is_empty:
                    # Escreve o cabeçalho caso o arquivo seja novo
                    writer.writerow(["Dataset_Config", "Detector", "Matcher", "Avg_Extract_Sec", "Avg_Match_Sec", "Total_Sum_Sec"])
                
                writer.writerow([fname, detector_name, matcher_name, f"{avg_extract:.5f}", f"{avg_match:.5f}", f"{total_sum:.5f}"])
            print(f"[INFO] Nova combinação detectada! Resultados salvos em '{log_file}'")
        else:
            print(f"[INFO] A combinação ({fname} + {detector_name} + {matcher_name}) já consta no log. Ignorando append.")

    output_image = f"results/{fname}{suffix}.png"
    cv2.imwrite(output_image, img2)

    traj_plotter.save_trajectories_tum(
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
        "--no-pnp",
        action="store_true",
        help="If set, PnP will be disabled and it will strictly use Essential Matrix.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=50, 
        help="Limita o número de frames para cálculo de amostragem de tempo (0 para rodar tudo).",
    )

    args = parser.parse_args()
    run(args)