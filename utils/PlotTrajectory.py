import csv
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy

from .tools import plot_keypoints

def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(
        img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"]
    )

class TrajPlotter(object):
    def __init__(self, width=800, height=800, scale=1):
        self.w, self.h = width, height
        self.offset_x, self.offset_y = self.w // 2, self.h // 2
        self.scale = scale
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Armazena as Poses no formato TUM
        self.poses_tum = {'gt': [], 'vo': [], 'wo': [], 'ba': []}
        
        self.positions = {'gt': [], 'vo': [], 'wo': [], 'ba': []}
        self.errors = {'vo': [], 'wo': [], 'ba': []}
        
        self.styles = {
            'gt': ((0, 0, 255), (0, 0, 255), "GT (Red)"),
            'vo': ((0, 255, 0), (50, 200, 50), "VO (Green)"),
            'wo': ((255, 0, 0), (150, 100, 50), "Wheel (Blue)"),
            'ba': ((255, 0, 255), (255, 0, 255), "BA (Magenta)")
        }

    def _get_xz(self, xyz):
        if xyz is None:
            return None
        return np.array([
            float(np.asarray(xyz[0]).flat[0]), 
            float(np.asarray(xyz[2]).flat[0])
        ])

    # --- NOVO: Conversão para o formato TUM ---
    def _format_tum_pose(self, timestamp, R, t):
        """Converte timestamp, R (3x3) e t (3x1) para a lista do TUM [timestamp, x, y, z, qx, qy, qz, qw]"""
        if R is None or t is None or timestamp is None:
            return None
        
        # Garante que a translação seja 1D, convertendo arrays numpy para escalares
        tx = float(np.asarray(t[0]).flat[0])
        ty = float(np.asarray(t[1]).flat[0])
        tz = float(np.asarray(t[2]).flat[0])
        
        # Converte a matriz de rotação 3x3 para quatérnio (Scipy retorna no formato x, y, z, w por padrão)
        r_quat = R_scipy.from_matrix(R).as_quat()
        qx, qy, qz, qw = r_quat
        
        return [timestamp, tx, ty, tz, qx, qy, qz, qw]

    def update(self, timestamp, R_vo, t_vo, gt_pose, wo_pose=None, ba_xyz=None, image=None):
        
        # 1. Armazenar dados no formato TUM
        if gt_pose is not None and timestamp is not None:
            R_gt = gt_pose[:3, :3]
            t_gt = gt_pose[:3, 3]
            self.poses_tum['gt'].append(self._format_tum_pose(timestamp, R_gt, t_gt))
            
        if R_vo is not None and t_vo is not None and timestamp is not None:
            self.poses_tum['vo'].append(self._format_tum_pose(timestamp, R_vo, t_vo))
            
        if wo_pose is not None and timestamp is not None:
            R_wo = wo_pose[:3, :3]
            t_wo = wo_pose[:3, 3]
            self.poses_tum['wo'].append(self._format_tum_pose(timestamp, R_wo, t_wo))
            
        if ba_xyz is not None and timestamp is not None:
            # Assume identidade para a rotação se o BA retornar apenas posições 3D
            self.poses_tum['ba'].append(self._format_tum_pose(timestamp, np.eye(3), ba_xyz))

        # 2. Lógica existente para plotagem 2D (invariada)
        gt_xyz = gt_pose[:3, 3] if gt_pose is not None else None
        wo_xyz = wo_pose[:3, 3] if wo_pose is not None else None

        current_pts = {
            'gt': self._get_xz(gt_xyz),
            'vo': self._get_xz(t_vo),
            'wo': self._get_xz(wo_xyz),
            'ba': self._get_xz(ba_xyz)
        }

        gt_pt = current_pts['gt']
        for key, pt in current_pts.items():
            if pt is not None:
                self.positions[key].append(pt)
                if key != 'gt' and gt_pt is not None:
                    self.errors[key].append(np.linalg.norm(pt - gt_pt))

        for key, pt in current_pts.items():
            if pt is None:
                continue
            pt_color, line_color, _ = self.styles[key]
            draw_x = int(pt[0] * self.scale) + self.offset_x
            draw_y = int(pt[1] * self.scale) + self.offset_y
            cv2.circle(self.traj, (draw_x, draw_y), 1, pt_color, 1)

            if len(self.positions[key]) > 1:
                prev_pt = self.positions[key][-2]
                prev_x = int(prev_pt[0] * self.scale) + self.offset_x
                prev_y = int(prev_pt[1] * self.scale) + self.offset_y
                cv2.line(self.traj, (prev_x, prev_y), (draw_x, draw_y), line_color, 1)

        cv2.rectangle(self.traj, (10, 20), (600, 100), (0, 0, 0), -1)
        legend_y = 40
        for key in ['vo', 'wo', 'ba']:
            if len(self.errors[key]) > 0:
                avg_err = np.mean(self.errors[key])
                text = f"{self.styles[key][2]} - {avg_err:.4f}m"
                cv2.putText(self.traj, text, (20, legend_y), cv2.FONT_HERSHEY_PLAIN, 1, self.styles[key][0], 1)
                legend_y += 15

        if image is not None:
            return self._combine_image_and_trajectory(image)
        return self.traj

    def _combine_image_and_trajectory(self, image):
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        img_h, img_w = image.shape[:2]
        traj_h, traj_w = self.traj.shape[:2]
        if img_h < traj_h:
            pad_top = (traj_h - img_h) // 2
            pad_bottom = traj_h - img_h - pad_top
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        elif img_h > traj_h:
            image = cv2.resize(image, (img_w, traj_h))
        return np.hstack([image, self.traj])

    # Função que salva todas as trajetórias em formato TUM 
    def save_trajectories_tum(self, config, detector_name="", matcher_name="", extra_suffix="", output_folder="results"):
        sequence = config['dataset'].get('sequence', 'unknown')
        seq_folder = os.path.join(output_folder, sequence)
        os.makedirs(seq_folder, exist_ok=True)
        
        name_prefix = f"{detector_name}_{matcher_name}" if detector_name and matcher_name else ""
        
        for key in ['gt', 'vo', 'wo', 'ba']:
            if not self.poses_tum[key]:
                continue
                
            base_name = f"{sequence}_{key}_{name_prefix}{extra_suffix}.txt" if name_prefix else f"{sequence}_{key}{extra_suffix}.txt"
            txt_filename = os.path.join(seq_folder, base_name)
            
            with open(txt_filename, mode='w') as txt_file:
                for pose in self.poses_tum[key]:
                    # Formato TUM: timestamp x y z qx qy qz qw
                    line = f"{pose[0]:.6f} {pose[1]:.6e} {pose[2]:.6e} {pose[3]:.6e} {pose[4]:.6e} {pose[5]:.6e} {pose[6]:.6e} {pose[7]:.6e}"
                    txt_file.write(line + "\n")
                    
            print(f"[INFO] Trajetória {key.upper()} salva no formato TUM em: {txt_filename}")