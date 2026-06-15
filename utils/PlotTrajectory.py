import csv
import os
import cv2
import numpy as np

from .tools import plot_keypoints

def keypoints_plot(img, vo):
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return plot_keypoints(
        img, vo.kptdescs["cur"]["keypoints"], vo.kptdescs["cur"]["scores"]
    )


class TrajPlotter(object):
    def __init__(self, width=800, height=800, is_robot=False):
        self.is_robot = is_robot
        self.w, self.h = width, height
        self.offset_x, self.offset_y = self.w // 2, self.h // 2
        self.scale = 200 if is_robot else 0.75
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # Estrutura simplificada usando dicionários
        self.positions = {'gt': [], 'vo': [], 'wo': [], 'ba': []}
        self.errors = {'vo': [], 'wo': [], 'ba': []}
        
        # Configuração de Estilos: (Cor do Ponto, Cor da Linha, Rótulo)
        self.styles = {
            'gt': ((0, 0, 255), (0, 0, 255), "GT (Red)"),
            'vo': ((0, 255, 0), (50, 200, 50), "VO (Green)"),
            'wo': ((255, 0, 0), (150, 100, 50), "Wheel (Blue)"),
            'ba': ((255, 0, 255), (255, 0, 255), "BA (Magenta)")
        }

    def _get_xz(self, xyz, is_wo=False):
        """Método auxiliar para extrair coordenadas 2D uniformemente."""
        if xyz is None:
            return None
        # O Wheel Odometry no seu código original usava os índices 0 e 1 (x, y)
        idx_2 = 1 if is_wo else 2
        return np.array([
            float(np.asarray(xyz[0]).flat[0]), 
            float(np.asarray(xyz[idx_2]).flat[0])
        ])

    def update(self, est_xyz, gt_xyz, wo_xyz=None, ba_xyz=None, image=None):
        """Atualiza a plotagem e calcula erros instantâneos."""
        
        # Agrupa os pontos atuais (Extraindo X e Z de forma segura)
        current_pts = {
            'gt': self._get_xz(gt_xyz),
            'vo': self._get_xz(est_xyz),
            'wo': self._get_xz(wo_xyz, is_wo=True),
            'ba': self._get_xz(ba_xyz)
        }

        # Atualiza o histórico de posições e calcula o erro pontual em relação ao GT
        gt_pt = current_pts['gt']
        for key, pt in current_pts.items():
            if pt is not None:
                self.positions[key].append(pt)
                if key != 'gt':
                    self.errors[key].append(np.linalg.norm(pt - gt_pt))

        # Desenha a trajetória
        for key, pt in current_pts.items():
            if pt is None:
                continue
                
            pt_color, line_color, _ = self.styles[key]
            
            # Coordenadas de desenho na tela
            draw_x = int(pt[0] * self.scale) + self.offset_x
            draw_y = int(pt[1] * self.scale) + self.offset_y
            
            # Desenha o ponto atual
            cv2.circle(self.traj, (draw_x, draw_y), 1, pt_color, 1)

            # Desenha a linha conectando ao frame anterior
            if len(self.positions[key]) > 1:
                prev_pt = self.positions[key][-2]
                prev_x = int(prev_pt[0] * self.scale) + self.offset_x
                prev_y = int(prev_pt[1] * self.scale) + self.offset_y
                cv2.line(self.traj, (prev_x, prev_y), (draw_x, draw_y), line_color, 1)

        # UI e Legendas
        cv2.rectangle(self.traj, (10, 20), (600, 100), (0, 0, 0), -1)
        #cv2.putText(self.traj, self.styles['gt'][2], (280, 40), cv2.FONT_HERSHEY_PLAIN, 1, self.styles['gt'][0], 1)
        
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
        # Apenas mantive a sua lógica inalterada
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

    def calculate_metrics(self, align_trajectories=True):
        """
        Calcula o ATE e RPE. Se align_trajectories=True, aplica o Umeyama 
        antes de computar as métricas para remover offsets globais de Referencial/Escala.
        """
        metrics = {}
        gt = np.array(self.positions['gt'])
        
        for key in ['vo', 'wo', 'ba']:
            est = np.array(self.positions[key])
            
            if len(est) < 2 or len(est) != len(gt):
                continue
                
            # Alinha a estimativa ao GT usando Umeyama
            if align_trajectories:
                # with_scale=True é fundamental para Odometria Monocular
                est_aligned = self.align_umeyama(est, gt, with_scale=True) 
            else:
                est_aligned = est
                
            # 1. ATE: RMSE das distâncias de translação globais (Est_Alinhada - GT)
            ate = np.sqrt(np.mean(np.sum((est_aligned - gt)**2, axis=1)))
            
            # 2. RPE: RMSE das translações relativas (Frame a Frame)
            est_delta = est_aligned[1:] - est_aligned[:-1]
            gt_delta = gt[1:] - gt[:-1]
            rpe = np.sqrt(np.mean(np.sum((est_delta - gt_delta)**2, axis=1)))
            
            metrics[key] = {'ATE': ate, 'RPE': rpe}
            
        return metrics
    def save_errors_to_csv(self, config, detector_name="", matcher_name="", output_folder="data"):
        sequence = config['dataset'].get('sequence', 'unknown')
        seq_folder = os.path.join(output_folder, sequence)
        os.makedirs(seq_folder, exist_ok=True)
        
        suffix = f"{detector_name}_{matcher_name}" if detector_name and matcher_name else ""
        base_name = f"{suffix}_errors.csv" if suffix else "errors.csv"
        csv_filename = os.path.join(seq_folder, base_name)
        
        # Encontra o limite de frames processados
        max_len = max([len(v) for v in self.errors.values()] + [0])
        
        # Calcula as métricas globais antes de salvar
        metrics = self.calculate_metrics()
        
        with open(csv_filename, mode='w', newline='') as csv_file:
            if detector_name or matcher_name:
                csv_file.write(f"# Detector: {detector_name}\n")
                csv_file.write(f"# Matcher: {matcher_name}\n")
            
            # Adiciona os resultados finais do ATE e RPE no cabeçalho
            for key, res in metrics.items():
                csv_file.write(f"# {key.upper()} - ATE: {res['ATE']:.4f} | RPE: {res['RPE']:.4f}\n")
            
            fieldnames = ['Frame', 'VO_Error', 'WO_Error', 'BA_Error']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(max_len):
                writer.writerow({
                    'Frame': i,
                    'VO_Error': self.errors['vo'][i] if i < len(self.errors['vo']) else '',
                    'WO_Error': self.errors['wo'][i] if i < len(self.errors['wo']) else '',
                    'BA_Error': self.errors['ba'][i] if i < len(self.errors['ba']) else '',
                })
        
        print(f"[INFO] Errors saved to {csv_filename}")
        for key, res in metrics.items():
            print(f"[INFO] {key.upper()} Final Metrics -> ATE: {res['ATE']:.4f}m, RPE: {res['RPE']:.4f}m")

    def align_umeyama(self, model, data, with_scale=True):
        """
        Alinha a trajetória estimada (model) à trajetória real (data) 
        usando o algoritmo de Umeyama.
        """
        # Centraliza os pontos nas origens de seus respectivos referenciais
        mu_M = model.mean(axis=0)
        mu_D = data.mean(axis=0)

        model_zero = model - mu_M
        data_zero = data - mu_D

        # Matriz de covariância
        C = (model_zero.T @ data_zero) / model.shape[0]

        # Decomposição em Valores Singulares (SVD)
        U, S, Vt = np.linalg.svd(C)
        V = Vt.T

        # Matriz para lidar com o problema de reflexão
        d = np.linalg.det(V @ U.T)
        D = np.eye(model.shape[1])
        if d < 0:
            D[-1, -1] = -1

        # Rotação ótima (R)
        R = V @ D @ U.T

        # Escala ótima (c)
        if with_scale:
            var_M = np.var(model_zero, axis=0).sum()
            scale = (1.0 / var_M) * np.sum(S) if var_M > 0 else 1.0
        else:
            scale = 1.0

        # Translação ótima (t)
        t = mu_D - scale * (R @ mu_M)

        # Aplica as transformações na trajetória original
        model_aligned = scale * (model @ R.T) + t

        return model_aligned