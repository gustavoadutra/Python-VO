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
        self.errors = []
        self.vo_errors = []
        self.wo_errors = []
        self.ekf_errors = []
        self.vo_positions = []
        self.wo_positions = []
        self.ekf_positions = []
        self.gt_positions = []
        self.is_robot = is_robot
        self.w, self.h = width, height
        self.offset_x = self.w // 2
        self.offset_y = self.h // 2
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

        # Creates 2D points for error calculation
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

        draw_x = int((x * self.scale).item()) + self.offset_x
        draw_y = int((z * self.scale).item()) + self.offset_y

        true_x = int((gt_x * self.scale).item()) + self.offset_x
        true_y = int((gt_z * self.scale).item()) + self.offset_y
        
        # Draw Visual Odometry (Green)
        cv2.circle(self.traj, (draw_x, draw_y), 1, (0, 255, 0), 1)

        # Draw Ground Truth (Red)
        cv2.circle(self.traj, (true_x, true_y), 1, (0, 0, 255), 1)

        # Draw Wheel Odometry (Blue) - if available
        if wo_xyz is not None:
            wo_x, wo_z = (
                int(wo_xyz[0].item() * self.scale) + self.offset_x,
                int(wo_xyz[1].item() * self.scale) + self.offset_y,
            )
            cv2.circle(self.traj, (wo_x, wo_z), 1, (255, 0, 0), 1)

        if ekf_xyz is not None:
            ekf_x, ekf_z = (
                int(ekf_xyz[0].item() * self.scale) + self.offset_x,
                int(ekf_xyz[2].item() * self.scale) + self.offset_y,
            )
            cv2.circle(self.traj, (ekf_x, ekf_z), 1, (255, 255, 0), 1)

        # Draw error trajectories as lines connecting consecutive error positions
        # VO Error trajectory (Green dotted)
        if len(self.vo_errors) > 1:
            prev_vo = self.vo_positions[-2]
            curr_vo = self.vo_positions[-1]
            
            # Draw line from VO to GT (error vector)
            cv2.line(self.traj, 
                    (int(prev_vo[0] * self.scale) + self.offset_x, 
                     int(prev_vo[1] * self.scale) + self.offset_y),
                    (int(curr_vo[0] * self.scale) + self.offset_x, 
                     int(curr_vo[1] * self.scale) + self.offset_y),
                    (50, 200, 50), 1)  # Darker green for error trajectory

        # WO Error trajectory (Blue dotted)
        if len(self.wo_errors) > 1 and wo_xyz is not None:
            prev_wo = self.wo_positions[-2]
            curr_wo = self.wo_positions[-1]
            cv2.line(self.traj,
                    (int(prev_wo[0] * self.scale) + self.offset_x,
                     int(prev_wo[1] * self.scale) + self.offset_y),
                    (int(curr_wo[0] * self.scale) + self.offset_x,
                     int(curr_wo[1] * self.scale) + self.offset_y),
                    (150, 100, 50), 1)  # Darker blue for error trajectory

        # EKF Error trajectory (Yellow dotted)
        if len(self.ekf_errors) > 1 and ekf_xyz is not None:
            prev_ekf = self.ekf_positions[-2]
            curr_ekf = self.ekf_positions[-1]
            cv2.line(self.traj,
                    (int(prev_ekf[0] * self.scale) + self.offset_x,
                     int(prev_ekf[1] * self.scale) + self.offset_y),
                    (int(curr_ekf[0] * self.scale) + self.offset_x,
                     int(curr_ekf[1] * self.scale) + self.offset_y),
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