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
        self.filter_errors = []
        self.ba_errors = []
        self.vo_positions = []
        self.wo_positions = []
        self.filter_positions = []
        self.ba_positions = []
        self.gt_positions = []
        self.is_robot = is_robot
        self.w, self.h = width, height
        self.offset_x = self.w // 2
        self.offset_y = self.h // 2
        self.traj = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.scale = 0.25 if not is_robot else 200  # Adjust scale for robot datasets
        
    def update(self, est_xyz, gt_xyz, wo_xyz=None, filter_xyz=None, ba_xyz=None):
        """
        Updates the trajectory plot.
        :param est_xyz: Visual Odometry position
        :param gt_xyz: Ground Truth position
        :param wo_xyz: Wheel Odometry position (Optional)
        :param filter_xyz: Filter position (Optional)
        :param ba_xyz: Bundle Adjustment position (Optional)
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

        # Calculate Filter error if available
        avg_filter_error = None
        if filter_xyz is not None:
            filter_est = np.array([filter_xyz[0], filter_xyz[2]]).reshape(2)
            filter_error = np.linalg.norm(filter_est - gt)
            self.filter_errors.append(filter_error)
            self.filter_positions.append([float(np.asarray(filter_xyz[0]).flat[0]), float(np.asarray(filter_xyz[2]).flat[0])])
            avg_filter_error = np.mean(np.array(self.filter_errors))

        avg_ba_error = None
        if ba_xyz is not None:
            ba_est = np.array([ba_xyz[0], ba_xyz[2]]).reshape(2)
            ba_error = np.linalg.norm(ba_est - gt)
            self.ba_errors.append(ba_error)
            self.ba_positions.append([float(np.asarray(ba_xyz[0]).flat[0]), float(np.asarray(ba_xyz[2]).flat[0])])
            avg_ba_error = np.mean(np.array(self.ba_errors))

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

        if filter_xyz is not None:
            filter_x, filter_z = (
                int(filter_xyz[0].item() * self.scale) + self.offset_x,
                int(filter_xyz[2].item() * self.scale) + self.offset_y,
            )
            cv2.circle(self.traj, (filter_x, filter_z), 1, (255, 255, 0), 1)

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

        # BA Error trajectory (Magenta dotted)
        if len(self.ba_errors) > 1 and ba_xyz is not None:
            prev_ba = self.ba_positions[-2]
            curr_ba = self.ba_positions[-1]
            cv2.line(self.traj,
                    (int(prev_ba[0] * self.scale) + self.offset_x,
                     int(prev_ba[1] * self.scale) + self.offset_y),
                    (int(curr_ba[0] * self.scale) + self.offset_x,
                     int(curr_ba[1] * self.scale) + self.offset_y),
                    (255, 0, 255), 1)  # Magenta for BA trajectory

        # Filter Error trajectory (Yellow dotted)
        if len(self.filter_errors) > 1 and filter_xyz is not None:
            prev_filter = self.filter_positions[-2]
            curr_filter = self.filter_positions[-1]
            cv2.line(self.traj,
                    (int(prev_filter[0] * self.scale) + self.offset_x,
                     int(prev_filter[1] * self.scale) + self.offset_y),
                    (int(curr_filter[0] * self.scale) + self.offset_x,
                     int(curr_filter[1] * self.scale) + self.offset_y),
                    (150, 150, 100), 1)  # Darker cyan for error trajectory

        # Legend and Text
        cv2.rectangle(self.traj, (10, 20), (600, 160), (0, 0, 0), -1)
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
        
        # Display Filter error if available
        if avg_filter_error is not None:
            text_filter = "Filter Error: %2.4fm" % (avg_filter_error)
            cv2.putText(
                self.traj, text_filter, (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, 8
            )
            y_offset += 15

        # Display BA error if available
        if avg_ba_error is not None:
            text_ba = "BA Error: %2.4fm" % (avg_ba_error)
            cv2.putText(
                self.traj, text_ba, (20, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1, 8
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
        if filter_xyz is not None:
            cv2.putText(
                self.traj,
                "Filter (Cyan)",
                (400, 80),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 0),
                1,
            )
        if ba_xyz is not None:
            cv2.putText(
                self.traj,
                "BA (Magenta)",
                (520, 80),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 255),
                1,
            )

        return self.traj

    def save_errors_to_csv(self, config, detector_name="", matcher_name="", output_folder="data"):
        # Create full path including sequence subdirectory
        sequence = config['dataset'].get('sequence', 'unknown')
        seq_folder = os.path.join(output_folder, sequence)
        os.makedirs(seq_folder, exist_ok=True)
        
        # Create filename with detector and matcher names (sequence already in path)
        suffix = f"{detector_name}_{matcher_name}" if detector_name and matcher_name else ""
        base_name = f"{suffix}_errors.csv" if suffix else "errors.csv"
        csv_filename = os.path.join(seq_folder, base_name)
        
        max_len = max(len(self.vo_errors), len(self.wo_errors), len(self.filter_errors), len(self.ba_errors))
        
        with open(csv_filename, mode='w', newline='') as csv_file:
            if detector_name or matcher_name:
                csv_file.write(f"# Detector: {detector_name}\n")
                csv_file.write(f"# Matcher: {matcher_name}\n")
            
            fieldnames = ['Frame', 'VO_Error', 'WO_Error', 'FILTER_Error', 'BA_Error']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for i in range(max_len):
                writer.writerow({
                    'Frame': i,
                    'VO_Error': self.vo_errors[i] if i < len(self.vo_errors) else '',
                    'WO_Error': self.wo_errors[i] if i < len(self.wo_errors) else '',
                    'FILTER_Error': self.filter_errors[i] if i < len(self.filter_errors) else '',
                    'BA_Error': self.ba_errors[i] if i < len(self.ba_errors) else '',
                })
        
        print(f"[INFO] Errors saved to {csv_filename}")