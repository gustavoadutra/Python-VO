import os
import cv2
import numpy as np


class RSTPHandler:
    """
    Handles RTSP image retrieval, calibration, and display.
    """
    
    def __init__(self, config, display_size=(640, 360)):
        """
        Initialize RTSP Handler.
        
        :param config: Configuration dictionary from the YAML file
        :param display_size: Size to display RTSP image (width, height)
        """
        self.config = config
        self.display_size = display_size
        self.available_timestamps = []
        self.calibration_maps = None
        self.rtsp_dir = None
        
        # Initialize RTSP directory and calibration
        self._setup_rtsp_dir()
        self._setup_calibration()
    
    def _setup_rtsp_dir(self):
        """Index RTSP images from the dataset directory."""
        dataroot = self.config["dataset"].get("root_path", "") + self.config["dataset"].get("sequence", "")
        self.rtsp_dir = os.path.join(dataroot, "rstp_titanium_3")
        
        if os.path.exists(self.rtsp_dir):
            print(f"[INFO] Indexing RTSP images from: {self.rtsp_dir}")
            for file in os.listdir(self.rtsp_dir):
                if file.endswith(".png"):
                    try:
                        ts = int(file.replace(".png", ""))
                        self.available_timestamps.append(ts)
                    except ValueError:
                        continue
            self.available_timestamps.sort()
            print(f"[INFO] Found {len(self.available_timestamps)} RTSP images.")
        else:
            print(f"[WARNING] RTSP directory not found at {self.rtsp_dir}")
    
    def _setup_calibration(self):
        """Setup RTSP camera calibration and undistortion maps."""
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
        
        self.calibration_maps = (map1, map2)
    
    def get_rtsp_image(self, timestamp):
        """
        Retrieve and process RTSP image for a given timestamp.
        
        :param timestamp: Timestamp from the main dataset
        :return: Processed and display-ready RTSP image, or None if not found
        """
        if not self.available_timestamps or self.rtsp_dir is None:
            return None, None, None
        
        # Parse timestamp as float
        raw_ts = float(timestamp)
        
        # Fix the scale (Seconds vs Nanoseconds)
        if raw_ts < 1e12:
            target_ts = int(raw_ts * 1e9)  # Convert seconds to nanoseconds
        else:
            target_ts = int(raw_ts)  # Already in nanoseconds
        
        # Find the closest RTSP timestamp
        closest_ts = min(self.available_timestamps, key=lambda x: abs(x - target_ts))
        rtsp_path = os.path.join(self.rtsp_dir, f"{closest_ts}.png")
        
        if os.path.exists(rtsp_path):
            rtsp_img = cv2.imread(rtsp_path)
            if rtsp_img is not None:
                # Apply calibration transformation
                rtsp_rectified = cv2.remap(rtsp_img, self.calibration_maps[0], self.calibration_maps[1], cv2.INTER_LINEAR)
                
                # Resize for display
                rtsp_display = cv2.resize(rtsp_rectified, self.display_size)
                
                # Calculate time difference in milliseconds
                diff_ms = abs(closest_ts - target_ts) / 1e6
                
                return rtsp_display, diff_ms, closest_ts
        
        return None, None, None
    
    def draw_sync_info(self, image, diff_ms, threshold_ms=50):
        """
        Draw synchronization information on the RTSP image.
        
        :param image: Image to draw on
        :param diff_ms: Time difference in milliseconds
        :param threshold_ms: Threshold for "in sync" (default 50ms)
        :return: Image with drawn info
        """
        text = f"Sync Diff: {diff_ms:.2f} ms"
        color = (0, 255, 0) if diff_ms < threshold_ms else (0, 0, 255)  # Green if in sync, Red if not
        
        # Draw with black border for better visibility
        cv2.putText(image, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(image, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return image
    
    def has_rtsp_images(self):
        """Check if RTSP images are available."""
        return len(self.available_timestamps) > 0
