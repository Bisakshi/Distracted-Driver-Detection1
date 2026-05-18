# import time
# import pprint
# import json
# import socket
# import csv
# from datetime import datetime

# import cv2
# import mediapipe as mp
# import numpy as np

# from attention_scorer import AttentionScorer as AttScorer
# from eye_detector import EyeDetector as EyeDet
# from parser import get_args
# from pose_estimation import HeadPoseEstimator as HeadPoseEst
# from utils import get_landmarks, load_camera_parameters


# class DirectionTracker:
#     """Tracks head and gaze direction over time"""
#     def __init__(self, direction_threshold_time=2.0):
#         self.direction_threshold_time = direction_threshold_time
#         self.direction_start_time = {}
#         self.current_direction = "center"
#         self.direction_duration = 0
        
#     def update_direction(self, direction, current_time):
#         if direction != self.current_direction:
#             self.current_direction = direction
#             self.direction_start_time[direction] = current_time
        
#         if direction in self.direction_start_time:
#             self.direction_duration = current_time - self.direction_start_time[direction]
#         else:
#             self.direction_duration = 0
            
#         return self.direction_duration > self.direction_threshold_time
    
#     def reset(self):
#         self.direction_start_time = {}
#         self.current_direction = "center"
#         self.direction_duration = 0


# class EnhancedHeadPoseEstimator(HeadPoseEst):
#     """Extended Head Pose Estimator with direction detection"""
#     def __init__(self, show_axis=False, camera_matrix=None, dist_coeffs=None, **kwargs):
#         super().__init__(show_axis=show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
#         self.head_direction_thresh = kwargs.get('head_direction_thresh', 20.0)
#         self.mirror_check_thresh = kwargs.get('mirror_check_thresh', 30.0)
#         self.shoulder_check_thresh = kwargs.get('shoulder_check_thresh', 60.0)
        
#     def get_head_direction(self, frame, landmarks, frame_size):
#         """Detect head direction (left/right/center) based on yaw angle"""
#         frame_det, roll, pitch, yaw = self.get_pose(frame, landmarks, frame_size)
        
#         direction = "center"
#         if yaw is not None:
#             yaw_angle = yaw[0]
            
#             # Define thresholds for direction detection
#             if yaw_angle > self.head_direction_thresh:  # Looking right
#                 direction = "right"
#             elif yaw_angle < -self.head_direction_thresh:  # Looking left
#                 direction = "left"
#             # Between thresholds is considered center
        
#         return frame_det, roll, pitch, yaw, direction
    
#     def detect_mirror_check(self, head_yaw, gaze_direction):
#         """Detect if driver is checking mirrors"""
#         if head_yaw is not None:
#             if head_yaw > self.mirror_check_thresh and gaze_direction == "right":
#                 return "checking_right_mirror"
#             elif head_yaw < -self.mirror_check_thresh and gaze_direction == "left":
#                 return "checking_left_mirror"
#         return "normal"
    
#     def detect_shoulder_check(self, head_yaw, head_roll):
#         """Detect shoulder check movements"""
#         if head_yaw is not None:
#             if head_yaw > self.shoulder_check_thresh:
#                 return "right_shoulder_check"
#             elif head_yaw < -self.shoulder_check_thresh:
#                 return "left_shoulder_check"
#         return "normal"


# class EnhancedEyeDetector(EyeDet):
#     """Extended Eye Detector with gaze direction detection"""
#     def __init__(self, show_processing=False, gaze_direction_thresh=0.2):
#         super().__init__(show_processing=show_processing)
#         self.gaze_direction_thresh = gaze_direction_thresh
        
#     def get_gaze_direction(self, frame, landmarks, frame_size):
#         """Detect gaze direction based on eye landmarks"""
#         if landmarks is None or len(landmarks) < 474:  # Need iris landmarks
#             return "center"
        
#         # MediaPipe landmark indices (x, y, z format)
#         # Landmarks are numpy arrays with [x, y, z] coordinates
#         left_eye_landmarks_indices = [33, 133, 157, 158, 159, 160, 161, 246]
#         right_eye_landmarks_indices = [362, 263, 386, 387, 388, 389, 390, 467]
        
#         # Get left eye horizontal direction
#         left_eye_landmarks = [landmarks[i] for i in left_eye_landmarks_indices if i < len(landmarks)]
#         if not left_eye_landmarks:
#             return "center"
        
#         # Landmarks are numpy arrays, not objects with .x attribute
#         left_eye_center_x = np.mean([lm[0] * frame_size[0] for lm in left_eye_landmarks])
        
#         # Get right eye horizontal direction
#         right_eye_landmarks = [landmarks[i] for i in right_eye_landmarks_indices if i < len(landmarks)]
#         if not right_eye_landmarks:
#             return "center"
#         right_eye_center_x = np.mean([lm[0] * frame_size[0] for lm in right_eye_landmarks])
        
#         # Get iris landmarks (if available)
#         left_iris_index = 468 if len(landmarks) > 468 else None
#         right_iris_index = 473 if len(landmarks) > 473 else None
        
#         gaze_direction = "center"
        
#         if left_iris_index is not None and right_iris_index is not None:
#             left_iris = landmarks[left_iris_index]
#             right_iris = landmarks[right_iris_index]
            
#             # Calculate gaze ratios
#             # Landmarks are normalized [0, 1], multiply by frame size to get pixel coordinates
#             left_gaze_ratio = (left_iris[0] * frame_size[0] - left_eye_center_x) / (frame_size[0] * 0.1)
#             right_gaze_ratio = (right_iris[0] * frame_size[0] - right_eye_center_x) / (frame_size[0] * 0.1)
            
#             avg_gaze_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
            
#             # Thresholds for direction detection
#             if avg_gaze_ratio > self.gaze_direction_thresh:
#                 gaze_direction = "right"
#             elif avg_gaze_ratio < -self.gaze_direction_thresh:
#                 gaze_direction = "left"
        
#         return gaze_direction

# class EnhancedAttentionScorer(AttScorer):
#     """Extended Attention Scorer with direction detection"""
#     def __init__(self, **kwargs):
#         # Extract direction-specific parameters first
#         direction_time_thresh = kwargs.pop('direction_time_thresh', 2.0)
        
#         # Call parent constructor with remaining kwargs
#         super().__init__(**kwargs)
        
#         # Add direction tracking parameters
#         self.looking_left_time = 0
#         self.looking_right_time = 0
#         self.looking_left_start = None
#         self.looking_right_start = None
#         self.direction_time_thresh = direction_time_thresh
        
#         # State tracking
#         self.extended_left_look = False
#         self.extended_right_look = False
        
#     def eval_scores_with_direction(self, t_now, ear_score, gaze_score, head_roll, 
#                                    head_pitch, head_yaw, head_direction="center", 
#                                    gaze_direction="center"):
#         """Evaluate scores including direction detection"""
        
#         # Call parent method for basic attention states
#         asleep, looking_away, distracted = self.eval_scores(
#             t_now=t_now,
#             ear_score=ear_score,
#             gaze_score=gaze_score,
#             head_roll=head_roll,
#             head_pitch=head_pitch,
#             head_yaw=head_yaw
#         )
        
#         # Reset extended look flags
#         self.extended_left_look = False
#         self.extended_right_look = False
        
#         # Check head direction for extended looks
#         if head_direction == "left":
#             if self.looking_left_start is None:
#                 self.looking_left_start = t_now
#             self.looking_left_time = t_now - self.looking_left_start
            
#             if self.looking_left_time > self.direction_time_thresh:
#                 self.extended_left_look = True
#         else:
#             self.looking_left_start = None
#             self.looking_left_time = 0
        
#         if head_direction == "right":
#             if self.looking_right_start is None:
#                 self.looking_right_start = t_now
#             self.looking_right_time = t_now - self.looking_right_start
            
#             if self.looking_right_time > self.direction_time_thresh:
#                 self.extended_right_look = True
#         else:
#             self.looking_right_start = None
#             self.looking_right_time = 0
        
#         # Additional check: If gaze confirms head direction
#         if gaze_direction == head_direction and head_direction != "center":
#             # Reduce threshold if both signals agree
#             effective_threshold = self.direction_time_thresh * 0.7
#             if head_direction == "left" and self.looking_left_time > effective_threshold:
#                 self.extended_left_look = True
#             elif head_direction == "right" and self.looking_right_time > effective_threshold:
#                 self.extended_right_look = True
        
#         return asleep, looking_away, distracted, self.extended_left_look, self.extended_right_look
    
#     def get_direction_durations(self):
#         """Get current direction durations"""
#         return {
#             "left_duration": self.looking_left_time,
#             "right_duration": self.looking_right_time
#         }


# class DataLogger:
#     """Logs detection data to CSV file"""
#     def __init__(self, log_file="driver_detection_log.csv"):
#         self.log_file = log_file
#         self.headers_written = False
        
#     def log_data(self, data):
#         """Log detection data to CSV file"""
#         try:
#             with open(self.log_file, 'a', newline='') as f:
#                 writer = csv.DictWriter(f, fieldnames=data.keys())
#                 if not self.headers_written:
#                     writer.writeheader()
#                     self.headers_written = True
#                 writer.writerow(data)
#         except Exception as e:
#             print(f"Error writing to log file: {e}")


# def send_to_frontend(data, host='localhost', port=5000):
#     """Send detection data to frontend via socket"""
#     try:
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((host, port))
#             s.sendall(json.dumps(data).encode())
#     except Exception as e:
#         print(f"Frontend connection error: {e}")


# def main():
#     args = get_args()

#     if not cv2.useOptimized():
#         try:
#             cv2.setUseOptimized(True)
#         except Exception as e:
#             print(f"OpenCV optimization error: {e}")

#     if args.camera_params:
#         camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
#     else:
#         camera_matrix, dist_coeffs = None, None

#     if args.verbose:
#         print("Arguments and Parameters:\n")
#         pprint.pp(vars(args), indent=4)
#         print("\nCamera Matrix:")
#         pprint.pp(camera_matrix, indent=4)
#         print("\nDistortion Coefficients:")
#         pprint.pp(dist_coeffs, indent=4)
#         print("\n")

#     # Initialize MediaPipe Face Mesh
#     Detector = mp.solutions.face_mesh.FaceMesh(
#         static_image_mode=False,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5,
#         refine_landmarks=True,
#     )

#     # Initialize enhanced detectors with parameters
#     Eye_det = EnhancedEyeDetector(
#         show_processing=args.show_eye_proc,
#         gaze_direction_thresh=args.gaze_direction_thresh
#     )
    
#     # Prepare kwargs for HeadPoseEstimator
#     head_pose_kwargs = {
#         'head_direction_thresh': args.head_direction_thresh,
#         'mirror_check_thresh': args.mirror_check_thresh,
#         'shoulder_check_thresh': args.shoulder_check_thresh
#     }
    
#     Head_pose = EnhancedHeadPoseEstimator(
#         show_axis=args.show_axis,
#         camera_matrix=camera_matrix,
#         dist_coeffs=dist_coeffs,
#         **head_pose_kwargs
#     )

#     # Initialize direction tracker
#     direction_tracker = DirectionTracker(
#         direction_threshold_time=args.direction_time_thresh
#     )

#     # Initialize data logger if enabled
#     data_logger = None
#     if args.save_logs:
#         data_logger = DataLogger(args.log_file)

#     # Timing variables
#     prev_time = time.perf_counter()
#     fps = 0.0
#     t_now = time.perf_counter()

#     # Prepare kwargs for EnhancedAttentionScorer
#     scorer_kwargs = {
#         't_now': t_now,
#         'ear_thresh': args.ear_thresh,
#         'gaze_time_thresh': args.gaze_time_thresh,
#         'roll_thresh': args.roll_thresh,
#         'pitch_thresh': args.pitch_thresh,
#         'yaw_thresh': args.yaw_thresh,
#         'ear_time_thresh': args.ear_time_thresh,
#         'gaze_thresh': args.gaze_thresh,
#         'pose_time_thresh': args.pose_time_thresh,
#         'verbose': args.verbose,
#         'direction_time_thresh': args.direction_time_thresh
#     }

#     # Initialize enhanced attention scorer
#     Scorer = EnhancedAttentionScorer(**scorer_kwargs)

#     # Open camera
#     cap = cv2.VideoCapture(args.camera)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         exit()

#     # Main loop variables
#     last_frontend_update = 0
#     log_data_buffer = []

#     while True:
#         # Get current time
#         t_now = time.perf_counter()
#         elapsed_time = t_now - prev_time
#         prev_time = t_now

#         # Calculate FPS
#         if elapsed_time > 0:
#             fps = np.round(1 / elapsed_time, 3)

#         # Read frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Can't receive frame")
#             break

#         # Flip if using webcam
#         if args.camera == 0:
#             frame = cv2.flip(frame, 2)

#         # Start processing timer
#         e1 = cv2.getTickCount()

#         # Convert to grayscale for processing
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame_size = frame.shape[1], frame.shape[0]
        
#         # Prepare image for MediaPipe
#         gray_rgb = np.expand_dims(gray, axis=2)
#         gray_rgb = np.concatenate([gray_rgb, gray_rgb, gray_rgb], axis=2)

#         # Detect faces
#         lms = Detector.process(gray_rgb).multi_face_landmarks

#         # Initialize detection variables
#         ear = None
#         gaze = None
#         roll = pitch = yaw = None
#         head_direction = "center"
#         gaze_direction = "center"
#         mirror_check = "normal"
#         shoulder_check = "normal"
#         perclos_score = 0
#         tired = False
#         asleep = looking_away = distracted = False
#         extended_left = extended_right = False
#         direction_duration = 0

#         if lms:
#             # Get landmarks
#             landmarks = get_landmarks(lms)

#             # Show eye keypoints if enabled
#             if args.show_eye_proc:
#                 Eye_det.show_eye_keypoints(
#                     color_frame=frame,
#                     landmarks=landmarks,
#                     frame_size=frame_size
#                 )

#             # Compute EAR
#             ear = Eye_det.get_EAR(landmarks=landmarks)

#             # Compute PERCLOS
#             tired, perclos_score = Scorer.get_rolling_PERCLOS(t_now, ear)

#             # Compute Gaze Score
#             gaze = Eye_det.get_Gaze_Score(
#                 frame=gray_rgb,
#                 landmarks=landmarks,
#                 frame_size=frame_size
#             )

#             # Get gaze direction
#             gaze_direction = Eye_det.get_gaze_direction(
#                 frame=gray_rgb,
#                 landmarks=landmarks,
#                 frame_size=frame_size
#             )

#             # Get head pose and direction
#             frame_det, roll, pitch, yaw, head_direction = Head_pose.get_head_direction(
#                 frame=frame,
#                 landmarks=landmarks,
#                 frame_size=frame_size
#             )

#             # Update frame if head pose detected
#             if frame_det is not None:
#                 frame = frame_det

#             # Detect special movements if enabled
#             if args.enable_mirror_check and yaw is not None:
#                 mirror_check = Head_pose.detect_mirror_check(yaw[0], gaze_direction)
            
#             if args.enable_shoulder_check and yaw is not None:
#                 shoulder_check = Head_pose.detect_shoulder_check(yaw[0], roll[0] if roll is not None else None)

#             # Update direction tracker
#             if head_direction == gaze_direction:
#                 final_direction = head_direction
#             else:
#                 final_direction = head_direction  # Prioritize head direction
            
#             extended_look_detected = direction_tracker.update_direction(final_direction, t_now)
#             direction_duration = direction_tracker.direction_duration

#             # Evaluate all scores
#             asleep, looking_away, distracted, extended_left, extended_right = Scorer.eval_scores_with_direction(
#                 t_now=t_now,
#                 ear_score=ear,
#                 gaze_score=gaze,
#                 head_roll=roll,
#                 head_pitch=pitch,
#                 head_yaw=yaw,
#                 head_direction=head_direction,
#                 gaze_direction=gaze_direction
#             )

#         # Display information on frame
#         y_offset = 50
#         line_height = 30

#         # Basic metrics
#         if ear is not None:
#             cv2.putText(
#                 frame,
#                 f"EAR: {ear:.3f}",
#                 (10, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2,
#             )
#             y_offset += line_height

#         if gaze is not None:
#             cv2.putText(
#                 frame,
#                 f"Gaze: {gaze:.3f}",
#                 (10, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 255, 255),
#                 2,
#             )
#             y_offset += line_height

#         cv2.putText(
#             frame,
#             f"PERCLOS: {perclos_score:.3f}",
#             (10, y_offset),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (255, 255, 255),
#             2,
#         )
#         y_offset += line_height

#         # Direction information
#         cv2.putText(
#             frame,
#             f"Head: {head_direction.upper()}",
#             (10, y_offset),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 255) if head_direction != "center" else (255, 255, 255),
#             2,
#         )
#         y_offset += line_height

#         cv2.putText(
#             frame,
#             f"Gaze: {gaze_direction.upper()}",
#             (10, y_offset),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.7,
#             (0, 255, 255) if gaze_direction != "center" else (255, 255, 255),
#             2,
#         )
#         y_offset += line_height

#         # Direction duration
#         if direction_duration > 0:
#             cv2.putText(
#                 frame,
#                 f"Looking {direction_tracker.current_direction}: {direction_duration:.1f}s",
#                 (10, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 200, 0),
#                 2,
#             )
#             y_offset += line_height

#         # Head pose angles
#         if roll is not None and pitch is not None and yaw is not None:
#             cv2.putText(
#                 frame,
#                 f"Roll: {roll[0]:.1f}°, Pitch: {pitch[0]:.1f}°, Yaw: {yaw[0]:.1f}°",
#                 (10, y_offset),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 0, 255),
#                 2,
#             )
#             y_offset += line_height

#         # Special movements
#         if args.show_special_movements:
#             if mirror_check != "normal":
#                 cv2.putText(
#                     frame,
#                     f"{mirror_check.replace('_', ' ').upper()}",
#                     (10, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 255, 0),
#                     2,
#                 )
#                 y_offset += line_height

#             if shoulder_check != "normal":
#                 cv2.putText(
#                     frame,
#                     f"{shoulder_check.replace('_', ' ').upper()}",
#                     (10, y_offset),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.7,
#                     (0, 200, 255),
#                     2,
#                 )
#                 y_offset += line_height

#         # Alert boxes at the bottom
#         alert_y = frame.shape[0] - 150
#         alert_height = 30

#         if tired:
#             cv2.putText(
#                 frame,
#                 "TIRED!",
#                 (10, alert_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 2,
#             )
#             alert_y -= alert_height

#         if asleep:
#             cv2.putText(
#                 frame,
#                 "ASLEEP!",
#                 (10, alert_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 2,
#             )
#             alert_y -= alert_height

#         if looking_away:
#             cv2.putText(
#                 frame,
#                 "LOOKING AWAY!",
#                 (10, alert_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 2,
#             )
#             alert_y -= alert_height

#         if distracted:
#             cv2.putText(
#                 frame,
#                 "DISTRACTED!",
#                 (10, alert_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 0, 255),
#                 2,
#             )
#             alert_y -= alert_height

#         if extended_left:
#             cv2.putText(
#                 frame,
#                 "EXTENDED LEFT LOOK!",
#                 (10, alert_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 100, 255),
#                 2,
#             )
#             alert_y -= alert_height

#         if extended_right:
#             cv2.putText(
#                 frame,
#                 "EXTENDED RIGHT LOOK!",
#                 (10, alert_y),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 100, 255),
#                 2,
#             )
#             alert_y -= alert_height

#         # Draw visual direction arrows if enabled
#         if args.show_direction_arrows and head_direction != "center":
#             if head_direction == "left":
#                 cv2.arrowedLine(frame, (100, 300), (50, 300), (0, 255, 255), 3, tipLength=0.3)
#             elif head_direction == "right":
#                 cv2.arrowedLine(frame, (frame.shape[1] - 100, 300), (frame.shape[1] - 50, 300), 
#                               (0, 255, 255), 3, tipLength=0.3)

#         # Send data to frontend periodically
#         if args.send_to_frontend and (t_now - last_frontend_update > args.frontend_update_interval):
#             detection_data = {
#                 "timestamp": datetime.now().isoformat(),
#                 "ear_score": float(ear) if ear is not None else 0.0,
#                 "gaze_score": float(gaze) if gaze is not None else 0.0,
#                 "perclos": float(perclos_score),
#                 "head_direction": head_direction,
#                 "gaze_direction": gaze_direction,
#                 "head_pose": {
#                     "roll": float(roll[0]) if roll is not None else 0.0,
#                     "pitch": float(pitch[0]) if pitch is not None else 0.0,
#                     "yaw": float(yaw[0]) if yaw is not None else 0.0
#                 },
#                 "alerts": {
#                     "tired": bool(tired),
#                     "asleep": bool(asleep),
#                     "looking_away": bool(looking_away),
#                     "distracted": bool(distracted),
#                     "extended_left_look": bool(extended_left),
#                     "extended_right_look": bool(extended_right)
#                 },
#                 "special_movements": {
#                     "mirror_check": mirror_check,
#                     "shoulder_check": shoulder_check
#                 },
#                 "direction_duration": float(direction_duration),
#                 "fps": float(fps)
#             }
            
#             send_to_frontend(detection_data, args.frontend_host, args.frontend_port)
            
#             # Add to log buffer
#             log_data_buffer.append(detection_data)
            
#             last_frontend_update = t_now

#         # Save logs if enabled
#         if args.save_logs and data_logger and len(log_data_buffer) > 0:
#             for data in log_data_buffer:
#                 data_logger.log_data(data)
#             log_data_buffer.clear()

#         # Display FPS and processing time
#         if args.show_fps:
#             cv2.putText(
#                 frame,
#                 f"FPS: {int(fps)}",
#                 (frame.shape[1] - 150, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (255, 0, 255),
#                 2,
#             )

#         if args.show_proc_time:
#             e2 = cv2.getTickCount()
#             proc_time_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
#             cv2.putText(
#                 frame,
#                 f"Proc: {proc_time_ms:.1f}ms",
#                 (frame.shape[1] - 150, 80),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.7,
#                 (255, 0, 255),
#                 2,
#             )

#         # Show frame
#         cv2.imshow("Driver Attention & Direction Detection - Press 'q' to exit", frame)

#         # Exit on 'q' press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()
    
#     # Save any remaining log data
#     if args.save_logs and data_logger and len(log_data_buffer) > 0:
#         for data in log_data_buffer:
#             data_logger.log_data(data)


# if __name__ == "__main__":
#     main()


import time
import pprint
import json
import socket
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import cv2
import mediapipe as mp
import numpy as np

from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from parser import get_args
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters


class DirectionTracker:
    """Tracks head and gaze direction over time"""
    def __init__(self, direction_threshold_time=2.0):
        self.direction_threshold_time = direction_threshold_time
        self.direction_start_time = {}
        self.current_direction = "center"
        self.current_vertical_direction = "center"
        self.direction_duration = 0
        self.vertical_direction_duration = 0
        
    def update_direction(self, horizontal_direction, vertical_direction, current_time):
        # Update horizontal direction
        if horizontal_direction != self.current_direction:
            self.current_direction = horizontal_direction
            self.direction_start_time[horizontal_direction] = current_time
        
        if horizontal_direction in self.direction_start_time:
            self.direction_duration = current_time - self.direction_start_time[horizontal_direction]
        else:
            self.direction_duration = 0
        
        # Update vertical direction
        if vertical_direction != self.current_vertical_direction:
            self.current_vertical_direction = vertical_direction
            self.direction_start_time[f"vertical_{vertical_direction}"] = current_time
        
        vertical_key = f"vertical_{vertical_direction}"
        if vertical_key in self.direction_start_time:
            self.vertical_direction_duration = current_time - self.direction_start_time[vertical_key]
        else:
            self.vertical_direction_duration = 0
            
        return (
            self.direction_duration > self.direction_threshold_time,
            self.vertical_direction_duration > self.direction_threshold_time
        )
    
    def reset(self):
        self.direction_start_time = {}
        self.current_direction = "center"
        self.current_vertical_direction = "center"
        self.direction_duration = 0
        self.vertical_direction_duration = 0


class EnhancedHeadPoseEstimator(HeadPoseEst):
    """Extended Head Pose Estimator with direction detection"""
    def __init__(self, show_axis=False, camera_matrix=None, dist_coeffs=None, **kwargs):
        super().__init__(show_axis=show_axis, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        self.head_direction_thresh = kwargs.get('head_direction_thresh', 20.0)
        self.head_vertical_thresh = kwargs.get('head_vertical_thresh', 15.0)
        self.mirror_check_thresh = kwargs.get('mirror_check_thresh', 30.0)
        self.shoulder_check_thresh = kwargs.get('shoulder_check_thresh', 60.0)
        
    def get_head_direction(self, frame, landmarks, frame_size):
        """Detect head direction (left/right/center and up/down/center)"""
        frame_det, roll, pitch, yaw = self.get_pose(frame, landmarks, frame_size)
        
        horizontal_direction = "center"
        vertical_direction = "center"
        
        if yaw is not None:
            yaw_angle = yaw[0]
            
            # Define thresholds for horizontal direction detection
            if yaw_angle > self.head_direction_thresh:  # Looking right
                horizontal_direction = "right"
            elif yaw_angle < -self.head_direction_thresh:  # Looking left
                horizontal_direction = "left"
        
        if pitch is not None:
            pitch_angle = pitch[0]
            
            # Define thresholds for vertical direction detection
            if pitch_angle > self.head_vertical_thresh:  # Looking down
                vertical_direction = "down"
            elif pitch_angle < -self.head_vertical_thresh:  # Looking up
                vertical_direction = "up"
        
        return frame_det, roll, pitch, yaw, horizontal_direction, vertical_direction
    
    def detect_mirror_check(self, head_yaw, gaze_direction):
        """Detect if driver is checking mirrors"""
        if head_yaw is not None:
            if head_yaw > self.mirror_check_thresh and gaze_direction == "right":
                return "checking_right_mirror"
            elif head_yaw < -self.mirror_check_thresh and gaze_direction == "left":
                return "checking_left_mirror"
        return "normal"
    
    def detect_shoulder_check(self, head_yaw, head_roll):
        """Detect shoulder check movements"""
        if head_yaw is not None:
            if head_yaw > self.shoulder_check_thresh:
                return "right_shoulder_check"
            elif head_yaw < -self.shoulder_check_thresh:
                return "left_shoulder_check"
        return "normal"


class EnhancedEyeDetector(EyeDet):
    """Extended Eye Detector with gaze direction detection"""
    def __init__(self, show_processing=False, gaze_direction_thresh=0.2, gaze_vertical_thresh=0.15):
        super().__init__(show_processing=show_processing)
        self.gaze_direction_thresh = gaze_direction_thresh
        self.gaze_vertical_thresh = gaze_vertical_thresh
        
    def get_gaze_direction(self, frame, landmarks, frame_size):
        """Detect gaze direction (horizontal and vertical) based on eye landmarks"""
        if landmarks is None or len(landmarks) < 474:  # Need iris landmarks
            return "center", "center"
        
        try:
            # MediaPipe landmark indices
            left_eye_landmarks_indices = [33, 133, 157, 158, 159, 160, 161, 246]
            right_eye_landmarks_indices = [362, 263, 386, 387, 388, 389, 390, 467]
            
            # Get left eye center (horizontal and vertical)
            left_eye_landmarks = [landmarks[i] for i in left_eye_landmarks_indices if i < len(landmarks)]
            if not left_eye_landmarks:
                return "center", "center"
            
            left_eye_center_x = np.mean([lm[0] * frame_size[0] for lm in left_eye_landmarks])
            left_eye_center_y = np.mean([lm[1] * frame_size[1] for lm in left_eye_landmarks])
            
            # Get right eye center (horizontal and vertical)
            right_eye_landmarks = [landmarks[i] for i in right_eye_landmarks_indices if i < len(landmarks)]
            if not right_eye_landmarks:
                return "center", "center"
            
            right_eye_center_x = np.mean([lm[0] * frame_size[0] for lm in right_eye_landmarks])
            right_eye_center_y = np.mean([lm[1] * frame_size[1] for lm in right_eye_landmarks])
            
            # Get iris landmarks
            left_iris_index = 468 if len(landmarks) > 468 else None
            right_iris_index = 473 if len(landmarks) > 473 else None
            
            horizontal_direction = "center"
            vertical_direction = "center"
            
            if left_iris_index is not None and right_iris_index is not None:
                left_iris = landmarks[left_iris_index]
                right_iris = landmarks[right_iris_index]
                
                # Calculate horizontal gaze ratios
                left_gaze_horizontal = (left_iris[0] * frame_size[0] - left_eye_center_x) / (frame_size[0] * 0.1)
                right_gaze_horizontal = (right_iris[0] * frame_size[0] - right_eye_center_x) / (frame_size[0] * 0.1)
                avg_gaze_horizontal = (left_gaze_horizontal + right_gaze_horizontal) / 2
                
                # Calculate vertical gaze ratios
                left_gaze_vertical = (left_iris[1] * frame_size[1] - left_eye_center_y) / (frame_size[1] * 0.1)
                right_gaze_vertical = (right_iris[1] * frame_size[1] - right_eye_center_y) / (frame_size[1] * 0.1)
                avg_gaze_vertical = (left_gaze_vertical + right_gaze_vertical) / 2
                
                # Determine horizontal direction
                if avg_gaze_horizontal > self.gaze_direction_thresh:
                    horizontal_direction = "right"
                elif avg_gaze_horizontal < -self.gaze_direction_thresh:
                    horizontal_direction = "left"
                
                # Determine vertical direction
                if avg_gaze_vertical > self.gaze_vertical_thresh:
                    vertical_direction = "down"
                elif avg_gaze_vertical < -self.gaze_vertical_thresh:
                    vertical_direction = "up"
            
            return horizontal_direction, vertical_direction
            
        except Exception as e:
            print(f"Error in get_gaze_direction: {e}")
            return "center", "center"


class EnhancedAttentionScorer(AttScorer):
    """Extended Attention Scorer with direction detection"""
    def __init__(self, **kwargs):
        # Extract direction-specific parameters first
        direction_time_thresh = kwargs.pop('direction_time_thresh', 2.0)
        vertical_time_thresh = kwargs.pop('vertical_time_thresh', 1.5)
        
        # Call parent constructor with remaining kwargs
        super().__init__(**kwargs)
        
        # Add direction tracking parameters
        self.looking_left_time = 0
        self.looking_right_time = 0
        self.looking_up_time = 0
        self.looking_down_time = 0
        
        self.looking_left_start = None
        self.looking_right_start = None
        self.looking_up_start = None
        self.looking_down_start = None
        
        self.direction_time_thresh = direction_time_thresh
        self.vertical_time_thresh = vertical_time_thresh
        
        # State tracking
        self.extended_left_look = False
        self.extended_right_look = False
        self.extended_up_look = False
        self.extended_down_look = False
        
    def eval_scores_with_direction(self, t_now, ear_score, gaze_score, head_roll, 
                                   head_pitch, head_yaw, head_direction="center", 
                                   head_vertical_direction="center", gaze_direction="center",
                                   gaze_vertical_direction="center"):
        """Evaluate scores including direction detection"""
        
        # Call parent method for basic attention states
        asleep, looking_away, distracted = self.eval_scores(
            t_now=t_now,
            ear_score=ear_score,
            gaze_score=gaze_score,
            head_roll=head_roll,
            head_pitch=head_pitch,
            head_yaw=head_yaw
        )
        
        # Reset extended look flags
        self.extended_left_look = False
        self.extended_right_look = False
        self.extended_up_look = False
        self.extended_down_look = False
        
        # Check horizontal head direction for extended looks
        if head_direction == "left":
            if self.looking_left_start is None:
                self.looking_left_start = t_now
            self.looking_left_time = t_now - self.looking_left_start
            
            if self.looking_left_time > self.direction_time_thresh:
                self.extended_left_look = True
        else:
            self.looking_left_start = None
            self.looking_left_time = 0
        
        if head_direction == "right":
            if self.looking_right_start is None:
                self.looking_right_start = t_now
            self.looking_right_time = t_now - self.looking_right_start
            
            if self.looking_right_time > self.direction_time_thresh:
                self.extended_right_look = True
        else:
            self.looking_right_start = None
            self.looking_right_time = 0
        
        # Check vertical head direction for extended looks
        if head_vertical_direction == "up":
            if self.looking_up_start is None:
                self.looking_up_start = t_now
            self.looking_up_time = t_now - self.looking_up_start
            
            if self.looking_up_time > self.vertical_time_thresh:
                self.extended_up_look = True
        else:
            self.looking_up_start = None
            self.looking_up_time = 0
        
        if head_vertical_direction == "down":
            if self.looking_down_start is None:
                self.looking_down_start = t_now
            self.looking_down_time = t_now - self.looking_down_start
            
            if self.looking_down_time > self.vertical_time_thresh:
                self.extended_down_look = True
        else:
            self.looking_down_start = None
            self.looking_down_time = 0
        
        # Additional check: If gaze confirms head direction
        if gaze_direction == head_direction and head_direction != "center":
            # Reduce threshold if both signals agree
            effective_threshold = self.direction_time_thresh * 0.7
            if head_direction == "left" and self.looking_left_time > effective_threshold:
                self.extended_left_look = True
            elif head_direction == "right" and self.looking_right_time > effective_threshold:
                self.extended_right_look = True
        
        if gaze_vertical_direction == head_vertical_direction and head_vertical_direction != "center":
            effective_threshold = self.vertical_time_thresh * 0.7
            if head_vertical_direction == "up" and self.looking_up_time > effective_threshold:
                self.extended_up_look = True
            elif head_vertical_direction == "down" and self.looking_down_time > effective_threshold:
                self.extended_down_look = True
        
        return asleep, looking_away, distracted, self.extended_left_look, self.extended_right_look, self.extended_up_look, self.extended_down_look
    
    def get_direction_durations(self):
        """Get current direction durations"""
        return {
            "left_duration": self.looking_left_time,
            "right_duration": self.looking_right_time,
            "up_duration": self.looking_up_time,
            "down_duration": self.looking_down_time
        }


class DataLogger:
    """Logs detection data to CSV file"""
    def __init__(self, log_file="driver_detection_log.csv"):
        self.log_file = log_file
        self.headers_written = False
        
    def log_data(self, data):
        """Log detection data to CSV file"""
        try:
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not self.headers_written:
                    writer.writeheader()
                    self.headers_written = True
                writer.writerow(data)
        except Exception as e:
            print(f"Error writing to log file: {e}")


def send_to_frontend(data, host='localhost', port=5000):
    """Send detection data to frontend via socket"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(json.dumps(data).encode())
    except Exception as e:
        print(f"Frontend connection error: {e}")


def main():
    args = get_args()

    if not cv2.useOptimized():
        try:
            cv2.setUseOptimized(True)
        except Exception as e:
            print(f"OpenCV optimization error: {e}")

    if args.camera_params:
        camera_matrix, dist_coeffs = load_camera_parameters(args.camera_params)
    else:
        camera_matrix, dist_coeffs = None, None

    if args.verbose:
        print("Arguments and Parameters:\n")
        pprint.pp(vars(args), indent=4)
        print("\nCamera Matrix:")
        pprint.pp(camera_matrix, indent=4)
        print("\nDistortion Coefficients:")
        pprint.pp(dist_coeffs, indent=4)
        print("\n")

    # Initialize MediaPipe Face Mesh
    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    # Initialize enhanced detectors with parameters
    Eye_det = EnhancedEyeDetector(
        show_processing=args.show_eye_proc,
        gaze_direction_thresh=args.gaze_direction_thresh,
        gaze_vertical_thresh=getattr(args, 'gaze_vertical_thresh', 0.15)
    )
    
    # Prepare kwargs for HeadPoseEstimator
    head_pose_kwargs = {
        'head_direction_thresh': args.head_direction_thresh,
        'head_vertical_thresh': getattr(args, 'head_vertical_thresh', 15.0),
        'mirror_check_thresh': args.mirror_check_thresh,
        'shoulder_check_thresh': args.shoulder_check_thresh
    }
    
    Head_pose = EnhancedHeadPoseEstimator(
        show_axis=args.show_axis,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        **head_pose_kwargs
    )

    # Initialize direction tracker
    direction_tracker = DirectionTracker(
        direction_threshold_time=args.direction_time_thresh
    )

    # Initialize data logger if enabled
    data_logger = None
    if args.save_logs:
        data_logger = DataLogger(args.log_file)

    # Timing variables
    prev_time = time.perf_counter()
    fps = 0.0
    t_now = time.perf_counter()

    # Prepare kwargs for EnhancedAttentionScorer
    scorer_kwargs = {
        't_now': t_now,
        'ear_thresh': args.ear_thresh,
        'gaze_time_thresh': args.gaze_time_thresh,
        'roll_thresh': args.roll_thresh,
        'pitch_thresh': args.pitch_thresh,
        'yaw_thresh': args.yaw_thresh,
        'ear_time_thresh': args.ear_time_thresh,
        'gaze_thresh': args.gaze_thresh,
        'pose_time_thresh': args.pose_time_thresh,
        'verbose': args.verbose,
        'direction_time_thresh': args.direction_time_thresh,
        'vertical_time_thresh': getattr(args, 'vertical_time_thresh', 1.5)
    }

    # Initialize enhanced attention scorer
    Scorer = EnhancedAttentionScorer(**scorer_kwargs)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Main loop variables
    last_frontend_update = 0
    log_data_buffer = []

    while True:
        # Get current time
        t_now = time.perf_counter()
        elapsed_time = t_now - prev_time
        prev_time = t_now

        # Calculate FPS
        if elapsed_time > 0:
            fps = np.round(1 / elapsed_time, 3)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame")
            break

        # Flip if using webcam
        if args.camera == 0:
            frame = cv2.flip(frame, 2)

        # Start processing timer
        e1 = cv2.getTickCount()

        # Convert to grayscale for processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_size = frame.shape[1], frame.shape[0]
        
        # Prepare image for MediaPipe
        gray_rgb = np.expand_dims(gray, axis=2)
        gray_rgb = np.concatenate([gray_rgb, gray_rgb, gray_rgb], axis=2)

        # Detect faces
        lms = Detector.process(gray_rgb).multi_face_landmarks

        # Initialize detection variables
        ear = None
        gaze = None
        roll = pitch = yaw = None
        head_direction = "center"
        head_vertical_direction = "center"
        gaze_direction = "center"
        gaze_vertical_direction = "center"
        mirror_check = "normal"
        shoulder_check = "normal"
        perclos_score = 0
        tired = False
        asleep = looking_away = distracted = False
        extended_left = extended_right = False
        extended_up = extended_down = False
        direction_duration = 0
        vertical_direction_duration = 0

        if lms:
            # Get landmarks
            landmarks = get_landmarks(lms)

            # Show eye keypoints if enabled
            if args.show_eye_proc:
                Eye_det.show_eye_keypoints(
                    color_frame=frame,
                    landmarks=landmarks,
                    frame_size=frame_size
                )

            # Compute EAR
            ear = Eye_det.get_EAR(landmarks=landmarks)

            # Compute PERCLOS
            tired, perclos_score = Scorer.get_rolling_PERCLOS(t_now, ear)

            # Compute Gaze Score
            gaze = Eye_det.get_Gaze_Score(
                frame=gray_rgb,
                landmarks=landmarks,
                frame_size=frame_size
            )

            # Get gaze direction (horizontal and vertical)
            gaze_direction, gaze_vertical_direction = Eye_det.get_gaze_direction(
                frame=gray_rgb,
                landmarks=landmarks,
                frame_size=frame_size
            )

            # Get head pose and direction (horizontal and vertical)
            frame_det, roll, pitch, yaw, head_direction, head_vertical_direction = Head_pose.get_head_direction(
                frame=frame,
                landmarks=landmarks,
                frame_size=frame_size
            )

            # Update frame if head pose detected
            if frame_det is not None:
                frame = frame_det

            # Detect special movements if enabled
            if args.enable_mirror_check and yaw is not None:
                mirror_check = Head_pose.detect_mirror_check(yaw[0], gaze_direction)
            
            if args.enable_shoulder_check and yaw is not None:
                shoulder_check = Head_pose.detect_shoulder_check(yaw[0], roll[0] if roll is not None else None)

            # Update direction tracker
            extended_horizontal, extended_vertical = direction_tracker.update_direction(
                head_direction, head_vertical_direction, t_now
            )
            direction_duration = direction_tracker.direction_duration
            vertical_direction_duration = direction_tracker.vertical_direction_duration

            # Evaluate all scores
            asleep, looking_away, distracted, extended_left, extended_right, extended_up, extended_down = Scorer.eval_scores_with_direction(
                t_now=t_now,
                ear_score=ear,
                gaze_score=gaze,
                head_roll=roll,
                head_pitch=pitch,
                head_yaw=yaw,
                head_direction=head_direction,
                head_vertical_direction=head_vertical_direction,
                gaze_direction=gaze_direction,
                gaze_vertical_direction=gaze_vertical_direction
            )

        # Display information on frame
        y_offset = 50
        line_height = 30

        # Basic metrics
        if ear is not None:
            cv2.putText(
                frame,
                f"EAR: {ear:.3f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += line_height

        if gaze is not None:
            cv2.putText(
                frame,
                f"Gaze: {gaze:.3f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y_offset += line_height

        cv2.putText(
            frame,
            f"PERCLOS: {perclos_score:.3f}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        y_offset += line_height

        # Direction information
        cv2.putText(
            frame,
            f"Head: {head_direction.upper()}/{head_vertical_direction.upper()}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if head_direction != "center" or head_vertical_direction != "center" else (255, 255, 255),
            2,
        )
        y_offset += line_height

        cv2.putText(
            frame,
            f"Gaze: {gaze_direction.upper()}/{gaze_vertical_direction.upper()}",
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255) if gaze_direction != "center" or gaze_vertical_direction != "center" else (255, 255, 255),
            2,
        )
        y_offset += line_height

        # Direction durations
        if direction_duration > 0:
            cv2.putText(
                frame,
                f"Looking {direction_tracker.current_direction}: {direction_duration:.1f}s",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 200, 0),
                2,
            )
            y_offset += line_height

        if vertical_direction_duration > 0:
            cv2.putText(
                frame,
                f"Looking {direction_tracker.current_vertical_direction}: {vertical_direction_duration:.1f}s",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 150, 0),
                2,
            )
            y_offset += line_height

        # Head pose angles
        if roll is not None and pitch is not None and yaw is not None:
            cv2.putText(
                frame,
                f"Roll: {roll[0]:.1f}°, Pitch: {pitch[0]:.1f}°, Yaw: {yaw[0]:.1f}°",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )
            y_offset += line_height

        # Special movements
        if args.show_special_movements:
            if mirror_check != "normal":
                cv2.putText(
                    frame,
                    f"{mirror_check.replace('_', ' ').upper()}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                y_offset += line_height

            if shoulder_check != "normal":
                cv2.putText(
                    frame,
                    f"{shoulder_check.replace('_', ' ').upper()}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )
                y_offset += line_height

        # Alert boxes at the bottom
        alert_y = frame.shape[0] - 200
        alert_height = 30

        if tired:
            cv2.putText(
                frame,
                "TIRED!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            alert_y -= alert_height

        if asleep:
            cv2.putText(
                frame,
                "ASLEEP!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            alert_y -= alert_height

        if looking_away:
            cv2.putText(
                frame,
                "LOOKING AWAY!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            alert_y -= alert_height

        if distracted:
            cv2.putText(
                frame,
                "DISTRACTED!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            alert_y -= alert_height

        if extended_left:
            cv2.putText(
                frame,
                "EXTENDED LEFT LOOK!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 100, 255),
                2,
            )
            alert_y -= alert_height

        if extended_right:
            cv2.putText(
                frame,
                "EXTENDED RIGHT LOOK!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 100, 255),
                2,
            )
            alert_y -= alert_height

        if extended_up:
            cv2.putText(
                frame,
                "EXTENDED UP LOOK!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 100, 0),
                2,
            )
            alert_y -= alert_height

        if extended_down:
            cv2.putText(
                frame,
                "EXTENDED DOWN LOOK!",
                (10, alert_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 100, 0),
                2,
            )
            alert_y -= alert_height

        # Draw visual direction arrows if enabled
        if args.show_direction_arrows:
            # Horizontal arrows
            if head_direction == "left":
                cv2.arrowedLine(frame, (100, 300), (50, 300), (0, 255, 255), 3, tipLength=0.3)
                cv2.putText(frame, "←", (30, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif head_direction == "right":
                cv2.arrowedLine(frame, (frame.shape[1] - 100, 300), (frame.shape[1] - 50, 300), 
                              (0, 255, 255), 3, tipLength=0.3)
                cv2.putText(frame, "→", (frame.shape[1] - 40, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Vertical arrows
            if head_vertical_direction == "up":
                cv2.arrowedLine(frame, (frame.shape[1]//2, 250), (frame.shape[1]//2, 200), 
                              (255, 100, 0), 3, tipLength=0.3)
                cv2.putText(frame, "↑", (frame.shape[1]//2 - 10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)
            elif head_vertical_direction == "down":
                cv2.arrowedLine(frame, (frame.shape[1]//2, 350), (frame.shape[1]//2, 400), 
                              (255, 100, 0), 3, tipLength=0.3)
                cv2.putText(frame, "↓", (frame.shape[1]//2 - 10, 420), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2)

        # Send data to frontend periodically
        if args.send_to_frontend and (t_now - last_frontend_update > args.frontend_update_interval):
            detection_data = {
                "timestamp": datetime.now().isoformat(),
                "ear_score": float(ear) if ear is not None else 0.0,
                "gaze_score": float(gaze) if gaze is not None else 0.0,
                "perclos": float(perclos_score),
                "head_direction": head_direction,
                "head_vertical_direction": head_vertical_direction,
                "gaze_direction": gaze_direction,
                "gaze_vertical_direction": gaze_vertical_direction,
                "head_pose": {
                    "roll": float(roll[0]) if roll is not None else 0.0,
                    "pitch": float(pitch[0]) if pitch is not None else 0.0,
                    "yaw": float(yaw[0]) if yaw is not None else 0.0
                },
                "alerts": {
                    "tired": bool(tired),
                    "asleep": bool(asleep),
                    "looking_away": bool(looking_away),
                    "distracted": bool(distracted),
                    "extended_left_look": bool(extended_left),
                    "extended_right_look": bool(extended_right),
                    "extended_up_look": bool(extended_up),
                    "extended_down_look": bool(extended_down)
                },
                "special_movements": {
                    "mirror_check": mirror_check,
                    "shoulder_check": shoulder_check
                },
                "direction_duration": float(direction_duration),
                "vertical_direction_duration": float(vertical_direction_duration),
                "fps": float(fps)
            }
            
            if args.send_to_frontend:
                send_to_frontend(detection_data, args.frontend_host, args.frontend_port)
            
            # Add to log buffer
            log_data_buffer.append(detection_data)
            
            last_frontend_update = t_now

        # Save logs if enabled
        if args.save_logs and data_logger and len(log_data_buffer) > 0:
            for data in log_data_buffer:
                data_logger.log_data(data)
            log_data_buffer.clear()

        # Display FPS and processing time
        if args.show_fps:
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (frame.shape[1] - 150, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )

        if args.show_proc_time:
            e2 = cv2.getTickCount()
            proc_time_ms = ((e2 - e1) / cv2.getTickFrequency()) * 1000
            cv2.putText(
                frame,
                f"Proc: {proc_time_ms:.1f}ms",
                (frame.shape[1] - 150, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

        # Show frame
        cv2.imshow("Driver Attention & Direction Detection - Press 'q' to exit", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Save any remaining log data
    if args.save_logs and data_logger and len(log_data_buffer) > 0:
        for data in log_data_buffer:
            data_logger.log_data(data)


if __name__ == "__main__":
    main()