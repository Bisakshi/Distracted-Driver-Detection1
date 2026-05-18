
# import argparse


# def get_args():
#     parser = argparse.ArgumentParser(description="Driver State Detection with Direction Detection")

#     # selection the camera number, default is 0 (webcam)
#     parser.add_argument(
#         "-c",
#         "--camera",
#         type=int,
#         default=0,
#         metavar="",
#         help="Camera number, default is 0 (webcam)",
#     )

#     parser.add_argument(
#         "--camera_params",
#         type=str,
#         help="Path to the camera parameters file (JSON or YAML).",
#     )

#     # visualisation parameters
#     parser.add_argument(
#         "--show_fps",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Show the actual FPS of the capture stream, default is true",
#     )
#     parser.add_argument(
#         "--show_proc_time",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Show the processing time for a single frame, default is true",
#     )
#     parser.add_argument(
#         "--show_eye_proc",
#         type=bool,
#         default=False,
#         metavar="",
#         help="Show the eyes processing, default is false",
#     )
#     parser.add_argument(
#         "--show_axis",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Show the head pose axis, default is true",
#     )
#     parser.add_argument(
#         "--verbose",
#         type=bool,
#         default=False,
#         metavar="",
#         help="Prints additional info, default is false",
#     )
#     parser.add_argument(
#         "--show_direction_arrows",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Show visual arrows for head/gaze direction, default is true",
#     )
#     parser.add_argument(
#         "--show_special_movements",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Show special movements detection (mirror/shoulder checks), default is true",
#     )

#     # Attention Scorer parameters (EAR, Gaze Score, Pose)
#     parser.add_argument(
#         "--smooth_factor",
#         type=float,
#         default=0.5,
#         metavar="",
#         help="Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5",
#     )
#     parser.add_argument(
#         "--ear_thresh",
#         type=float,
#         default=0.15,
#         metavar="",
#         help="Sets the EAR threshold for the Attention Scorer, default is 0.15",
#     )
#     parser.add_argument(
#         "--ear_time_thresh",
#         type=float,
#         default=2,
#         metavar="",
#         help="Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds",
#     )
#     parser.add_argument(
#         "--gaze_thresh",
#         type=float,
#         default=0.015,
#         metavar="",
#         help="Sets the Gaze Score threshold for the Attention Scorer, default is 0.2",
#     )
#     parser.add_argument(
#         "--gaze_time_thresh",
#         type=float,
#         default=2,
#         metavar="",
#         help="Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2 seconds",
#     )
#     parser.add_argument(
#         "--pitch_thresh",
#         type=float,
#         default=20,
#         metavar="",
#         help="Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees",
#     )
#     parser.add_argument(
#         "--yaw_thresh",
#         type=float,
#         default=20,
#         metavar="",
#         help="Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees",
#     )
#     parser.add_argument(
#         "--roll_thresh",
#         type=float,
#         default=20,
#         metavar="",
#         help="Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees",
#     )
#     parser.add_argument(
#         "--pose_time_thresh",
#         type=float,
#         default=2.5,
#         metavar="",
#         help="Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds",
#     )

#     # Direction Detection parameters
#     parser.add_argument(
#         "--direction_time_thresh",
#         type=float,
#         default=2.0,
#         metavar="",
#         help="Time threshold for extended direction looks (seconds), default is 2.0 seconds",
#     )
#     parser.add_argument(
#         "--head_direction_thresh",
#         type=float,
#         default=20.0,
#         metavar="",
#         help="Head yaw angle threshold for left/right direction (degrees), default is 20.0 degrees",
#     )
#     parser.add_argument(
#         "--gaze_direction_thresh",
#         type=float,
#         default=0.2,
#         metavar="",
#         help="Gaze ratio threshold for left/right direction, default is 0.2",
#     )
#     parser.add_argument(
#         "--mirror_check_thresh",
#         type=float,
#         default=30.0,
#         metavar="",
#         help="Yaw threshold for mirror check detection (degrees), default is 30.0 degrees",
#     )
#     parser.add_argument(
#         "--shoulder_check_thresh",
#         type=float,
#         default=60.0,
#         metavar="",
#         help="Yaw threshold for shoulder check detection (degrees), default is 60.0 degrees",
#     )
#     parser.add_argument(
#         "--enable_mirror_check",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Enable mirror check detection, default is true",
#     )
#     parser.add_argument(
#         "--enable_shoulder_check",
#         type=bool,
#         default=True,
#         metavar="",
#         help="Enable shoulder check detection, default is true",
#     )

#     # Frontend/Output parameters
#     parser.add_argument(
#         "--send_to_frontend",
#         type=bool,
#         default=False,
#         metavar="",
#         help="Send detection data to frontend, default is false",
#     )
#     parser.add_argument(
#         "--frontend_host",
#         type=str,
#         default="localhost",
#         metavar="",
#         help="Frontend server host, default is localhost",
#     )
#     parser.add_argument(
#         "--frontend_port",
#         type=int,
#         default=5000,
#         metavar="",
#         help="Frontend server port, default is 5000",
#     )
#     parser.add_argument(
#         "--frontend_update_interval",
#         type=float,
#         default=0.5,
#         metavar="",
#         help="Frontend data update interval (seconds), default is 0.5 seconds",
#     )
#     parser.add_argument(
#         "--save_logs",
#         type=bool,
#         default=False,
#         metavar="",
#         help="Save detection logs to file, default is false",
#     )
#     parser.add_argument(
#         "--log_file",
#         type=str,
#         default="driver_detection_log.csv",
#         metavar="",
#         help="Log file path, default is driver_detection_log.csv",
#     )
#     parser.add_argument(
#         "--alert_sound",
#         type=bool,
#         default=False,
#         metavar="",
#         help="Play alert sound for extended left/right looks, default is false",
#     )

#     # parse the arguments and store them in the args variable dictionary
#     args, _ = parser.parse_known_args()

#     return args


import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Driver State Detection with Direction Detection")

    # selection the camera number, default is 0 (webcam)
    parser.add_argument(
        "-c",
        "--camera",
        type=int,
        default=0,
        metavar="",
        help="Camera number, default is 0 (webcam)",
    )

    parser.add_argument(
        "--camera_params",
        type=str,
        help="Path to the camera parameters file (JSON or YAML).",
    )

    # visualisation parameters
    parser.add_argument(
        "--show_fps",
        type=bool,
        default=True,
        metavar="",
        help="Show the actual FPS of the capture stream, default is true",
    )
    parser.add_argument(
        "--show_proc_time",
        type=bool,
        default=True,
        metavar="",
        help="Show the processing time for a single frame, default is true",
    )
    parser.add_argument(
        "--show_eye_proc",
        type=bool,
        default=False,
        metavar="",
        help="Show the eyes processing, default is false",
    )
    parser.add_argument(
        "--show_axis",
        type=bool,
        default=True,
        metavar="",
        help="Show the head pose axis, default is true",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        metavar="",
        help="Prints additional info, default is false",
    )
    parser.add_argument(
        "--show_direction_arrows",
        type=bool,
        default=True,
        metavar="",
        help="Show visual arrows for head/gaze direction, default is true",
    )
    parser.add_argument(
        "--show_special_movements",
        type=bool,
        default=True,
        metavar="",
        help="Show special movements detection (mirror/shoulder checks), default is true",
    )

    # Attention Scorer parameters (EAR, Gaze Score, Pose)
    parser.add_argument(
        "--smooth_factor",
        type=float,
        default=0.5,
        metavar="",
        help="Sets the smooth factor for the head pose estimation keypoint smoothing, default is 0.5",
    )
    parser.add_argument(
        "--ear_thresh",
        type=float,
        default=0.15,
        metavar="",
        help="Sets the EAR threshold for the Attention Scorer, default is 0.15",
    )
    parser.add_argument(
        "--ear_time_thresh",
        type=float,
        default=2,
        metavar="",
        help="Sets the EAR time (seconds) threshold for the Attention Scorer, default is 2 seconds",
    )
    parser.add_argument(
        "--gaze_thresh",
        type=float,
        default=0.015,
        metavar="",
        help="Sets the Gaze Score threshold for the Attention Scorer, default is 0.2",
    )
    parser.add_argument(
        "--gaze_time_thresh",
        type=float,
        default=2,
        metavar="",
        help="Sets the Gaze Score time (seconds) threshold for the Attention Scorer, default is 2 seconds",
    )
    parser.add_argument(
        "--pitch_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the PITCH threshold (degrees) for the Attention Scorer, default is 30 degrees",
    )
    parser.add_argument(
        "--yaw_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the YAW threshold (degrees) for the Attention Scorer, default is 20 degrees",
    )
    parser.add_argument(
        "--roll_thresh",
        type=float,
        default=20,
        metavar="",
        help="Sets the ROLL threshold (degrees) for the Attention Scorer, default is 30 degrees",
    )
    parser.add_argument(
        "--pose_time_thresh",
        type=float,
        default=2.5,
        metavar="",
        help="Sets the Pose time threshold (seconds) for the Attention Scorer, default is 2.5 seconds",
    )

    # Direction Detection parameters
    parser.add_argument(
        "--direction_time_thresh",
        type=float,
        default=2.0,
        metavar="",
        help="Time threshold for extended direction looks (seconds), default is 2.0 seconds",
    )
    parser.add_argument(
        "--vertical_time_thresh",
        type=float,
        default=1.5,
        metavar="",
        help="Time threshold for extended up/down looks (seconds), default is 1.5 seconds",
    )
    parser.add_argument(
        "--head_direction_thresh",
        type=float,
        default=20.0,
        metavar="",
        help="Head yaw angle threshold for left/right direction (degrees), default is 20.0 degrees",
    )
    parser.add_argument(
        "--head_vertical_thresh",
        type=float,
        default=15.0,
        metavar="",
        help="Head pitch angle threshold for up/down direction (degrees), default is 15.0 degrees",
    )
    parser.add_argument(
        "--gaze_direction_thresh",
        type=float,
        default=0.2,
        metavar="",
        help="Gaze ratio threshold for left/right direction, default is 0.2",
    )
    parser.add_argument(
        "--gaze_vertical_thresh",
        type=float,
        default=0.15,
        metavar="",
        help="Gaze ratio threshold for up/down direction, default is 0.15",
    )
    parser.add_argument(
        "--mirror_check_thresh",
        type=float,
        default=30.0,
        metavar="",
        help="Yaw threshold for mirror check detection (degrees), default is 30.0 degrees",
    )
    parser.add_argument(
        "--shoulder_check_thresh",
        type=float,
        default=60.0,
        metavar="",
        help="Yaw threshold for shoulder check detection (degrees), default is 60.0 degrees",
    )
    parser.add_argument(
        "--enable_mirror_check",
        type=bool,
        default=True,
        metavar="",
        help="Enable mirror check detection, default is true",
    )
    parser.add_argument(
        "--enable_shoulder_check",
        type=bool,
        default=True,
        metavar="",
        help="Enable shoulder check detection, default is true",
    )

    # Frontend/Output parameters
    parser.add_argument(
        "--send_to_frontend",
        type=bool,
        default=False,
        metavar="",
        help="Send detection data to frontend, default is false",
    )
    parser.add_argument(
        "--frontend_host",
        type=str,
        default="localhost",
        metavar="",
        help="Frontend server host, default is localhost",
    )
    parser.add_argument(
        "--frontend_port",
        type=int,
        default=5000,
        metavar="",
        help="Frontend server port, default is 5000",
    )
    parser.add_argument(
        "--frontend_update_interval",
        type=float,
        default=0.5,
        metavar="",
        help="Frontend data update interval (seconds), default is 0.5 seconds",
    )
    parser.add_argument(
        "--save_logs",
        type=bool,
        default=False,
        metavar="",
        help="Save detection logs to file, default is false",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="driver_detection_log.csv",
        metavar="",
        help="Log file path, default is driver_detection_log.csv",
    )
    parser.add_argument(
        "--alert_sound",
        type=bool,
        default=False,
        metavar="",
        help="Play alert sound for extended looks, default is false",
    )

    # parse the arguments and store them in the args variable dictionary
    args, _ = parser.parse_known_args()

    return args