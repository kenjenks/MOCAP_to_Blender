# 1_ExtractPoseData.py (Version 4.0 - Specified the image dimensions)

import os
import cv2
import json
import sys
import warnings
import logging
import csv
import argparse
from datetime import datetime
import importlib.util  # Added for advanced module checking if needed later

# --- Import Project Utilities ---
# Add the project root (HMP_SW) to the system path so we can import utils.py
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Normalize path for better compatibility (e.g., C:\...)
    path_to_check = os.path.normpath(current_script_dir)
    project_root = None

    # Robustly traverse upwards until 'utils.py' is found
    while True:
        # Check if 'utils.py' exists in the current directory being checked
        if os.path.exists(os.path.join(path_to_check, "utils.py")):
            project_root = path_to_check
            break

        parent_path = os.path.dirname(path_to_check)

        # If we reached the filesystem root (path doesn't change), stop
        if parent_path == path_to_check:
            raise FileNotFoundError("Project root (containing 'utils.py') not found in parent directories.")

        path_to_check = parent_path

    # Add the found project root to the system path
    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

    # Now import the utilities file from the dynamically found project root
    # Note: If this still fails, you must verify the contents of the utils.py file itself.
    from utils import (
        script_log,
        load_global_config,
        get_project_root,
        get_scene_paths,
        get_current_show_name,
        get_current_scene_name,
        set_current_scene_name,
        load_tsv_list,
        get_processing_step_paths,
        load_landmark_config
    )

except ImportError as e:
    # This specifically catches the import failure
    print("--------------------------------------------------------------------------------------------------")
    print(f"FATAL ERROR: Could not import HMP_SW/utils.py.")
    print(f"Error: {e}")
    # Print effective sys.path to aid debugging shadowing issues
    print("\nCURRENT EFFECTIVE SYS.PATH (Check this list for other 'utils' module locations):")
    for i, p in enumerate(sys.path):
        print(f"  {i:02d}: {p}")
    print(
        "\nACTION REQUIRED: Please confirm that 'C:\\Users\\ken\\Documents\\Ken\\Fiction\\HMP_SW\\utils.py' exists and defines the function 'script_log'.")
    print("--------------------------------------------------------------------------------------------------")
    sys.exit(1)
except FileNotFoundError as e:
    # This catches the robust path search failure
    print(f"FATAL ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during utility setup: {e}")
    sys.exit(1)

# === TensorFlow/MediaPipe imports ===
try:
    import tensorflow as tf
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from absl import logging as absl_logging
except ImportError as e:
    script_log(f"FATAL ERROR: Required library not found: {e}", force_log=True)
    script_log(
        "Please ensure TensorFlow and MediaPipe are installed (e.g., pip install mediapipe opencv-python tensorflow)",
        force_log=True)
    sys.exit(1)

# Suppress unnecessary warnings and set logging levels for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)
absl_logging.set_verbosity(absl_logging.ERROR)

LANDMARK_CONFIG_FILE = "0_RunMocapAnimPipeline_LANDMARKS.json"


def configure_tensorflow():
    """Configures TensorFlow for optimal MediaPipe performance on CPU."""
    # Enable CPU optimizations
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
    # Set thread count (capped at 4 to prevent overutilization)
    os.environ['OMP_NUM_THREADS'] = str(min(4, os.cpu_count() or 4))

    # Configure TensorFlow settings
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)  # Suppress autograph warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def check_system_config():
    """Logs system information for context and debugging."""
    try:
        script_log(f"System: {sys.platform} | Python: {sys.version.split(' ')[0]}")
        script_log(f"TensorFlow Version: {tf.__version__} | MediaPipe Version: {mp.__version__}")
        script_log(f"OpenCV Version: {cv2.__version__}")
        script_log(f"CPU Count: {os.cpu_count()}")
    except Exception as e:
        script_log(f"Warning: Could not check system configuration details: {e}")


def extract_pose_data(input_video_path, output_json_path, model_asset_path):
    """
    Performs pose detection on a video file using MediaPipe Pose Landmarker.
    Outputs raw pose data without coordinate system corrections.

    Args:
        input_video_path (str): Absolute path to the input video file.
        output_json_path (str): Absolute path to the output JSON file.
        model_asset_path (str): Absolute path to the MediaPipe model file.

    Returns:
        int: The number of frames successfully processed.
    """
    script_log(f"Using MediaPipe model: {model_asset_path}")

    # Load landmark configuration from utils with specific file path
    landmark_config = load_landmark_config(LANDMARK_CONFIG_FILE)
    LANDMARK_INDICES = {}
    for landmark_name, landmark_data in landmark_config["landmarks"].items():
        LANDMARK_INDICES[landmark_name] = landmark_data["index"]

    script_log(f"Processing video: {input_video_path}")
    script_log(f"Loaded {len(LANDMARK_INDICES)} landmarks from config")
    script_log(f"Config file: {os.path.abspath(f"./{LANDMARK_CONFIG_FILE}")}")
    script_log(f"Config file MediaPipe version: {landmark_config.get('mediapipe_version', 'unknown')}")

    # --- 1. Setup Landmarker ---
    try:
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            # Set to VIDEO mode for efficient sequential processing
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        landmarker = vision.PoseLandmarker.create_from_options(options)
    except Exception as e:
        script_log(
            f"FATAL ERROR: Could not initialize MediaPipe Pose Landmarker. Check model path and integrity. Error: {e}",
            force_log=True)
        return 0

    # --- 2. Setup Video Capture ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        script_log(f"FATAL ERROR: Could not open video file at: {input_video_path}", force_log=True)
        landmarker.close()
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    script_log(f"Total Frames: {total_frames}, FPS: {fps:.2f}")

    pose_data = {}
    frame_count = 0

    # --- 3. Process Video Frames ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        height, width, _ = frame.shape
        timestamp_ms = int(frame_count * 1000 / fps)

        # Convert the frame to a MediaPipe Image object with explicit dimensions
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        # Perform pose detection
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Store pose data for the frame - only the specific landmarks we want
        frame_key = str(frame_count)
        pose_data[frame_key] = {}

        if results.pose_landmarks and results.pose_landmarks[0]:  # Check if a person was detected
            landmarks = results.pose_landmarks[0]

            # Extract only the specific landmarks we need - NO COORDINATE CORRECTIONS
            for landmark_name, landmark_index in LANDMARK_INDICES.items():
                if landmark_index < len(landmarks):
                    landmark = landmarks[landmark_index]

                    # Use raw MediaPipe coordinates - coordinate correction happens in step 2
                    pose_data[frame_key][landmark_name] = {
                        "x": round(float(landmark.x), 4),      # Raw X coordinate
                        "y": round(float(landmark.y), 4),      # Raw Y coordinate (no flip)
                        "z": round(float(landmark.z), 4),      # Raw Z coordinate
                        "visibility": round(float(landmark.visibility), 2)
                    }

                else:
                    # If landmark index is out of range, add empty data
                    pose_data[frame_key][landmark_name] = {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.0,
                        "visibility": 0.0
                    }

        frame_count += 1
        if frame_count % 100 == 0:
            progress = (frame_count / total_frames) * 100
            script_log(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

    # Cleanup
    cap.release()
    landmarker.close()

    # Save results to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(pose_data, f, indent=2)

    script_log("\n=== PROCESSING COMPLETE ===")
    script_log(f"Saved RAW pose data for {frame_count} frames to {output_json_path}")

    return frame_count

def main():
    # --- 1. Initialize and Parse Arguments ---
    parser = argparse.ArgumentParser(description="Extracts 3D pose data from a video file using MediaPipe.")
    parser.add_argument('--show', type=str, default=None, help='Override the default show name (e.g., "aae").')
    parser.add_argument('--scene', type=str, default=None,
                        help='Override the current scene name (e.g., "Scene-80000").')
    args = parser.parse_args()

    # --- 2. Configure System ---
    configure_tensorflow()
    check_system_config()

    # --- 3. Load Configurations and Set Context ---
    # utils.py handles which show/scene is active based on production_specific_config.json or arguments
    show_name = args.show if args.show else get_current_show_name()
    scene_name = args.scene if args.scene else get_current_scene_name(show_name)

    if not show_name:
        script_log("FATAL ERROR: Could not determine show name. Check production_specific_config.json.", force_log=True)
        sys.exit(1)

    script_log(f"\n=== POSE EXTRACTION STARTED: Show '{show_name}', Scene '{scene_name}' ===")

    # --- 4. Get File Paths from Utilities ---
    try:
        # Get processing step paths - this reads from scene-config.json
        paths = get_processing_step_paths(show_name, scene_name, 'extract_pose')
        input_video_path = paths['input_file']
        output_json_path = paths['output_file']

        # The pose model path is relative to the HMP_SW root
        global_config = load_global_config()
        model_asset_path = os.path.join(get_project_root(), global_config['pose_model_dir_relative_to_root'],
                                        'pose_landmarker_heavy.task')

    except Exception as e:
        script_log(f"FATAL ERROR during path/config lookup: {e}", force_log=True)
        sys.exit(1)

    # --- 5. Log Parameters ---
    script_log(f"Input Video Path: {input_video_path}")
    script_log(f"Output JSON Path: {output_json_path}")
    script_log(f"MediaPipe Model Path: {model_asset_path}")

    # --- 6. Execute Extraction ---
    if not os.path.exists(input_video_path):
        script_log(f"FATAL ERROR: Input video file not found at: {input_video_path}", force_log=True)
        sys.exit(1)

    if not os.path.exists(model_asset_path):
        script_log(f"FATAL ERROR: MediaPipe model file not found at: {model_asset_path}", force_log=True)
        script_log(
            "Please download 'pose_landmarker_heavy.task' and place it in the 'models/' folder at the project root.",
            force_log=True)
        sys.exit(1)

    # Run the main pose extraction logic
    extract_pose_data(input_video_path, output_json_path, model_asset_path)


if __name__ == "__main__":
    main()