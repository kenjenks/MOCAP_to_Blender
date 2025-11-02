# 2_FilterData.py (Version 9.0 - Added up/down coordinate system detection and correction)

import os
import sys
import json
import argparse
import numpy as np

# Add the project root to the system path so we can import utils.py
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_check = os.path.normpath(current_script_dir)
    project_root = None

    while True:
        if os.path.exists(os.path.join(path_to_check, "utils.py")):
            project_root = path_to_check
            break

        parent_path = os.path.dirname(path_to_check)
        if parent_path == path_to_check:
            raise FileNotFoundError("Project root (containing 'utils.py') not found in parent directories.")
        path_to_check = parent_path

    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

    from utils import (
        script_log,
        load_global_config,
        get_project_root,
        get_scene_paths,
        get_current_show_name,
        get_current_scene_name,
        set_current_scene_name,
        load_tsv_list,
        get_processing_step_paths
    )

except ImportError as e:
    print("--------------------------------------------------------------------------------------------------")
    print(f"FATAL ERROR: Could not import HMP_SW/utils.py.")
    print(f"Error: {e}")
    print("\nCURRENT EFFECTIVE SYS.PATH:")
    for i, p in enumerate(sys.path):
        print(f"  {i:02d}: {p}")
    print(
        "\nACTION REQUIRED: Please confirm that 'C:\\Users\\ken\\Documents\\Ken\\Fiction\\HMP_SW\\utils.py' exists and defines the function 'script_log'.")
    print("--------------------------------------------------------------------------------------------------")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during utility setup: {e}")
    sys.exit(1)

# Configuration file path
CONFIG_FILE = "filter_data_config.json"


def load_config():
    """Load configuration from JSON file."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        script_log(f"ERROR: Could not load configuration file {CONFIG_FILE}: {e}", force_log=True)
        # Return default config if file not found
        return {
            "coordinate_detection": {
                "sample_frame_count": 10,
                "upside_down_threshold": 0.5,
                "min_samples_required": 5,
                "manual_override": "auto",  # "auto", "force_flip", "force_no_flip"
                "confidence_threshold": 0.7
            },
            "parameters": {
                "smoothing_radius": 3,
                "z_score_threshold": 3.0
            }
        }


def detect_and_correct_coordinate_system(pose_data, config):
    """
    Detects if the coordinate system is upside down by checking head vs feet positions.
    Flips Y coordinates if head appears below feet on average.

    Args:
        pose_data: Dictionary of pose data frames
        config: Configuration dictionary

    Returns:
        Tuple: (corrected_pose_data, was_flipped, confidence)
    """
    if not pose_data:
        return pose_data, False, 0.0

    coord_config = config.get("coordinate_detection", {})
    manual_override = coord_config.get("manual_override", "auto")
    sample_frame_count = coord_config.get("sample_frame_count", 10)
    upside_down_threshold = coord_config.get("upside_down_threshold", 0.5)
    min_samples_required = coord_config.get("min_samples_required", 5)
    confidence_threshold = coord_config.get("confidence_threshold", 0.7)

    # Handle manual overrides
    if manual_override == "force_flip":
        script_log("Manual override: Forcing Y-coordinate flip", force_log=True)
        return flip_y_coordinates(pose_data), True, 1.0
    elif manual_override == "force_no_flip":
        script_log("Manual override: Skipping coordinate correction", force_log=True)
        return pose_data, False, 1.0

    # Check if we have the necessary landmarks
    required_landmarks = ['HEAD_TOP', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']
    frame_keys = sorted([int(k) for k in pose_data.keys()])

    if not frame_keys:
        return pose_data, False, 0.0

    # Sample multiple frames to avoid single-frame anomalies
    sample_count = min(sample_frame_count, len(frame_keys))
    sample_indices = [frame_keys[int(i * len(frame_keys) / sample_count)] for i in range(sample_count)]

    head_above_feet_count = 0
    total_comparisons = 0
    confidence_scores = []

    for idx in sample_indices:
        frame_key = str(idx)
        if (frame_key in pose_data and
                all(landmark in pose_data[frame_key] for landmark in required_landmarks)):

            head_y = pose_data[frame_key]['HEAD_TOP']['y']
            left_foot_y = pose_data[frame_key]['LEFT_FOOT_INDEX']['y']
            right_foot_y = pose_data[frame_key]['RIGHT_FOOT_INDEX']['y']
            head_visibility = pose_data[frame_key]['HEAD_TOP']['visibility']
            left_foot_visibility = pose_data[frame_key]['LEFT_FOOT_INDEX']['visibility']
            right_foot_visibility = pose_data[frame_key]['RIGHT_FOOT_INDEX']['visibility']

            # Only use frames with good visibility
            min_visibility = 0.5
            if (head_visibility >= min_visibility and
                    left_foot_visibility >= min_visibility and
                    right_foot_visibility >= min_visibility):

                avg_foot_y = (left_foot_y + right_foot_y) / 2

                # In correct coordinate system: head_y < foot_y (head is above feet)
                if head_y < avg_foot_y:
                    head_above_feet_count += 1

                total_comparisons += 1

                # Calculate confidence based on distance (larger distance = more confident)
                vertical_distance = abs(head_y - avg_foot_y)
                confidence = min(vertical_distance * 10, 1.0)  # Normalize to 0-1
                confidence_scores.append(confidence)

    if total_comparisons < min_samples_required:
        script_log(
            f"WARNING: Insufficient samples for coordinate detection ({total_comparisons}/{min_samples_required})",
            force_log=True)
        return pose_data, False, 0.0

    upside_down_ratio = head_above_feet_count / total_comparisons
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

    script_log(f"Coordinate detection: Head above feet in {upside_down_ratio * 100:.1f}% of samples", force_log=True)
    script_log(f"Detection confidence: {avg_confidence:.3f}", force_log=True)

    needs_flip = upside_down_ratio < upside_down_threshold
    is_confident = avg_confidence >= confidence_threshold

    if needs_flip and is_confident:
        script_log("Auto-correcting upside-down coordinate system by flipping Y coordinates...", force_log=True)
        return flip_y_coordinates(pose_data), True, avg_confidence
    elif needs_flip and not is_confident:
        script_log("WARNING: Coordinate system appears upside down but confidence is low. No correction applied.",
                   force_log=True)
        script_log("Consider using manual_override: 'force_flip' in config if needed.", force_log=True)
        return pose_data, False, avg_confidence
    else:
        script_log("Coordinate system orientation OK", force_log=True)
        return pose_data, False, avg_confidence


def flip_y_coordinates(pose_data):
    """Flip Y coordinates for all landmarks in all frames."""
    flipped_data = {}

    for frame_key, frame_data in pose_data.items():
        flipped_data[frame_key] = {}
        for landmark_name, landmark_data in frame_data.items():
            flipped_data[frame_key][landmark_name] = {
                'x': landmark_data['x'],
                'y': -landmark_data['y'],  # Flip Y coordinate
                'z': landmark_data['z'],
                'visibility': landmark_data['visibility']
            }

    return flipped_data


def load_pose_data(input_json_path):
    """Load pose data from JSON file."""
    try:
        with open(input_json_path, 'r') as f:
            pose_data = json.load(f)
        script_log(f"Loaded pose data from {input_json_path}")
        return pose_data
    except Exception as e:
        script_log(f"ERROR: Could not load pose data from {input_json_path}: {e}", force_log=True)
        return {}


def save_pose_data(pose_data, output_json_path):
    """Save pose data to JSON file."""
    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(pose_data, f, indent=2)
        script_log(f"Saved filtered pose data to {output_json_path}")
    except Exception as e:
        script_log(f"ERROR: Could not save pose data to {output_json_path}: {e}", force_log=True)


def smooth_trajectory(trajectory, radius=3):
    """Apply moving average smoothing to a trajectory."""
    if len(trajectory) <= radius * 2:
        return trajectory

    smoothed = []
    for i in range(len(trajectory)):
        start = max(0, i - radius)
        end = min(len(trajectory), i + radius + 1)
        window = trajectory[start:end]
        smoothed.append(np.mean(window))
    return smoothed


def filter_pose_data(input_json_path, output_json_path, config):
    """Main filtering function with coordinate system correction and smoothing."""
    # Load raw pose data
    pose_data = load_pose_data(input_json_path)
    if not pose_data:
        script_log("ERROR: No pose data loaded", force_log=True)
        return False

    original_frame_count = len(pose_data)
    script_log(f"Processing {original_frame_count} frames", force_log=True)

    # Step 1: Detect and correct coordinate system
    pose_data, was_flipped, confidence = detect_and_correct_coordinate_system(pose_data, config)

    # Step 2: Extract trajectories for smoothing
    landmarks = list(pose_data['0'].keys()) if '0' in pose_data else []
    if not landmarks:
        script_log("ERROR: No landmarks found in pose data", force_log=True)
        return False

    # Convert to numpy arrays for processing
    trajectories = {coord: {landmark: [] for landmark in landmarks} for coord in ['x', 'y', 'z']}
    visibilities = {landmark: [] for landmark in landmarks}

    frame_keys = sorted([int(k) for k in pose_data.keys()])
    for frame_key in frame_keys:
        frame_data = pose_data[str(frame_key)]
        for landmark in landmarks:
            if landmark in frame_data:
                trajectories['x'][landmark].append(frame_data[landmark]['x'])
                trajectories['y'][landmark].append(frame_data[landmark]['y'])
                trajectories['z'][landmark].append(frame_data[landmark]['z'])
                visibilities[landmark].append(frame_data[landmark]['visibility'])
            else:
                # Fill with zeros if landmark missing
                trajectories['x'][landmark].append(0.0)
                trajectories['y'][landmark].append(0.0)
                trajectories['z'][landmark].append(0.0)
                visibilities[landmark].append(0.0)

    # Step 3: Apply smoothing
    smoothing_radius = config.get("parameters", {}).get("smoothing_radius", 3)
    script_log(f"Applying smoothing with radius {smoothing_radius}", force_log=True)

    smoothed_trajectories = {coord: {} for coord in ['x', 'y', 'z']}
    for coord in ['x', 'y', 'z']:
        for landmark in landmarks:
            smoothed_trajectories[coord][landmark] = smooth_trajectory(
                trajectories[coord][landmark], smoothing_radius
            )

    # Step 4: Reconstruct pose data with smoothed values
    filtered_pose_data = {}
    for i, frame_key in enumerate(frame_keys):
        filtered_pose_data[str(frame_key)] = {}
        for landmark in landmarks:
            filtered_pose_data[str(frame_key)][landmark] = {
                'x': float(smoothed_trajectories['x'][landmark][i]),
                'y': float(smoothed_trajectories['y'][landmark][i]),
                'z': float(smoothed_trajectories['z'][landmark][i]),
                'visibility': float(visibilities[landmark][i])
            }

    # Step 5: Save filtered data
    save_pose_data(filtered_pose_data, output_json_path)

    script_log(f"Filtering complete: {len(filtered_pose_data)} frames processed", force_log=True)
    script_log(f"Coordinate system was {'corrected' if was_flipped else 'unchanged'}", force_log=True)

    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Filter and smooth pose data with coordinate system correction.")
    parser.add_argument('--show', type=str, default=None, help='Override the default show name (e.g., "aae").')
    parser.add_argument('--scene', type=str, default=None,
                        help='Override the current scene name (e.g., "Scene-80000").')
    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Set up context
    show_name = args.show if args.show else get_current_show_name()
    scene_name = args.scene if args.scene else get_current_scene_name(show_name)

    if not show_name:
        script_log("FATAL ERROR: Could not determine show name. Check production_specific_config.json.", force_log=True)
        sys.exit(1)

    script_log(f"\n=== POSE FILTERING STARTED: Show '{show_name}', Scene '{scene_name}' ===", force_log=True)

    # Get file paths USING PROCESSING STEPS CONFIG
    try:
        paths = get_processing_step_paths(show_name, scene_name, 'filter_data')
        input_json_path = paths['input_file']
        output_json_path = paths['output_file']

        script_log(f"Input JSON Path: {input_json_path}", force_log=True)
        script_log(f"Output JSON Path: {output_json_path}", force_log=True)

    except Exception as e:
        script_log(f"FATAL ERROR during path lookup: {e}", force_log=True)
        sys.exit(1)

    # Check if input file exists
    if not os.path.exists(input_json_path):
        script_log(f"FATAL ERROR: Input JSON file not found at: {input_json_path}", force_log=True)
        sys.exit(1)

    # Run filtering
    success = filter_pose_data(input_json_path, output_json_path, config)

    if success:
        script_log("=== FILTERING COMPLETED SUCCESSFULLY ===", force_log=True)
    else:
        script_log("=== FILTERING FAILED ===", force_log=True)
        sys.exit(1)


if __name__ == "__main__":
    main()