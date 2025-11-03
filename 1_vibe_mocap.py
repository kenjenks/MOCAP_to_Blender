# 1_vibe_mocap.py (Version 1.0 - First VIBE)
"""
VIBE-based motion capture extraction
Uses VIBE (Video Inference for Human Body Pose and Shape) for 3D pose estimation
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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




def setup_vibe_environment():
    """
    Set up VIBE environment and check dependencies
    """
    try:
        # Try to import VIBE dependencies
        import torch
        import torchvision
        import smplx
        from lib.models import VIBE
        from lib.dataset.inference import Inference
        from lib.utils.demo_utils import (
            convert_crop_cam_to_orig_img,
            prepare_rendering_results,
            video_to_images,
            images_to_video,
        )
        script_log("VIBE dependencies loaded successfully")
        return True
    except ImportError as e:
        script_log(f"VIBE dependencies not available: {e}")
        script_log("Please install VIBE requirements: pip install torch torchvision smplx")
        return False


def run_vibe_inference(video_path, output_dir, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Run VIBE inference on video file
    """
    try:
        script_log(f"Starting VIBE inference on: {video_path}")
        script_log(f"Using device: {device}")

        # VIBE inference code would go here
        # This is a placeholder for the actual VIBE implementation

        # For now, we'll create a mock implementation
        vibe_output = mock_vibe_inference(video_path)

        return vibe_output

    except Exception as e:
        script_log(f"Error during VIBE inference: {e}")
        raise


def mock_vibe_inference(video_path):
    """
    Mock VIBE inference for testing - replace with actual VIBE code
    """
    script_log(f"Mock VIBE inference on: {video_path}")

    # This would be replaced with actual VIBE output
    # For now, return a mock output structure
    mock_output = {
        'joints3d': [],  # This would contain the actual 3D joint positions
        'vertices': [],  # 3D mesh vertices
        'pose': [],  # SMPL pose parameters
        'betas': [],  # SMPL shape parameters
        'cameras': []  # Camera parameters
    }

    return mock_output


def convert_vibe_to_mediapipe_format(vibe_output, frame_count):
    """
    Convert VIBE output to MediaPipe JSON format
    This is a simplified conversion - will need refinement based on actual VIBE output
    """
    script_log("Converting VIBE output to MediaPipe format")

    # This mapping will need to be refined based on actual VIBE joint order
    vibe_to_mediapipe_mapping = {
        # Basic body joints - this is a simplified mapping
        0: 'NOSE',  # Approximate - VIBE doesn't have exact nose joint
        11: 'LEFT_SHOULDER',
        12: 'RIGHT_SHOULDER',
        13: 'LEFT_ELBOW',
        14: 'RIGHT_ELBOW',
        15: 'LEFT_WRIST',
        16: 'RIGHT_WRIST',
        23: 'LEFT_HIP',
        24: 'RIGHT_HIP',
        25: 'LEFT_KNEE',
        26: 'RIGHT_KNEE',
        27: 'LEFT_ANKLE',
        28: 'RIGHT_ANKLE',
    }

    mediapipe_data = {}

    # Create mock data for each frame
    for frame_idx in range(frame_count):
        frame_data = {}

        for vibe_idx, mp_name in vibe_to_mediapipe_mapping.items():
            # Create mock coordinates - replace with actual VIBE joint data
            frame_data[mp_name] = {
                'x': 0.0,  # These would come from vibe_output['joints3d'][frame_idx][vibe_idx]
                'y': 0.0,
                'z': 0.0,
                'visibility': 1.0
            }

        mediapipe_data[str(frame_idx)] = frame_data

    return mediapipe_data


def main():
    """
    Main function to run VIBE motion capture extraction
    """
    script_log("Starting VIBE motion capture extraction")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VIBE Motion Capture Extraction')
    parser.add_argument('--show', type=str, help='Show name (optional)')
    parser.add_argument('--scene', type=str, help='Scene name (optional)')
    parser.add_argument('--video', type=str, help='Input video path (optional)')
    parser.add_argument('--output', type=str, help='Output JSON path (optional)')

    args = parser.parse_args()

    try:
        # Determine show and scene names
        show_name = args.show or get_current_show_name()
        scene_name = args.scene or get_current_scene_name(show_name)

        script_log(f"Processing show: {show_name}, scene: {scene_name}")

        # Get scene paths
        scene_paths = get_scene_paths(show_name, scene_name)
        input_video = args.video or scene_paths['input_video']
        output_json = args.output or scene_paths['output_pose_data'].replace('.json', '_VIBE.json')

        script_log(f"Input video: {input_video}")
        script_log(f"Output JSON: {output_json}")

        # Check if input video exists
        if not os.path.exists(input_video):
            script_log(f"Error: Input video not found: {input_video}")
            return 1

        # Setup VIBE environment
        if not setup_vibe_environment():
            script_log("VIBE environment setup failed")
            return 1

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_json)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Run VIBE inference
        vibe_output = run_vibe_inference(input_video, output_dir)

        # Estimate frame count (this would come from actual VIBE output)
        # For now, use a placeholder - you might need video analysis to get actual frame count
        frame_count = 100  # Placeholder

        # Convert to MediaPipe format
        mediapipe_data = convert_vibe_to_mediapipe_format(vibe_output, frame_count)

        # Save output
        with open(output_json, 'w') as f:
            json.dump(mediapipe_data, f, indent=2)

        script_log(f"VIBE motion capture completed: {output_json}")
        script_log(f"Generated {len(mediapipe_data)} frames of pose data")

        return 0

    except Exception as e:
        script_log(f"Error in VIBE motion capture: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)