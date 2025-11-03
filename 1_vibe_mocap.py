# 1_vibe_mocap.py (version 1.0 = First VIBE)
"""
VIBE-based motion capture extraction
Uses VIBE (Video Inference for Human Body Pose and Shape) for 3D pose estimation
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Add the project root to the Python path to ensure utils can be imported
script_dir = Path(__file__).parent
project_root = script_dir  # Adjust if your project root is different
sys.path.insert(0, str(project_root))

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
        import cv2
        import numpy as np

        # Add VIBE to Python path
        vibe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VIBE')
        if os.path.exists(vibe_dir):
            sys.path.insert(0, vibe_dir)

            # Add lib subdirectory specifically
            lib_dir = os.path.join(vibe_dir, 'lib')
            if os.path.exists(lib_dir):
                sys.path.insert(0, lib_dir)

            try:
                # Try to import VIBE modules
                from lib.models import VIBE
                from lib.dataset.inference import Inference
                if utils_available:
                    script_log("VIBE dependencies loaded successfully")
                else:
                    print("VIBE dependencies loaded successfully")
                return True
            except ImportError as e:
                if utils_available:
                    script_log(f"VIBE modules import failed: {e}")
                    script_log("This is normal if VIBE model weights are not downloaded yet")
                else:
                    print(f"VIBE modules import failed: {e}")
                    print("This is normal if VIBE model weights are not downloaded yet")
                return False
        else:
            if utils_available:
                script_log(f"VIBE directory not found at: {vibe_dir}")
            else:
                print(f"VIBE directory not found at: {vibe_dir}")
            return False

    except ImportError as e:
        if utils_available:
            script_log(f"VIBE dependencies not available: {e}")
            script_log("Please install: pip install torch torchvision smplx opencv-python numpy")
        else:
            print(f"VIBE dependencies not available: {e}")
            print("Please install: pip install torch torchvision smplx opencv-python numpy")
        return False


def download_vibe_model_weights():
    """
    Download VIBE pre-trained model weights
    """
    try:
        import gdown
        import torch

        # Create models directory
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VIBE', 'data')
        os.makedirs(model_dir, exist_ok=True)

        # VIBE model weights URL from official repository
        model_url = 'https://drive.google.com/uc?id=1X0-8OYbq8--3Q9pcVGfnnf6MLDq_0LOK'
        model_path = os.path.join(model_dir, 'vibe_model_w_3dpw.pth')

        if not os.path.exists(model_path):
            if utils_available:
                script_log("Downloading VIBE model weights...")
            else:
                print("Downloading VIBE model weights...")
            gdown.download(model_url, model_path, quiet=False)
            if utils_available:
                script_log(f"Model weights downloaded to: {model_path}")
            else:
                print(f"Model weights downloaded to: {model_path}")
        else:
            if utils_available:
                script_log(f"Model weights already exist at: {model_path}")
            else:
                print(f"Model weights already exist at: {model_path}")

        return model_path

    except ImportError:
        if utils_available:
            script_log("gdown not available. Please install: pip install gdown")
        else:
            print("gdown not available. Please install: pip install gdown")
        return None
    except Exception as e:
        if utils_available:
            script_log(f"Error downloading model weights: {e}")
        else:
            print(f"Error downloading model weights: {e}")
        return None


def run_vibe_inference(video_path, output_dir):
    """
    Run VIBE inference on video file
    """
    try:
        if utils_available:
            script_log(f"Starting VIBE inference on: {video_path}")
        else:
            print(f"Starting VIBE inference on: {video_path}")

        # Import here to avoid issues if VIBE not available
        import torch
        from lib.models import VIBE
        from lib.dataset.inference import Inference
        import cv2
        import numpy as np

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if utils_available:
            script_log(f"Using device: {device}")
        else:
            print(f"Using device: {device}")

        # Download model weights
        model_path = download_vibe_model_weights()
        if not model_path:
            if utils_available:
                script_log("Could not download model weights, using mock implementation")
            else:
                print("Could not download model weights, using mock implementation")
            return mock_vibe_inference(video_path)

        # Initialize VIBE model
        model = VIBE(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(device)

        # Load pre-trained weights
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()

        if utils_available:
            script_log("VIBE model loaded successfully")
        else:
            print("VIBE model loaded successfully")

        # Initialize inference module
        # Note: You may need to provide proper config here
        inference = Inference(
            model=model,
            config=None,  # You might need to create a config object
            device=device,
        )

        # Run inference
        if utils_available:
            script_log("Running VIBE inference...")
        else:
            print("Running VIBE inference...")
        results = inference.run_on_video(video_path)

        return results

    except Exception as e:
        if utils_available:
            script_log(f"Error during VIBE inference: {e}")
            import traceback
            script_log(f"Traceback: {traceback.format_exc()}")
            script_log("Falling back to mock implementation")
        else:
            print(f"Error during VIBE inference: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            print("Falling back to mock implementation")
        # Fall back to mock implementation for testing
        return mock_vibe_inference(video_path)


def mock_vibe_inference(video_path):
    """
    Mock VIBE inference for testing - replace with actual VIBE code
    """
    if utils_available:
        script_log(f"Mock VIBE inference on: {video_path}")
    else:
        print(f"Mock VIBE inference on: {video_path}")

    # Get video info to determine frame count
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if utils_available:
        script_log(f"Video info: {frame_count} frames, {fps} FPS")
    else:
        print(f"Video info: {frame_count} frames, {fps} FPS")

    # Create mock VIBE output structure
    mock_output = {
        'joints3d': [],
        'vertices': [],
        'poses': [],
        'betas': [],
        'cameras': [],
        'frame_count': frame_count,
        'fps': fps
    }

    # Generate mock data for each frame
    for i in range(frame_count):
        # Mock 3D joints (49 joints as in VIBE output)
        import numpy as np
        joints = np.random.randn(49, 3) * 0.1  # Small random values
        mock_output['joints3d'].append(joints.tolist())

        # Mock vertices (6890 vertices as in SMPL)
        vertices = np.random.randn(6890, 3) * 0.1
        mock_output['vertices'].append(vertices.tolist())

        # Mock pose parameters (72 values)
        pose = np.random.randn(72) * 0.1
        mock_output['poses'].append(pose.tolist())

        # Mock shape parameters (10 values)
        beta = np.random.randn(10) * 0.1
        mock_output['betas'].append(beta.tolist())

        # Mock camera parameters (3 values)
        camera = np.random.randn(3) * 0.1
        mock_output['cameras'].append(camera.tolist())

    return mock_output


def convert_vibe_to_mediapipe_format(vibe_output):
    """
    Convert VIBE output to MediaPipe JSON format
    """
    if utils_available:
        script_log("Converting VIBE output to MediaPipe format")
    else:
        print("Converting VIBE output to MediaPipe format")

    # VIBE to MediaPipe joint mapping (simplified)
    # VIBE uses SMPL 24 joints + extra joints = 49 total
    vibe_to_mediapipe_mapping = {
        41: 'NOSE',  # VIBE nose joint index (approximate)
        12: 'LEFT_SHOULDER',  # SMPL joint 16
        13: 'RIGHT_SHOULDER',  # SMPL joint 17
        14: 'LEFT_ELBOW',  # SMPL joint 18
        15: 'RIGHT_ELBOW',  # SMPL joint 19
        16: 'LEFT_WRIST',  # SMPL joint 20
        17: 'RIGHT_WRIST',  # SMPL joint 21
        1: 'LEFT_HIP',  # SMPL joint 1
        2: 'RIGHT_HIP',  # SMPL joint 2
        4: 'LEFT_KNEE',  # SMPL joint 4
        5: 'RIGHT_KNEE',  # SMPL joint 5
        7: 'LEFT_ANKLE',  # SMPL joint 7
        8: 'RIGHT_ANKLE',  # SMPL joint 8
        # Add more mappings as needed
    }

    mediapipe_data = {}
    frame_count = vibe_output.get('frame_count', len(vibe_output['joints3d']))

    for frame_idx in range(frame_count):
        frame_data = {}

        for vibe_idx, mp_name in vibe_to_mediapipe_mapping.items():
            if frame_idx < len(vibe_output['joints3d']):
                joints = vibe_output['joints3d'][frame_idx]
                if vibe_idx < len(joints):
                    # Convert VIBE coordinates to MediaPipe-like format
                    # VIBE: +X Right, +Y Up, +Z Backward → MediaPipe: +X Right, -Y Up, +Z Forward
                    x, y, z = joints[vibe_idx]

                    frame_data[mp_name] = {
                        'x': round(float(x), 6),
                        'y': round(float(-y), 6),  # Flip Y for MediaPipe coordinate system
                        'z': round(float(-z), 6),  # Flip Z for forward direction
                        'visibility': 1.0
                    }

        mediapipe_data[str(frame_idx)] = frame_data

    return mediapipe_data


def main():
    """
    Main function to run VIBE motion capture extraction
    """
    if utils_available:
        script_log("Starting VIBE motion capture extraction")
    else:
        print("Starting VIBE motion capture extraction")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='VIBE Motion Capture Extraction')
    parser.add_argument('--show', type=str, help='Show name (optional)')
    parser.add_argument('--scene', type=str, help='Scene name (optional)')
    parser.add_argument('--video', type=str, help='Input video path (optional)')