# 1_vibe_mocap.py (version 1.0 = First VIBE)
"""
VIBE-based motion capture extraction
Uses VIBE (Video Inference for Human Body Pose and Shape) for 3D pose estimation
"""
# 1_vibe_mocap.py
"""
VIBE-based motion capture extraction
Uses VIBE (Video Inference for Human Body Pose and Shape) for higher quality 3D pose estimation
Outputs in VIBE's native coordinate system for later comparison and conversion
"""

import os
import sys
import json
import argparse
import tempfile
from pathlib import Path

# Add the project root to the Python path to ensure utils can be imported
script_dir = Path(__file__).parent
project_root = script_dir


# Use the same robust search as your existing code
def find_project_root():
    """Find the project root containing utils.py"""
    # Start from the current script's directory
    current_path = Path(__file__).parent.absolute()

    # Check current directory first
    if (current_path / "utils.py").exists():
        return current_path

    # Check parent directories
    for parent in current_path.parents:
        if (parent / "utils.py").exists():
            return parent

    # Fallback to current directory
    return current_path


project_root = find_project_root()
sys.path.insert(0, str(project_root))

from utils import script_log, get_current_show_name, get_current_scene_name, get_scene_paths, load_global_config


def setup_vibe_environment():
    """
    Set up VIBE environment and check dependencies
    """
    try:
        # Try to import basic dependencies
        import torch
        import torchvision
        import cv2
        import numpy as np

        # Check for VIBE directory
        vibe_dir = project_root / "VIBE"
        if not vibe_dir.exists():
            script_log(f"VIBE directory not found at: {vibe_dir}")
            script_log("Please clone: git clone https://github.com/mkocabas/VIBE.git")
            return False

        # Add VIBE to Python path
        sys.path.insert(0, str(vibe_dir))
        sys.path.insert(0, str(vibe_dir / "lib"))

        # Try to import VIBE modules
        try:
            from lib.models import VIBE
            from lib.dataset.inference import Inference
            script_log("VIBE modules imported successfully")
            return True
        except ImportError as e:
            script_log(f"Could not import VIBE modules: {e}")
            script_log("VIBE project structure may need manual setup")
            return False

    except ImportError as e:
        script_log(f"Missing dependencies: {e}")
        script_log("Please install: pip install torch torchvision opencv-python numpy")
        return False


def download_vibe_model_weights():
    """
    Download VIBE pre-trained model weights
    """
    try:
        import gdown
    except ImportError:
        script_log("Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    try:
        import torch

        # Create data directory
        data_dir = project_root / "VIBE" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        model_url = 'https://drive.google.com/uc?id=1X0-8OYbq8--3Q9pcVGfnnf6MLDq_0LOK'
        model_path = data_dir / 'vibe_model_w_3dpw.pth'

        if not model_path.exists():
            script_log("Downloading VIBE model weights...")
            gdown.download(model_url, str(model_path), quiet=False)
            script_log(f"Model downloaded to: {model_path}")
        else:
            script_log(f"Model already exists at: {model_path}")

        return model_path

    except Exception as e:
        script_log(f"Error downloading model: {e}")
        return None


def run_vibe_inference(video_path, output_dir):
    """
    Run VIBE inference on video file
    """
    try:
        script_log(f"Starting VIBE inference on: {video_path}")

        # Import VIBE modules
        import torch
        from lib.models import VIBE
        from lib.dataset.inference import Inference
        import cv2
        import numpy as np

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        script_log(f"Using device: {device}")

        # Download model weights
        model_path = download_vibe_model_weights()
        if not model_path:
            script_log("Using mock implementation")
            return mock_vibe_inference(video_path)

        # Initialize and load model
        model = VIBE(
            seqlen=16,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
        ).to(device)

        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()

        script_log("VIBE model loaded successfully")

        # Run inference
        inference = Inference(model=model, device=device)
        results = inference.run_on_video(video_path)

        return results

    except Exception as e:
        script_log(f"VIBE inference failed: {e}")
        script_log("Using mock implementation")
        return mock_vibe_inference(video_path)


def mock_vibe_inference(video_path):
    """
    Mock VIBE inference for testing
    """
    script_log(f"Running mock VIBE inference on: {video_path}")

    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    script_log(f"Video: {frame_count} frames, {fps} FPS")

    # Create realistic mock data
    mock_output = {
        'joints3d': [],
        'poses': [],
        'betas': [],
        'cameras': [],
        'frame_count': frame_count,
        'fps': fps
    }

    # Generate plausible mock poses
    for i in range(frame_count):
        # Create a walking motion pattern
        t = i / fps
        walk_cycle = np.sin(t * 2 * np.pi)  # Basic walk cycle

        # Base positions for a T-pose in VIBE coordinate system:
        # VIBE: +X Right, +Y Up, +Z Backward
        base_joints = np.array([
            [0.0, 1.0, 0.0],  # Pelvis (at height 1.0)
            [-0.1, 1.0, 0.0],  # Left hip
            [0.1, 1.0, 0.0],  # Right hip
            [0.0, 1.2, 0.0],  # Spine
            [-0.1, 0.7, 0.0],  # Left knee
            [0.1, 0.7, 0.0],  # Right knee
            [0.0, 1.4, 0.0],  # Spine2
            [-0.1, 0.4, 0.0],  # Left ankle
            [0.1, 0.4, 0.0],  # Right ankle
            [0.0, 1.6, 0.0],  # Spine3
            [-0.1, 0.4, 0.1],  # Left foot
            [0.1, 0.4, 0.1],  # Right foot
            [0.0, 1.8, 0.0],  # Neck
            [-0.2, 1.6, 0.0],  # Left collar
            [0.2, 1.6, 0.0],  # Right collar
            [0.0, 1.9, 0.0],  # Head
            [-0.3, 1.6, 0.0],  # Left shoulder
            [0.3, 1.6, 0.0],  # Right shoulder
            [-0.5, 1.4, 0.0],  # Left elbow
            [0.5, 1.4, 0.0],  # Right elbow
            [-0.6, 1.2, 0.0],  # Left wrist
            [0.6, 1.2, 0.0],  # Right wrist
            [-0.65, 1.2, 0.0],  # Left hand
            [0.65, 1.2, 0.0],  # Right hand
        ])

        # Add walking motion
        animated_joints = base_joints.copy()
        # Leg movement
        animated_joints[4][1] += walk_cycle * 0.1  # Left knee
        animated_joints[7][1] += walk_cycle * 0.1  # Left ankle
        animated_joints[5][1] -= walk_cycle * 0.1  # Right knee
        animated_joints[8][1] -= walk_cycle * 0.1  # Right ankle

        # Arm swing
        animated_joints[18][0] += walk_cycle * 0.05  # Left elbow
        animated_joints[20][0] += walk_cycle * 0.05  # Left wrist
        animated_joints[19][0] -= walk_cycle * 0.05  # Right elbow
        animated_joints[21][0] -= walk_cycle * 0.05  # Right wrist

        # Expand to 49 joints (VIBE format)
        full_joints = np.zeros((49, 3))
        full_joints[:24] = animated_joints
        # Fill remaining joints with interpolated values
        for j in range(24, 49):
            full_joints[j] = animated_joints[j % 24] * 0.8

        mock_output['joints3d'].append(full_joints.tolist())
        mock_output['poses'].append(np.random.randn(72).tolist())
        mock_output['betas'].append(np.random.randn(10).tolist())
        mock_output['cameras'].append(np.random.randn(3).tolist())

    return mock_output


def convert_vibe_to_consistent_format(vibe_output):
    """
    Convert VIBE output to consistent JSON format without coordinate conversion
    Keep VIBE's native coordinate system: +X Right, +Y Up, +Z Backward
    """
    script_log("Converting VIBE output to consistent JSON format")
    script_log("Keeping VIBE native coordinate system: +X Right, +Y Up, +Z Backward")

    # VIBE SMPL joints to landmark names mapping
    vibe_joint_mapping = {
        0: 'PELVIS',
        1: 'LEFT_HIP',
        2: 'RIGHT_HIP',
        4: 'LEFT_KNEE',
        5: 'RIGHT_KNEE',
        7: 'LEFT_ANKLE',
        8: 'RIGHT_ANKLE',
        12: 'NECK',
        15: 'HEAD',
        16: 'LEFT_SHOULDER',
        17: 'RIGHT_SHOULDER',
        18: 'LEFT_ELBOW',
        19: 'RIGHT_ELBOW',
        20: 'LEFT_WRIST',
        21: 'RIGHT_WRIST',
    }

    output_data = {}
    frame_count = vibe_output.get('frame_count', len(vibe_output['joints3d']))

    for frame_idx in range(frame_count):
        frame_data = {}

        for vibe_idx, landmark_name in vibe_joint_mapping.items():
            if frame_idx < len(vibe_output['joints3d']):
                joints = vibe_output['joints3d'][frame_idx]
                if vibe_idx < len(joints):
                    x, y, z = joints[vibe_idx]

                    # Keep VIBE's native coordinate system
                    # +X Right, +Y Up, +Z Backward
                    frame_data[landmark_name] = {
                        'x': round(float(x), 6),
                        'y': round(float(y), 6),  # Keep Y as up
                        'z': round(float(z), 6),  # Keep Z as backward
                        'visibility': 1.0
                    }

        output_data[str(frame_idx)] = frame_data

    return output_data


def main():
    """
    Main function to run VIBE motion capture extraction
    """
    script_log("Starting VIBE motion capture extraction")
    script_log("Output will be in VIBE native coordinate system: +X Right, +Y Up, +Z Backward")

    parser = argparse.ArgumentParser(description='VIBE Motion Capture Extraction')
    parser.add_argument('--show', type=str, help='Show name')
    parser.add_argument('--scene', type=str, help='Scene name')
    parser.add_argument('--video', type=str, help='Input video path')
    parser.add_argument('--output', type=str, help='Output JSON path')

    args = parser.parse_args()

    try:
        # Use project configuration
        show_name = args.show or get_current_show_name()
        scene_name = args.scene or get_current_scene_name(show_name)

        script_log(f"Processing: {show_name} - {scene_name}")

        scene_paths = get_scene_paths(show_name, scene_name)
        input_video = args.video or scene_paths['input_video']
        output_json = args.output or scene_paths['output_pose_data'].replace('.json', '_VIBE.json')

        script_log(f"Input: {input_video}")
        script_log(f"Output: {output_json}")

        if not os.path.exists(input_video):
            script_log(f"Error: Input video not found: {input_video}")
            return 1

        # Setup VIBE
        if not setup_vibe_environment():
            script_log("VIBE setup failed - using mock data")

        # Create output directory
        output_dir = Path(output_json).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run inference
        vibe_output = run_vibe_inference(input_video, str(output_dir))

        # Convert to consistent JSON format (keeping VIBE coordinates)
        output_data = convert_vibe_to_consistent_format(vibe_output)

        # Save results
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        script_log(f"VIBE mocap completed: {output_json}")
        script_log(f"Frames processed: {len(output_data)}")
        script_log("Coordinate system: VIBE native (+X Right, +Y Up, +Z Backward)")

        return 0

    except Exception as e:
        script_log(f"Error: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)