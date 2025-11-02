# 3_ApplyPhysics.py (Version 3.4 - Added shoulder stabilization)

import argparse
import os
import json
import math
import sys
import numpy as np
from datetime import datetime
from enum import Enum, auto
from collections import deque
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import random

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
    from utils import (
        script_log,
        comment,
        get_project_root,
        get_scene_paths,
        get_current_show_name,
        get_current_scene_name,
        set_current_scene_name,
        load_tsv_list,
        get_show_path,
        get_blender_path,
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

# Landmark configuration file - same as Steps 1 and 2
LANDMARK_CONFIG_FILE = "0_RunMocapAnimPipeline_LANDMARKS.json"

# Load configuration at module level
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'apply_physics_config.json')
with open(CONFIG_FILE_PATH) as config_file:
    CONFIG = json.load(config_file)

# Load debug flags from config instead of hard-coding
debug_flags = CONFIG.get('debug_flags', {})
TURN_OFF_Z_UP_TRANSFORMATION = debug_flags.get('turn_off_z_up_transformation', False)
TURN_OFF_FORWARD_TRANSFORMATION = debug_flags.get('turn_off_forward_transformation', True)
TURN_OFF_SHOULDER_SHIFT = debug_flags.get('turn_off_shoulder_shift', True)
TURN_OFF_HIP_SHIFT = debug_flags.get('turn_off_hip_shift', True)
TURN_OFF_BIOMECHANICAL_CONSTRAINTS = debug_flags.get('turn_off_biomechanical_constraints', True)
TURN_OFF_HIP_HEEL_FLOOR_SHIFT = debug_flags.get('turn_off_hip_heel_floor_shift', True)
TURN_OFF_DEPTH_ADJUSTMENT = debug_flags.get('turn_off_depth_adjustment', True)
TURN_OFF_SHOULDER_STABILIZATION = debug_flags.get('turn_off_shoulder_stabilization', True)
TURN_OFF_HEAD_STABILIZATION = debug_flags.get('turn_off_head_stabilization', True)

class ShoulderStabilizer:
    """Detects and corrects upper/lower body scaling mismatches"""

    def __init__(self, landmark_config=None):
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})
        self.shoulder_hip_history = deque(maxlen=10)  # Track shoulder-hip relationships

    def analyze_shoulder_hip_relationship(self, frame):
        """Analyze the relationship between shoulders and hips to detect scaling mismatches"""
        left_shoulder = frame.get('LEFT_SHOULDER')
        right_shoulder = frame.get('RIGHT_SHOULDER')
        left_hip = frame.get('LEFT_HIP')
        right_hip = frame.get('RIGHT_HIP')

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        # Calculate shoulder and hip midpoints
        shoulder_mid = {
            'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
            'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
            'z': (left_shoulder['z'] + right_shoulder['z']) / 2
        }

        hip_mid = {
            'x': (left_hip['x'] + right_hip['x']) / 2,
            'y': (left_hip['y'] + right_hip['y']) / 2,
            'z': (left_hip['z'] + right_hip['z']) / 2
        }

        # Calculate vertical distance (torso height)
        torso_height = abs(shoulder_mid['z'] - hip_mid['z'])

        # Calculate shoulder width and hip width
        shoulder_width = calculate_3d_distance(left_shoulder, right_shoulder)
        hip_width = calculate_3d_distance(left_hip, right_hip)

        # Normal expected ratios (based on human proportions)
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 1e-6 else 1.0
        torso_aspect_ratio = shoulder_width / torso_height if torso_height > 1e-6 else 1.0

        return {
            'shoulder_mid': shoulder_mid,
            'hip_mid': hip_mid,
            'torso_height': torso_height,
            'shoulder_width': shoulder_width,
            'hip_width': hip_width,
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'torso_aspect_ratio': torso_aspect_ratio,
            'vertical_offset': shoulder_mid['z'] - hip_mid['z']
        }

    def detect_scaling_mismatch(self, frame_analysis):
        """Detect if shoulders have different scaling than hips"""
        if not frame_analysis:
            return None

        # Get expected proportions from config or use defaults
        stabilization_config = CONFIG.get('shoulder_stabilization', {})
        EXPECTED_SHOULDER_HIP_RATIO = stabilization_config.get('expected_shoulder_hip_ratio', 1.2)
        EXPECTED_TORSO_ASPECT = stabilization_config.get('expected_torso_aspect_ratio', 1.4)
        TOLERANCE = stabilization_config.get('proportion_tolerance', 0.3)

        shoulder_hip_ratio = frame_analysis['shoulder_hip_ratio']
        torso_aspect_ratio = frame_analysis['torso_aspect_ratio']

        issues = []

        # Check shoulder-hip width ratio
        if abs(shoulder_hip_ratio - EXPECTED_SHOULDER_HIP_RATIO) > TOLERANCE:
            issues.append(f"shoulder_hip_ratio_{shoulder_hip_ratio:.2f}")

        # Check torso proportions
        if abs(torso_aspect_ratio - EXPECTED_TORSO_ASPECT) > TOLERANCE:
            issues.append(f"torso_aspect_{torso_aspect_ratio:.2f}")

        return issues if issues else None

    def stabilize_shoulders(self, frame, frame_analysis):
        """Apply corrections to stabilize shoulders relative to hips"""
        if not frame_analysis:
            return frame

        left_shoulder = frame.get('LEFT_SHOULDER')
        right_shoulder = frame.get('RIGHT_SHOULDER')
        left_hip = frame.get('LEFT_HIP')
        right_hip = frame.get('RIGHT_HIP')

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return frame

        # Get current analysis
        shoulder_mid = frame_analysis['shoulder_mid']
        hip_mid = frame_analysis['hip_mid']

        # Calculate desired shoulder position based on hips
        stabilization_config = CONFIG.get('shoulder_stabilization', {})
        TARGET_VERTICAL_OFFSET = stabilization_config.get('target_vertical_offset', 0.4)
        CORRECTION_FACTOR = stabilization_config.get('correction_factor', 0.3)

        current_offset = frame_analysis['vertical_offset']
        offset_correction = TARGET_VERTICAL_OFFSET - current_offset

        # Apply smooth correction (don't jump abruptly)
        applied_correction = offset_correction * CORRECTION_FACTOR

        # Correct shoulder positions
        for shoulder in ['LEFT_SHOULDER', 'RIGHT_SHOULDER']:
            if shoulder in frame:
                frame[shoulder]['z'] += applied_correction

        # Also correct connected joints (neck, elbows) to maintain arm proportions
        self._correct_connected_joints(frame, applied_correction)

        return frame

    def _correct_connected_joints(self, frame, z_correction):
        """Correct joints connected to shoulders to maintain proportions"""
        # Correct neck/head
        for joint in ['NOSE', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EAR', 'RIGHT_EAR']:
            if joint in frame:
                frame[joint]['z'] += z_correction * 0.8  # Slightly less correction

        # Correct elbows (partial correction to maintain arm shape)
        for joint in ['LEFT_ELBOW', 'RIGHT_ELBOW']:
            if joint in frame:
                frame[joint]['z'] += z_correction * 0.5


class HeadStabilizer:
    """Detects and corrects head position relative to body in Blender coordinate system"""

    def __init__(self, landmark_config=None):
        script_log("Initializing head stabilization for Blender coordinates")
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})

        # Load config
        self.config = CONFIG.get('head_stabilization', {})
        self.distance_config = self.config.get('distance_controls', {})
        self.height_config = self.config.get('height_controls', {})

    def calculate_body_reference_frame(self, frame):
        """Calculate body-relative 'up' and 'forward' directions for current pose"""
        left_shoulder = frame.get('LEFT_SHOULDER')
        right_shoulder = frame.get('RIGHT_SHOULDER')
        left_hip = frame.get('LEFT_HIP')
        right_hip = frame.get('RIGHT_HIP')

        if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
            return None

        # Convert to numpy arrays
        ls = np.array([left_shoulder['x'], left_shoulder['y'], left_shoulder['z']])
        rs = np.array([right_shoulder['x'], right_shoulder['y'], right_shoulder['z']])
        lh = np.array([left_hip['x'], left_hip['y'], left_hip['z']])
        rh = np.array([right_hip['x'], right_hip['y'], right_hip['z']])

        # Calculate body directions
        shoulder_mid = (ls + rs) / 2
        hip_mid = (lh + rh) / 2

        # Body "up" direction (from hips to shoulders)
        body_up = shoulder_mid - hip_mid
        body_up_norm = np.linalg.norm(body_up)
        if body_up_norm > 1e-6:
            body_up = body_up / body_up_norm
        else:
            body_up = np.array([0, 0, 1])  # Fallback to global up

        # Body "forward" direction (perpendicular to shoulder line and body up)
        shoulder_line = rs - ls
        body_forward = np.cross(shoulder_line, body_up)
        body_forward_norm = np.linalg.norm(body_forward)
        if body_forward_norm > 1e-6:
            body_forward = body_forward / body_forward_norm
        else:
            body_forward = np.array([0, 1, 0])  # Fallback to global forward

        # Ensure forward points in +Y direction (toward camera in Blender)
        if body_forward[1] < 0:
            body_forward = -body_forward

        return {
            'shoulder_mid': shoulder_mid,
            'hip_mid': hip_mid,
            'body_up': body_up,
            'body_forward': body_forward,
            'shoulder_line': shoulder_line
        }

    def analyze_head_position(self, frame, body_frame):
        """Analyze head position relative to body in body-relative coordinates"""
        head_top = frame.get('HEAD_TOP')
        nose = frame.get('NOSE')

        if not all([head_top, nose, body_frame]):
            return None

        # Convert to numpy arrays
        ht = np.array([head_top['x'], head_top['y'], head_top['z']])
        ns = np.array([nose['x'], nose['y'], nose['z']])

        shoulder_mid = body_frame['shoulder_mid']
        body_up = body_frame['body_up']
        body_forward = body_frame['body_forward']

        # Calculate head position relative to shoulders
        head_to_shoulders = ht - shoulder_mid
        nose_to_shoulders = ns - shoulder_mid

        # Project onto body-relative directions
        head_vertical = np.dot(head_to_shoulders, body_up)  # How far "above" shoulders
        head_forward = np.dot(head_to_shoulders, body_forward)  # How far "in front of" shoulders

        nose_vertical = np.dot(nose_to_shoulders, body_up)
        nose_forward = np.dot(nose_to_shoulders, body_forward)

        # Calculate head height (distance between head_top and nose)
        head_height = np.linalg.norm(ht - ns)

        return {
            'head_vertical': head_vertical,  # Positive = above shoulders
            'head_forward': head_forward,  # Positive = in front of shoulders
            'nose_vertical': nose_vertical,
            'nose_forward': nose_forward,
            'head_height': head_height,
            'head_top_global': ht,
            'nose_global': ns,
            'shoulder_mid_global': shoulder_mid
        }

    def detect_head_issues(self, head_analysis):
        """Detect if head needs correction"""
        if not head_analysis:
            return None

        MIN_HEAD_HEIGHT = self.height_config.get('min_head_height', 0.15)
        MIN_HEAD_VERTICAL = self.height_config.get('min_head_vertical', 0.10)  # Min height above shoulders
        MIN_HEAD_FORWARD = self.distance_config.get('min_head_forward', 0.08)  # Min forward from shoulders
        NOSE_FORWARD_ADVANTAGE = self.distance_config.get('nose_forward_advantage',
                                                          0.02)  # Nose should be further forward

        issues = []

        # Check head height (distance from head_top to nose)
        if head_analysis['head_height'] < MIN_HEAD_HEIGHT:
            issues.append(f"head_squished_{head_analysis['head_height']:.3f}")

        # Check if head is too low relative to shoulders
        if head_analysis['head_vertical'] < MIN_HEAD_VERTICAL:
            issues.append(f"head_too_low_{head_analysis['head_vertical']:.3f}")

        # Check if head is too far back relative to shoulders
        if head_analysis['head_forward'] < MIN_HEAD_FORWARD:
            issues.append(f"head_too_far_back_{head_analysis['head_forward']:.3f}")

        # Check if nose is sufficiently forward of head top
        if head_analysis['nose_forward'] < head_analysis['head_forward'] + NOSE_FORWARD_ADVANTAGE:
            issues.append(f"nose_not_forward_enough_{head_analysis['nose_forward']:.3f}")

        return issues if issues else None

    def correct_head_position(self, frame, head_analysis, body_frame):
        """Apply corrections to position head properly relative to body"""
        if not head_analysis:
            return frame

        head_top = frame.get('HEAD_TOP')
        nose = frame.get('NOSE')

        if not all([head_top, nose]):
            return frame

        # Get config values
        TARGET_HEAD_VERTICAL = self.height_config.get('target_head_vertical', 0.20)
        TARGET_HEAD_FORWARD = self.distance_config.get('target_head_forward', 0.12)
        TARGET_HEAD_HEIGHT = self.height_config.get('target_head_height', 0.18)
        VERTICAL_CORRECTION_STRENGTH = self.height_config.get('vertical_correction_strength', 0.4)
        FORWARD_CORRECTION_STRENGTH = self.distance_config.get('forward_correction_strength', 0.4)
        HEIGHT_CORRECTION_STRENGTH = self.height_config.get('height_correction_strength', 0.3)

        body_up = body_frame['body_up']
        body_forward = body_frame['body_forward']

        # 1. Vertical correction - move head above shoulders
        vertical_correction = max(0,
                                  TARGET_HEAD_VERTICAL - head_analysis['head_vertical']) * VERTICAL_CORRECTION_STRENGTH

        # 2. Forward correction - move head in front of shoulders
        forward_correction = max(0, TARGET_HEAD_FORWARD - head_analysis['head_forward']) * FORWARD_CORRECTION_STRENGTH

        # 3. Height correction - ensure proper head proportions
        height_correction = max(0, TARGET_HEAD_HEIGHT - head_analysis['head_height']) * HEIGHT_CORRECTION_STRENGTH

        # Apply corrections along body-relative directions
        head_top['x'] += body_up[0] * vertical_correction + body_forward[0] * forward_correction
        head_top['y'] += body_up[1] * vertical_correction + body_forward[1] * forward_correction
        head_top['z'] += body_up[2] * vertical_correction + body_forward[2] * forward_correction

        # Move nose with head, plus extra forward and height
        nose['x'] += body_up[0] * vertical_correction + body_forward[0] * (forward_correction + 0.03)
        nose['y'] += body_up[1] * vertical_correction + body_forward[1] * (forward_correction + 0.03)
        nose['z'] += body_up[2] * vertical_correction + body_forward[2] * (forward_correction + 0.03)

        # Apply head height correction (move head_top up relative to nose)
        head_top['z'] += height_correction

        return frame

    def process_head_frame(self, frame):
        """Main method to detect and correct head position for a single frame"""
        # Calculate body reference frame for current pose
        body_frame = self.calculate_body_reference_frame(frame)

        if not body_frame:
            return frame

        # Analyze head position relative to body
        head_analysis = self.analyze_head_position(frame, body_frame)

        if not head_analysis:
            return frame

        # Detect issues
        issues = self.detect_head_issues(head_analysis)

        if issues:
            if random.random() < 0.03:
                script_log(f"Head stabilization issues: {issues}")
            # Apply corrections
            frame = self.correct_head_position(frame, head_analysis, body_frame)

        return frame

class MotionState(Enum):
    STANDING = auto()
    WALKING = auto()
    RUNNING = auto()
    JUMPING = auto()
    LANDING = auto()
    FALLING = auto()
    FEET_OFF_FLOOR = auto()  # For handstands, rolls, etc.
    TRANSITION = auto()


class MotionPhaseDetector:
    def __init__(self, window_size=5, landmark_config=None):
        self.window_size = CONFIG['motion_detection']['state_transition_window_size']
        self.state_history = deque(maxlen=window_size)
        self.current_state = MotionState.STANDING
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})

    def lowest_z_of_body_parts(self, frame):
        """Find the lowest body part using landmarks from config"""
        min_z = float('inf')

        # Check all landmarks from config
        for landmark_name in self.landmarks.keys():
            landmark_data = frame.get(landmark_name, {})
            if isinstance(landmark_data, dict) and 'z' in landmark_data:
                z_val = landmark_data['z']
                if z_val is not None and z_val < min_z:
                    min_z = z_val

        return min_z if min_z != float('inf') else 0.0

    def detect_state(self, frame, prev_frame=None):
        """Determine the current motion state based on body positions"""
        # Get relevant landmarks from config
        foot_landmarks = []
        hand_landmarks = []
        head_landmarks = []

        for landmark_name, landmark_data in self.landmarks.items():
            # Check if this landmark is used for motion detection
            if landmark_data.get("motion_detection", False):
                if "FOOT" in landmark_name:
                    foot_landmarks.append(landmark_name)
                elif "WRIST" in landmark_name or "HAND" in landmark_name:
                    hand_landmarks.append(landmark_name)
                elif "HEAD" in landmark_name or "NOSE" in landmark_name:
                    head_landmarks.append(landmark_name)

        # Get foot positions
        foot_z_values = []
        for foot_landmark in foot_landmarks:
            foot_z = frame.get(foot_landmark, {}).get('z', float('inf'))
            if foot_z != float('inf'):
                foot_z_values.append(foot_z)

        foot_min_z = min(foot_z_values) if foot_z_values else float('inf')

        # Get hand positions for FEET_OFF_FLOOR detection
        hand_z_values = []
        for hand_landmark in hand_landmarks:
            hand_z = frame.get(hand_landmark, {}).get('z', float('inf'))
            if hand_z != float('inf'):
                hand_z_values.append(hand_z)

        hand_min_z = min(hand_z_values) if hand_z_values else float('inf')

        # Get head positions
        head_z_values = []
        for head_landmark in head_landmarks:
            head_z = frame.get(head_landmark, {}).get('z', float('inf'))
            if head_z != float('inf'):
                head_z_values.append(head_z)

        head_min_z = min(head_z_values) if head_z_values else float('inf')

        # Calculate velocities if previous frame exists
        foot_vel = 0
        if prev_frame and foot_z_values:
            prev_foot_z = []
            for foot_landmark in foot_landmarks:
                prev_z = prev_frame.get(foot_landmark, {}).get('z', 0)
                if prev_z != 0:
                    prev_foot_z.append(prev_z)

            if prev_foot_z and foot_z_values:
                prev_avg = sum(prev_foot_z) / len(prev_foot_z)
                current_avg = sum(foot_z_values) / len(foot_z_values)
                foot_vel = current_avg - prev_avg

        # State detection logic
        new_state = self.current_state

        # Check for FEET_OFF_FLOOR state (handstand, headstand, etc.)
        min_body_z = self.lowest_z_of_body_parts(frame)
        if (min_body_z < foot_min_z and
                (hand_min_z == min_body_z or head_min_z == min_body_z)):
            new_state = MotionState.FEET_OFF_FLOOR

        # Check for jumping/landing
        elif foot_min_z > CONFIG['motion_detection']['feet_off_ground_threshold']:
            if foot_vel > CONFIG['motion_detection']['upward_velocity_threshold']:
                new_state = MotionState.JUMPING
            elif foot_vel < CONFIG['motion_detection']['downward_velocity_threshold']:
                new_state = MotionState.LANDING
            else:
                new_state = MotionState.FALLING  # Mid-air with neutral velocity

        # Check for walking/running
        elif 0 < foot_min_z <= CONFIG['motion_detection']['feet_off_ground_threshold']:
            # Near ground but not touching
            if abs(foot_vel) > CONFIG['motion_detection']['running_velocity_threshold']:
                # Fast movement
                new_state = MotionState.RUNNING
            elif abs(foot_vel) > CONFIG['motion_detection']['walking_velocity_threshold']:
                # Slow movement
                new_state = MotionState.WALKING
            else:
                new_state = MotionState.STANDING

        # State transition smoothing
        self.state_history.append(new_state)
        if len(self.state_history) == self.window_size:
            # Require consistent state for window_size frames before changing
            if all(s == new_state for s in self.state_history):
                self.current_state = new_state
            else:
                self.current_state = MotionState.TRANSITION

        return self.current_state


class HeightAdjuster:
    def __init__(self, landmark_config=None):
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})
        self.state_detector = MotionPhaseDetector(landmark_config=landmark_config)
        self.previous_frames = deque(maxlen=CONFIG['motion_detection']['jump_detection_min_frames'])
        self.ground_history = deque(maxlen=CONFIG['height_adjustment']['ground_history_size'])
        self.jump_trajectory = None

    def process_frames(self, all_frames):
        # First pass: Detect motion phases
        states = {}
        prev_frame = None
        for frame_num, frame in sorted(all_frames.items(), key=lambda x: int(x[0])):
            state = self.state_detector.detect_state(frame, prev_frame)
            states[frame_num] = state
            prev_frame = frame
            self.previous_frames.append(frame)

        # Second pass: Apply phase-specific corrections
        for frame_num, frame in all_frames.items():
            state = states[frame_num]

            if state == MotionState.FEET_OFF_FLOOR:
                self.adjust_feet_off_floor(frame)
            elif state in [MotionState.WALKING, MotionState.RUNNING, MotionState.STANDING]:
                self.adjust_foot_contact(frame)
            elif state == MotionState.JUMPING:
                self.adjust_jumping(frame, frame_num, all_frames)
            elif state == MotionState.LANDING:
                self.adjust_landing(frame)
            elif state == MotionState.FALLING:
                self.adjust_falling(frame)

        # Third pass: Physics-based smoothing
        self.apply_physics_smoothing(all_frames)

        return all_frames

    def adjust_feet_off_floor(self, frame):
        """Adjust height when other joints are lowest point"""
        # Find lowest body part
        min_z = self.state_detector.lowest_z_of_body_parts(frame)

        # Shift entire body so lowest point is at Z=0
        for landmark in frame.values():
            if isinstance(landmark, dict) and 'z' in landmark:
                landmark['z'] -= min_z

    def adjust_foot_contact(self, frame):
        """Ensure lowest foot is at ground level"""
        # Get foot landmarks from config
        foot_landmarks = []
        for landmark_name, landmark_data in self.landmarks.items():
            if "FOOT" in landmark_name or "ANKLE" in landmark_name or "HEEL" in landmark_name:
                foot_landmarks.append(landmark_name)

        foot_z_values = []
        for foot_landmark in foot_landmarks:
            foot_z = frame.get(foot_landmark, {}).get('z', float('inf'))
            if foot_z != float('inf'):
                foot_z_values.append(foot_z)

        if foot_z_values:
            min_foot_z = min(foot_z_values)
            shift = min_foot_z
            for landmark in frame.values():
                if isinstance(landmark, dict) and 'z' in landmark:
                    landmark['z'] -= shift
            self.ground_history.append(shift)

    def adjust_jumping(self, frame, frame_num, all_frames):
        """Smooth jump trajectory using polynomial fitting"""
        # Get foot landmarks from config
        foot_landmarks = []
        for landmark_name, landmark_data in self.landmarks.items():
            if "FOOT" in landmark_name or "ANKLE" in landmark_name:
                foot_landmarks.append(landmark_name)

        # Collect nearby frames for trajectory analysis
        frame_int = int(frame_num)
        window = CONFIG['motion_detection']['state_transition_window_size']
        start = max(0, frame_int - window)
        end = min(len(all_frames) - 1, frame_int + window)

        # Extract foot heights in this window
        foot_heights = []
        frame_numbers = []
        for i in range(start, end + 1):
            f = all_frames.get(str(i))
            if f:
                foot_z_values = []
                for foot_landmark in foot_landmarks:
                    foot_z = f.get(foot_landmark, {}).get('z', 0)
                    foot_z_values.append(foot_z)

                if foot_z_values:
                    avg_foot_z = sum(foot_z_values) / len(foot_z_values)
                    foot_heights.append(avg_foot_z)
                    frame_numbers.append(i)

        # Need at least 3 points to fit a quadratic curve
        if len(foot_heights) < 3:
            script_log(f"Debug: Insufficient samples ({len(foot_heights)}) for jump trajectory fitting")
            return

        X = np.array(frame_numbers).reshape(-1, 1)
        y = np.array(foot_heights)

        try:
            # Use simple linear regression if we have exactly 3 points
            if len(foot_heights) == 3:
                model = make_pipeline(
                    PolynomialFeatures(degree=1),  # Linear instead of quadratic
                    RANSACRegressor(min_samples=2)  # Need at least 2 points for linear
                )
            else:
                model = make_pipeline(
                    PolynomialFeatures(degree=2),
                    RANSACRegressor(min_samples=min(3, len(foot_heights) - 1))
                )

            model.fit(X, y)
            target_z = model.predict([[frame_int]])[0]

            # Apply adjustment
            current_foot_z_values = []
            for foot_landmark in foot_landmarks:
                foot_z = frame.get(foot_landmark, {}).get('z', 0)
                current_foot_z_values.append(foot_z)

            if current_foot_z_values:
                current_z = sum(current_foot_z_values) / len(current_foot_z_values)
                shift = current_z - target_z

                for landmark in frame.values():
                    if isinstance(landmark, dict) and 'z' in landmark:
                        landmark['z'] -= shift

        except Exception as e:
            script_log(f"Debug: Jump trajectory fitting failed: {str(e)}")

    def adjust_landing(self, frame):
        """Gradually approach ground level during landing"""
        if len(self.ground_history) > 0:
            avg_ground = sum(self.ground_history) / len(self.ground_history)

            # Get current foot positions
            foot_landmarks = []
            for landmark_name, landmark_data in self.landmarks.items():
                if "FOOT" in landmark_name or "ANKLE" in landmark_name:
                    foot_landmarks.append(landmark_name)

            foot_z_values = []
            for foot_landmark in foot_landmarks:
                foot_z = frame.get(foot_landmark, {}).get('z', float('inf'))
                if foot_z != float('inf'):
                    foot_z_values.append(foot_z)

            if foot_z_values:
                current_z = min(foot_z_values)

                # Blend between current position and ground level
                blend_factor = CONFIG['height_adjustment']['landing_blend_factor']  # Higher = faster approach to ground
                target_z = current_z * (1 - blend_factor) + avg_ground * blend_factor
                shift = current_z - target_z

                for landmark in frame.values():
                    if isinstance(landmark, dict) and 'z' in landmark:
                        landmark['z'] -= shift

    def adjust_falling(self, frame):
        """Emergency correction for unexpected falls"""
        # Find the lowest point in the frame
        min_z = min(
            landmark.get('z', float('inf'))
            for landmark in frame.values()
            if isinstance(landmark, dict)
        )

        if min_z < 0:  # Only correct if below ground
            for landmark in frame.values():
                if isinstance(landmark, dict) and 'z' in landmark:
                    landmark['z'] -= min_z

    def apply_physics_smoothing(self, all_frames):
        """Apply spring-damper smoothing to all height adjustments"""
        # Implementation of physics-based smoothing would go here
        # This would maintain velocity continuity between frames
        pass


class DepthAdjuster:
    def __init__(self, landmark_config=None):
        self.epsilon = CONFIG['general']['epsilon']
        self.depth_config = CONFIG.get('depth_adjustment', {})
        self.sensitivity = self.depth_config.get('sensitivity', {})
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})

    def calculate_biomechanical_metrics(self, all_frames):
        """Calculate limb lengths and detect depth exaggeration using landmark config"""
        limb_lengths = {
            'upper_arms': [], 'forearms': [], 'thighs': [], 'shins': [],
            'torso_heights': [], 'shoulder_widths': []
        }

        for frame_data in all_frames.values():
            # Calculate limb lengths based on landmark relationships from config
            # Upper arms (shoulder to elbow)
            left_upper = self._calculate_limb_length(frame_data, 'LEFT_SHOULDER', 'LEFT_ELBOW')
            right_upper = self._calculate_limb_length(frame_data, 'RIGHT_SHOULDER', 'RIGHT_ELBOW')
            if left_upper: limb_lengths['upper_arms'].append(left_upper)
            if right_upper: limb_lengths['upper_arms'].append(right_upper)

            # Forearms (elbow to wrist)
            left_forearm = self._calculate_limb_length(frame_data, 'LEFT_ELBOW', 'LEFT_WRIST')
            right_forearm = self._calculate_limb_length(frame_data, 'RIGHT_ELBOW', 'RIGHT_WRIST')
            if left_forearm: limb_lengths['forearms'].append(left_forearm)
            if right_forearm: limb_lengths['forearms'].append(right_forearm)

            # Thighs (hip to knee)
            left_thigh = self._calculate_limb_length(frame_data, 'LEFT_HIP', 'LEFT_KNEE')
            right_thigh = self._calculate_limb_length(frame_data, 'RIGHT_HIP', 'RIGHT_KNEE')
            if left_thigh: limb_lengths['thighs'].append(left_thigh)
            if right_thigh: limb_lengths['thighs'].append(right_thigh)

            # Shins (knee to ankle)
            left_shin = self._calculate_limb_length(frame_data, 'LEFT_KNEE', 'LEFT_ANKLE')
            right_shin = self._calculate_limb_length(frame_data, 'RIGHT_KNEE', 'RIGHT_ANKLE')
            if left_shin: limb_lengths['shins'].append(left_shin)
            if right_shin: limb_lengths['shins'].append(right_shin)

            # Torso height (shoulder to hip)
            torso = self._calculate_limb_length(frame_data, 'LEFT_SHOULDER', 'LEFT_HIP')
            if torso: limb_lengths['torso_heights'].append(torso)

            # Shoulder width
            shoulder_width = self._calculate_limb_length(frame_data, 'LEFT_SHOULDER', 'RIGHT_SHOULDER')
            if shoulder_width: limb_lengths['shoulder_widths'].append(shoulder_width)

        return limb_lengths

    def _calculate_limb_length(self, frame_data, joint1, joint2):
        """Calculate distance between two joints"""
        p1 = frame_data.get(joint1)
        p2 = frame_data.get(joint2)

        if not (p1 and p2 and
                all(k in p1 for k in ['x', 'y', 'z']) and p1['x'] is not None and
                all(k in p2 for k in ['x', 'y', 'z']) and p2['x'] is not None):
            return None

        return math.sqrt((p2['x'] - p1['x']) ** 2 +
                         (p2['y'] - p1['y']) ** 2 +
                         (p2['z'] - p1['z']) ** 2)

    def detect_depth_exaggeration(self, limb_lengths, all_frames):
        """Detect depth exaggeration using multiple strategies"""
        # Check for manual override first
        manual_scale = self.depth_config.get('manual_depth_scale_factor')
        if manual_scale is not None:
            script_log(f"Using manual depth scale factor: {manual_scale}")
            return manual_scale

        # Check for forced minimal depth
        if self.depth_config.get('force_minimal_depth_variation', False):
            script_log("Forcing minimal depth variation based on config")
            return self._calculate_aggressive_depth_reduction(all_frames)

        # Strategy 1: Biomechanical ratios (existing approach)
        biomechanical_scale = self._detect_via_biomechanical_ratios(limb_lengths)

        # Strategy 2: Depth variation analysis (new - for cases with minimal true depth)
        depth_variation_scale = self._detect_via_depth_variation(all_frames)

        # Strategy 3: Planar motion detection (new - for XZ-only motion)
        planar_scale = self._detect_via_planar_motion(all_frames)

        # Combine strategies - take the most aggressive (smallest) scale factor
        scale_factors = [biomechanical_scale, depth_variation_scale, planar_scale]
        valid_scales = [s for s in scale_factors if s < 0.95]  # Only consider significant corrections

        if valid_scales:
            final_scale = min(valid_scales)
            script_log(f"Depth analysis: Biomechanical={biomechanical_scale:.3f}, "
                       f"DepthVariation={depth_variation_scale:.3f}, "
                       f"Planar={planar_scale:.3f}, Final={final_scale:.3f}")

            # Apply min/max scale limits from config
            min_scale = self.sensitivity.get('min_scale_factor', 0.1)
            max_scale = self.sensitivity.get('max_scale_factor', 1.0)
            return max(min_scale, min(max_scale, final_scale))

        return 1.0  # No adjustment needed

    def _detect_via_biomechanical_ratios(self, limb_lengths):
        """Original biomechanical approach with configurable threshold"""
        expected_ratios = {
            'forearm_to_upper_arm': 0.8,
            'shin_to_thigh': 0.8,
            'upper_arm_to_torso': 0.4,
            'shoulder_to_torso': 0.6
        }

        if (limb_lengths['upper_arms'] and limb_lengths['forearms'] and
                limb_lengths['thighs'] and limb_lengths['shins'] and
                limb_lengths['torso_heights'] and limb_lengths['shoulder_widths']):

            avg_upper_arm = np.median(limb_lengths['upper_arms'])
            avg_forearm = np.median(limb_lengths['forearms'])
            avg_thigh = np.median(limb_lengths['thighs'])
            avg_shin = np.median(limb_lengths['shins'])
            avg_torso = np.median(limb_lengths['torso_heights'])
            avg_shoulder = np.median(limb_lengths['shoulder_widths'])

            actual_ratios = {
                'forearm_to_upper_arm': avg_forearm / avg_upper_arm if avg_upper_arm > self.epsilon else 0,
                'shin_to_thigh': avg_shin / avg_thigh if avg_thigh > self.epsilon else 0,
                'upper_arm_to_torso': avg_upper_arm / avg_torso if avg_torso > self.epsilon else 0,
                'shoulder_to_torso': avg_shoulder / avg_torso if avg_torso > self.epsilon else 0
            }

            exaggeration_factors = []
            # Use configurable discrepancy threshold (default: 1.1 = 10% discrepancy)
            discrepancy_threshold = self.sensitivity.get('biomechanical_discrepancy_threshold', 1.1)

            for ratio_name, expected in expected_ratios.items():
                actual = actual_ratios[ratio_name]
                if actual > self.epsilon:
                    ratio_discrepancy = actual / expected
                    if ratio_discrepancy > discrepancy_threshold:
                        exaggeration_factors.append(math.sqrt(ratio_discrepancy))

            if exaggeration_factors:
                return 1.0 / np.mean(exaggeration_factors)

        return 1.0

    def _detect_via_depth_variation(self, all_frames):
        """Detect if depth variation is unrealistically large compared to height/width"""
        # Calculate overall bounding box dimensions
        all_y_values = []
        all_x_values = []
        all_z_values = []

        for frame_data in all_frames.values():
            for coords in frame_data.values():
                if all(k in coords for k in ['x', 'y', 'z']) and coords['x'] is not None:
                    all_x_values.append(coords['x'])
                    all_y_values.append(coords['y'])
                    all_z_values.append(coords['z'])

        if not all_y_values:
            return 1.0

        depth_range = max(all_y_values) - min(all_y_values)
        height_range = max(all_z_values) - min(all_z_values)
        width_range = max(all_x_values) - min(all_x_values)

        # Use configurable trigger ratio (default: 1.2 = 20% larger than width)
        trigger_ratio = self.sensitivity.get('depth_variation_trigger_ratio', 1.2)
        max_allowed_ratio = self.depth_config.get('max_allowed_depth_ratio', 0.5)

        # If depth is significantly larger than width, we have exaggeration
        if depth_range > width_range * trigger_ratio:
            depth_ratio = width_range / depth_range
            script_log(f"Depth variation detection: depth_range={depth_range:.3f}, "
                       f"width_range={width_range:.3f}, ratio={depth_ratio:.3f}, "
                       f"trigger={trigger_ratio:.1f}")

            min_scale = self.sensitivity.get('min_scale_factor', 0.1)
            return max(min_scale, depth_ratio)  # More aggressive scaling

        return 1.0

    def _detect_via_planar_motion(self, all_frames):
        """New: Detect if motion should be primarily in XZ plane (minimal true depth variation)"""
        # Analyze Y-coordinate variance across all frames for each landmark
        landmark_y_variance = {}

        for landmark_name in all_frames[list(all_frames.keys())[0]].keys():
            y_values = []
            for frame_data in all_frames.values():
                coords = frame_data.get(landmark_name, {})
                if 'y' in coords and coords['y'] is not None:
                    y_values.append(coords['y'])

            if y_values:
                y_variance = np.var(y_values)
                landmark_y_variance[landmark_name] = y_variance

        if landmark_y_variance:
            # Calculate average Y-variance across all landmarks
            avg_y_variance = np.mean(list(landmark_y_variance.values()))

            # For reference, calculate X and Z variances
            x_values = []
            z_values = []
            for frame_data in all_frames.values():
                for coords in frame_data.values():
                    if all(k in coords for k in ['x', 'y', 'z']) and coords['x'] is not None:
                        x_values.append(coords['x'])
                        z_values.append(coords['z'])

            if x_values and z_values:
                x_variance = np.var(x_values)
                z_variance = np.var(z_values)

                # Use configurable variance ratio (default: 0.3 = 30% of min(X,Z))
                variance_ratio = self.sensitivity.get('planar_motion_variance_ratio', 0.3)

                # If Y variance is significant compared to X/Z, we need more aggressive scaling
                if avg_y_variance > min(x_variance, z_variance) * variance_ratio:
                    reduction_factor = min(x_variance, z_variance) / avg_y_variance
                    script_log(f"Planar motion detection: Y_variance={avg_y_variance:.4f}, "
                               f"min(X,Z)_variance={min(x_variance, z_variance):.4f}, "
                               f"reduction_factor={reduction_factor:.3f}, "
                               f"trigger_ratio={variance_ratio:.1f}")

                    min_scale = self.sensitivity.get('min_scale_factor', 0.1)
                    return max(min_scale, reduction_factor * 0.5)  # Extra aggressive for planar motion

        return 1.0

    def _calculate_aggressive_depth_reduction(self, all_frames):
        """Calculate very aggressive scaling for known planar motion"""
        # Find the median Y value and scale to minimize variation around it
        all_y_values = []
        for frame_data in all_frames.values():
            for coords in frame_data.values():
                if 'y' in coords and coords['y'] is not None:
                    all_y_values.append(coords['y'])

        if not all_y_values:
            return 0.1

        y_median = np.median(all_y_values)
        y_mad = np.median(np.abs(all_y_values - y_median))  # Median Absolute Deviation

        # If there's significant variation from median, scale it down aggressively
        if y_mad > 0.05:  # More than 5cm variation
            scale = 0.05 / y_mad  # Target 5cm total variation
            script_log(f"Aggressive depth reduction: MAD={y_mad:.3f}, scale={scale:.3f}")
            return max(0.01, min(0.3, scale))

        return 0.1  # Default aggressive scaling

    def apply_depth_scaling(self, all_frames, scale_factor):
        """Apply depth scaling while preserving shoulder proportions"""
        if abs(scale_factor - 1.0) < self.epsilon:
            return all_frames

        script_log(f"Applying depth scaling factor: {scale_factor:.3f} to Y-axis")

        scaled_frames = {}
        for frame_num, frame_data in all_frames.items():
            scaled_frame = {}

            # Calculate shoulder center before scaling
            left_shoulder = frame_data.get('LEFT_SHOULDER')
            right_shoulder = frame_data.get('RIGHT_SHOULDER')
            shoulder_center_y = None

            if left_shoulder and right_shoulder:
                shoulder_center_y = (left_shoulder['y'] + right_shoulder['y']) / 2

            for landmark_name, coords in frame_data.items():
                if all(k in coords for k in ['x', 'y', 'z']) and coords['x'] is not None:
                    # Special handling for shoulders to preserve width
                    if landmark_name in ['LEFT_SHOULDER', 'RIGHT_SHOULDER'] and shoulder_center_y is not None:
                        # Scale shoulder positions relative to shoulder center to preserve width
                        offset_from_center = coords['y'] - shoulder_center_y
                        scaled_y = shoulder_center_y + (offset_from_center * scale_factor)
                    else:
                        # Normal scaling for other landmarks
                        scaled_y = coords['y'] * scale_factor

                    scaled_frame[landmark_name] = {
                        'x': coords['x'],
                        'y': scaled_y,
                        'z': coords['z'],
                        'visibility': coords.get('visibility', 0.0)
                    }
                else:
                    scaled_frame[landmark_name] = coords
            scaled_frames[frame_num] = scaled_frame

        return scaled_frames

    def process_frames(self, all_frames):
        """Main method to detect and correct depth exaggeration"""
        script_log("Starting depth adjustment analysis...")

        # Calculate biomechanical metrics
        limb_lengths = self.calculate_biomechanical_metrics(all_frames)

        # Detect depth exaggeration using multiple strategies
        depth_scale = self.detect_depth_exaggeration(limb_lengths, all_frames)

        # Apply scaling if needed
        if abs(depth_scale - 1.0) > self.epsilon:
            return self.apply_depth_scaling(all_frames, depth_scale)
        else:
            script_log("No significant depth exaggeration detected.")
            return all_frames


def calculate_3d_distance(p1, p2):
    """
    Calculates the 3D Euclidean distance between two points.
    Points are expected as dictionaries with 'x', 'y', 'z' keys.
    Returns None if any coordinate is missing.
    """
    if not (p1 and p2 and
            p1.get('x') is not None and p1.get('y') is not None and p1.get('z') is not None and
            p2.get('x') is not None and p2.get('y') is not None and p2.get('z') is not None):
        return None

    return math.sqrt((p2['x'] - p1['x']) ** 2 +
                     (p2['y'] - p1['y']) ** 2 +
                     (p2['z'] - p1['z']) ** 2)


def calculate_angle_3d(p1, p2, p3):
    """
    Calculates the angle (in radians) at point p2 formed by vectors p2->p1 and p2->p3.
    Points are expected as dictionaries with 'x', 'y', 'z' keys.
    Returns None if any coordinate is missing or vectors are degenerate.
    """
    if not (p1 and p2 and p3 and
            all(k in p1 for k in ['x', 'y', 'z']) and p1['x'] is not None and
            all(k in p2 for k in ['x', 'y', 'z']) and p2['x'] is not None and
            all(k in p3 for k in ['x', 'y', 'z']) and p3['x'] is not None):
        return None

    vec1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
    vec2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    if norm_vec1 < 1e-6 or norm_vec2 < 1e-6:
        return None  # Degenerate vectors

    dot_product = np.dot(vec1, vec2)

    # Clamp dot_product to avoid floating point errors leading to NaN in arccos
    clamped_dot_product = np.clip(dot_product / (norm_vec1 * norm_vec2), -1.0, 1.0)

    angle_rad = math.acos(clamped_dot_product)
    return angle_rad


def apply_physics(input_file, output_file, output_biometrics_file):
    """
    Transforms MediaPipe pose data by applying various physics-based and biomechanical
    constraints and transformations.

    Args:
        input_file (str): Path to the input JSON file containing MediaPipe pose data.
        output_file (str): Path to the output JSON file for transformed data.
        output_biometrics_file (str): Path to the output JSON file for biometric data.
    """
    # Load landmark configuration
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        landmark_config_path = os.path.join(script_dir, LANDMARK_CONFIG_FILE)
        landmark_config = load_landmark_config(landmark_config_path)
        script_log(f"Loaded landmark configuration: {len(landmark_config.get('landmarks', {}))} landmarks")
        script_log(f"Config file MediaPipe version: {landmark_config.get('mediapipe_version', 'unknown')}")
    except Exception as e:
        script_log(f"WARNING: Could not load landmark config from {LANDMARK_CONFIG_FILE}: {e}")
        landmark_config = {"landmarks": {}}

    if not os.path.exists(input_file):
        script_log(f"Error: Input file not found at '{input_file}'")
        return

    try:
        with open(input_file, 'r') as f:
            all_frames_data = json.load(f)

        if not all_frames_data:
            script_log("No frame data found in the input JSON file for transformation.")
            return

        # --- Step 1: Improved Coordinate System Transformation ---
        script_log("Starting improved Z-up transformation...")

        if not TURN_OFF_Z_UP_TRANSFORMATION:
            transformed_frames_data = {}

            # Initialize stabilizers if enabled
            shoulder_stabilizer = None
            if not TURN_OFF_SHOULDER_STABILIZATION:
                shoulder_stabilizer = ShoulderStabilizer(landmark_config=landmark_config)
                script_log("Shoulder stabilization enabled")
            else:
                script_log("Shoulder stabilization disabled")

            head_stabilizer = None
            if not TURN_OFF_HEAD_STABILIZATION:
                head_stabilizer = HeadStabilizer(landmark_config=landmark_config)
                script_log("Head stabilization enabled")
            else:
                script_log("Head stabilization disabled")

            script_log("Looping through all_frames_data.items()")
            for frame_num, frame_data in all_frames_data.items():
                print_debug = False
                if random.random() < 0.03:
                    print_debug = True

                if print_debug:
                    script_log(f"Loopy frame_num: {frame_num}")
                transformed_frame_data = {}

                if print_debug:
                    script_log(f"Frame {frame_num}: Coord transform")

                # First pass: Apply basic coordinate transformation
                if print_debug:
                    script_log(f"First Pass")
                for landmark_name, coords in frame_data.items():
                    if print_debug:
                        script_log(f"Loopier frame_num: {frame_num}, landmark_name: {landmark_name}, coords: {coords}")

                    if all(k in coords for k in ['x', 'y', 'z']) and coords['x'] is not None:
                        transformed_frame_data[landmark_name] = {
                            'x': round(coords['x'], 4),
                            'y': round(coords['z'], 4),  # Z becomes Y (forward)
                            'z': round(-coords['y'], 4),  # -Y becomes Z (up)
                            'visibility': round(coords.get('visibility', 0.0), 4)
                        }

                if print_debug:
                    script_log(f"Second Pass: Frame {frame_num}: Shoulder Stab")
                # Second pass: Analyze and stabilize shoulder-hip relationships if enabled
                if shoulder_stabilizer:
                    frame_analysis = shoulder_stabilizer.analyze_shoulder_hip_relationship(transformed_frame_data)

                    if frame_analysis:
                        scaling_issues = shoulder_stabilizer.detect_scaling_mismatch(frame_analysis)

                        if scaling_issues:
                            script_log(
                                f"Frame {frame_num}: Shoulder scaling issues detected - {', '.join(scaling_issues)}")
                            # Apply stabilization
                            transformed_frame_data = shoulder_stabilizer.stabilize_shoulders(transformed_frame_data,
                                                                                             frame_analysis)

                            # Log correction
                            new_analysis = shoulder_stabilizer.analyze_shoulder_hip_relationship(transformed_frame_data)
                            if new_analysis:
                                script_log(
                                    f"Frame {frame_num}: Corrected vertical offset from {frame_analysis['vertical_offset']:.3f} to {new_analysis['vertical_offset']:.3f}")

                transformed_frames_data[frame_num] = transformed_frame_data

                if print_debug:
                    script_log(f"Frame {frame_num}: Transformed frame data")

            script_log("Applied coordinate transformation")

            # --- Depth Adjustment: Correct exaggerated depth estimates ---
            if not TURN_OFF_DEPTH_ADJUSTMENT:
                script_log("Starting depth adjustment for exaggerated depth estimates...")
                depth_adjuster = DepthAdjuster(landmark_config=landmark_config)
                transformed_frames_data = depth_adjuster.process_frames(transformed_frames_data)
                script_log("Completed depth adjustment")
            else:
                script_log("Depth adjustment skipped (turn_off_depth_adjustment is True).")

            # --- Head Stabilization: Position head relative to body (AFTER depth adjustment) ---
            if head_stabilizer and not TURN_OFF_HEAD_STABILIZATION:
                script_log("Starting head stabilization...")
                stabilized_frames_data = {}
                for frame_num, frame_data in transformed_frames_data.items():
                    # Apply head stabilization AFTER depth adjustment
                    stabilized_frame = head_stabilizer.process_head_frame(frame_data)
                    stabilized_frames_data[frame_num] = stabilized_frame
                transformed_frames_data = stabilized_frames_data
                script_log("Completed head stabilization")

            # The HIP_HEEL_FLOOR_SHIFT only makes sense if we have already done the Z-Up Rotation
            if not TURN_OFF_HIP_HEEL_FLOOR_SHIFT:
                script_log("Applying enhanced HIP_HEEL_FLOOR_SHIFT with state machine...")
                height_adjuster = HeightAdjuster(landmark_config=landmark_config)
                transformed_frames_data = height_adjuster.process_frames(transformed_frames_data)
                script_log("Completed HIP_HEEL_FLOOR_SHIFT transformation")
            else:
                script_log("HIP_HEEL_FLOOR_SHIFT skipped (turn_off_hip_heel_floor_shift is True).")

        else:
            script_log("Z-up transformation skipped (turn_off_z_up_transformation is True).")
            transformed_frames_data = all_frames_data

        # --- Step 2: Ground the feet at Z=0 ---
        script_log("Grounding feet at Z=0...")

        # Find the lowest point across all frames to establish ground level
        global_min_z = float('inf')
        for frame_data in transformed_frames_data.values():
            for coords in frame_data.values():
                if (isinstance(coords, dict) and 'z' in coords):
                    try:
                        z_val = float(coords['z'])
                        if np.isfinite(z_val):
                            global_min_z = min(global_min_z, z_val)
                    except (ValueError, TypeError):
                        continue

        # Shift all frames so the lowest point is at Z=0
        if global_min_z != float('inf'):
            for frame_data in transformed_frames_data.values():
                for coords in frame_data.values():
                    if (isinstance(coords, dict) and 'z' in coords):
                        try:
                            z_val = float(coords['z'])
                            if np.isfinite(z_val):
                                coords['z'] = z_val - global_min_z
                        except (ValueError, TypeError):
                            continue
            script_log(f"Grounded data by shifting Z coordinates by {-global_min_z:.4f}")
        else:
            script_log("Warning: Could not determine ground level")

        if not TURN_OFF_FORWARD_TRANSFORMATION:
            # --- Step 2: Determine Forward Orientation (Body facing +Y) ---
            script_log("Determining forward orientation...")
            avg_foot_forward_vector = np.array([0.0, 0.0, 0.0])
            count_foot_vectors = 0
            avg_body_normal_vector = np.array([0.0, 0.0, 0.0])
            count_body_normals = 0

            for frame_data in transformed_frames_data.values():
                # Get landmarks from config
                left_foot_index = frame_data.get("LEFT_FOOT_INDEX")
                right_foot_index = frame_data.get("RIGHT_FOOT_INDEX")
                left_heel = frame_data.get("LEFT_HEEL")
                right_heel = frame_data.get("RIGHT_HEEL")
                left_shoulder = frame_data.get("LEFT_SHOULDER")
                right_shoulder = frame_data.get("RIGHT_SHOULDER")
                left_hip = frame_data.get("LEFT_HIP")
                right_hip = frame_data.get("RIGHT_HIP")

                # Foot forward vector
                if (left_foot_index and right_foot_index and left_heel and right_heel and
                        all(k in left_foot_index for k in ['x', 'y', 'z']) and left_foot_index['x'] is not None and
                        all(k in right_foot_index for k in ['x', 'y', 'z']) and right_foot_index['x'] is not None and
                        all(k in left_heel for k in ['x', 'y', 'z']) and left_heel['x'] is not None and
                        all(k in right_heel for k in ['x', 'y', 'z']) and right_heel['x'] is not None):
                    foot_index_midpoint = np.array([(left_foot_index['x'] + right_foot_index['x']) / 2,
                                                    (left_foot_index['y'] + right_foot_index['y']) / 2,
                                                    (left_foot_index['z'] + right_foot_index['z']) / 2])
                    heel_midpoint = np.array([(left_heel['x'] + right_heel['x']) / 2,
                                              (left_heel['y'] + right_heel['y']) / 2,
                                              (left_heel['z'] + right_heel['z']) / 2])
                    foot_forward_vec_current = foot_index_midpoint - heel_midpoint
                    if np.linalg.norm(foot_forward_vec_current) > 1e-6:
                        avg_foot_forward_vector += foot_forward_vec_current / np.linalg.norm(foot_forward_vec_current)
                        count_foot_vectors += 1

                # Body normal vector (for front/back orientation)
                if (left_shoulder and right_shoulder and left_hip and right_hip and
                        all(k in left_shoulder for k in ['x', 'y', 'z']) and left_shoulder['x'] is not None and
                        all(k in right_shoulder for k in ['x', 'y', 'z']) and right_shoulder['x'] is not None and
                        all(k in left_hip for k in ['x', 'y', 'z']) and left_hip['x'] is not None and
                        all(k in right_hip for k in ['x', 'y', 'z']) and right_hip['x'] is not None):

                    shoulder_midpoint = np.array([(left_shoulder['x'] + right_shoulder['x']) / 2,
                                                  (left_shoulder['y'] + right_shoulder['y']) / 2,
                                                  (left_shoulder['z'] + right_shoulder['z']) / 2])
                    hip_midpoint = np.array([(left_hip['x'] + right_hip['x']) / 2,
                                             (left_hip['y'] + right_hip['y']) / 2,
                                             (left_hip['z'] + right_hip['z']) / 2])
                    up_vector_body = shoulder_midpoint - hip_midpoint

                    shoulder_vec = np.array([right_shoulder['x'] - left_shoulder['x'],
                                             right_shoulder['y'] - left_shoulder['y'],
                                             right_shoulder['z'] - left_shoulder['z']])

                    body_normal_current = np.cross(shoulder_vec, up_vector_body)
                    if np.linalg.norm(body_normal_current) > 1e-6:
                        avg_body_normal_vector += body_normal_current / np.linalg.norm(body_normal_current)
                        count_body_normals += 1

            if count_foot_vectors > 0:
                avg_foot_forward_vector /= count_foot_vectors
                # Project onto the X-Y plane (since we've already done Z-up)
                avg_foot_forward_vector[2] = 0.0
                if np.linalg.norm(avg_foot_forward_vector) > 1e-6:
                    avg_foot_forward_vector /= np.linalg.norm(avg_foot_forward_vector)

            if count_body_normals > 0:
                avg_body_normal_vector /= count_body_normals
                # Project onto the X-Y plane
                avg_body_normal_vector[2] = 0.0
                if np.linalg.norm(avg_body_normal_vector) > 1e-6:
                    avg_body_normal_vector /= np.linalg.norm(avg_body_normal_vector)

            # Use feet forward vector as the primary orientation
            if np.linalg.norm(avg_foot_forward_vector) > 1e-6:
                forward_vector_figure = avg_foot_forward_vector
            elif np.linalg.norm(avg_body_normal_vector) > 1e-6:
                # Fallback to body normal if feet are not a good indicator
                # We want the normal to be pointing "forward" along the Y axis
                forward_vector_figure = np.cross(np.array([0, 0, 1]), avg_body_normal_vector)
            else:
                forward_vector_figure = np.array([0, 1, 0])  # Default to Y-axis forward

            desired_forward_vector = np.array([0, 1, 0])  # +Y axis
            # Calculate rotation matrix for forward orientation
            # Axis of rotation is the Z-axis (since we are on the X-Y plane)
            angle_forward = math.atan2(desired_forward_vector[1], desired_forward_vector[0]) - \
                            math.atan2(forward_vector_figure[1], forward_vector_figure[0])

            forward_rotation_matrix = np.array([[np.cos(angle_forward), -np.sin(angle_forward), 0],
                                                [np.sin(angle_forward), np.cos(angle_forward), 0],
                                                [0, 0, 1]])

            script_log(f"Debug: Forward Rotation Matrix (Z-axis rotation): \n{forward_rotation_matrix}")

            # Apply forward rotation to all data
            for frame_num in transformed_frames_data:
                frame_data = transformed_frames_data[frame_num]
                for landmark_name, coords in frame_data.items():
                    if all(k in coords for k in ['x', 'y', 'z']) and coords['x'] is not None:
                        current_vector = np.array([coords['x'], coords['y'], coords['z']])
                        rotated_vector = np.dot(forward_rotation_matrix, current_vector)
                        frame_data[landmark_name] = {
                            'x': round(rotated_vector[0], 4),
                            'y': round(rotated_vector[1], 4),
                            'z': round(rotated_vector[2], 4),
                            'visibility': round(coords.get('visibility', 0.0), 4)
                        }

            script_log(
                f"Data rotated {math.degrees(angle_forward):.2f} degrees around the Z-axis to face forward (+Y).")

            # DEBUG: Log after forward transformation
            script_log("=== DEBUG: Z-values after forward transformation ===")
            for frame_num in range(170, 182):
                frame_key = str(frame_num)
                if frame_key in transformed_frames_data:
                    frame_data = transformed_frames_data[frame_key]
                    wrist_landmarks = [name for name in landmark_config.get("landmarks", {}) if "WRIST" in name]
                    hip_landmarks = [name for name in landmark_config.get("landmarks", {}) if "HIP" in name]
                    shoulder_landmarks = [name for name in landmark_config.get("landmarks", {}) if "SHOULDER" in name]

                    if wrist_landmarks and hip_landmarks and shoulder_landmarks:
                        wrist = frame_data.get(wrist_landmarks[0], {})
                        hip = frame_data.get(hip_landmarks[0], {})
                        shoulder = frame_data.get(shoulder_landmarks[0], {})

                        wrist_z = wrist.get('z', 'N/A')
                        hip_z = hip.get('z', 'N/A')
                        shoulder_z = shoulder.get('z', 'N/A')

                        dist_to_hip = abs(wrist_z - hip_z) if all(
                            isinstance(v, (int, float)) for v in [wrist_z, hip_z]) else 'N/A'
                        dist_to_shoulder = abs(wrist_z - shoulder_z) if all(
                            isinstance(v, (int, float)) for v in [wrist_z, shoulder_z]) else 'N/A'

                        script_log(f"Frame {frame_num}: Wrist_Z={wrist_z}, Hip_Z={hip_z}, Shoulder_Z={shoulder_z}, "
                                   f"DistToHip={dist_to_hip}, DistToShoulder={dist_to_shoulder}")
        else:
            script_log("Forward transformation skipped (turn_off_forward_transformation is True).")

        # Debug: Check what's in the first frame
        if transformed_frames_data:
            first_frame_key = list(transformed_frames_data.keys())[0]
            first_frame = transformed_frames_data[first_frame_key]
            script_log(f"Debug: First frame keys: {list(first_frame.keys())}")

            # Check first landmark
            first_landmark_key = list(first_frame.keys())[0]
            first_landmark = first_frame[first_landmark_key]
            script_log(f"Debug: First landmark ({first_landmark_key}): {first_landmark}")
            script_log(f"Debug: First landmark type: {type(first_landmark)}")

            if isinstance(first_landmark, dict):
                script_log(f"Debug: First landmark keys: {list(first_landmark.keys())}")

        # --- Step 3: Global Scaling and Centering ---
        script_log("Applying global scaling and centering...")
        # Get min/max Z to determine height
        min_z = float('inf')
        max_z = float('-inf')
        all_x = []
        all_y = []

        for frame_data in transformed_frames_data.values():
            for coords in frame_data.values():
                # More robust check that handles numpy types
                if (isinstance(coords, dict) and
                        'x' in coords and 'y' in coords and 'z' in coords):

                    # Convert to float to handle numpy types and check if valid
                    try:
                        x_val = float(coords['x'])
                        y_val = float(coords['y'])
                        z_val = float(coords['z'])

                        # Check if values are finite numbers (not NaN or inf)
                        if (np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val)):
                            min_z = min(min_z, z_val)
                            max_z = max(max_z, z_val)
                            all_x.append(x_val)
                            all_y.append(y_val)
                    except (ValueError, TypeError):
                        # Skip coordinates that can't be converted
                        continue

        # Check if we found any valid coordinates
        if not all_x or min_z == float('inf'):
            script_log("Warning: No valid coordinates found for scaling. Skipping scaling step.")
        else:
            initial_height = max_z - min_z
            desired_height = CONFIG['height_adjustment']['desired_height']  # meters
            scale_factor = desired_height / initial_height if initial_height > CONFIG['general']['epsilon'] else 1.0

            center_x = (min(all_x) + max(all_x)) / 2 if all_x else 0.0
            center_y = (min(all_y) + max(all_y)) / 2 if all_y else 0.0

            script_log(
                f"Debug: Scaling factors - Height: {initial_height:.3f}m, Scale: {scale_factor:.3f}, Center: ({center_x:.3f}, {center_y:.3f})")

            # Apply scaling and centering to all frames
            for frame_data in transformed_frames_data.values():
                for landmark_name, coords in frame_data.items():
                    if (isinstance(coords, dict) and
                            'x' in coords and 'y' in coords and 'z' in coords):

                        try:
                            # Convert to float to handle numpy types
                            x_val = float(coords['x'])
                            y_val = float(coords['y'])
                            z_val = float(coords['z'])

                            if (np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val)):
                                # Apply scaling only
                                coords['x'] = x_val * scale_factor
                                coords['y'] = y_val * scale_factor
                                coords['z'] = z_val * scale_factor

                                # Apply centering only (no Z-offset)
                                coords['x'] -= (center_x * scale_factor)
                                coords['y'] -= (center_y * scale_factor)
                        except (ValueError, TypeError):
                            # Skip coordinates that can't be processed
                            continue

            script_log(f"Data scaled to a height of {desired_height}m and centered at (0,0).")

        # Ensure no negative Z values in all frames
        for frame_data in transformed_frames_data.values():
            # Find lowest point in each frame
            frame_min_z = float('inf')
            for coords in frame_data.values():
                if (isinstance(coords, dict) and 'z' in coords):
                    try:
                        z_val = float(coords['z'])
                        if np.isfinite(z_val):
                            frame_min_z = min(frame_min_z, z_val)
                    except (ValueError, TypeError):
                        continue

            # Shift entire frame up if any points are below Z=0
            if frame_min_z < 0 and frame_min_z != float('inf'):
                for coords in frame_data.values():
                    if (isinstance(coords, dict) and 'z' in coords):
                        try:
                            z_val = float(coords['z'])
                            if np.isfinite(z_val):
                                coords['z'] = z_val - frame_min_z
                        except (ValueError, TypeError):
                            continue

        script_log(f"Data scaled to a height of {desired_height}m and centered at (0,0).")

        if not TURN_OFF_BIOMECHANICAL_CONSTRAINTS:
            # --- Step 4: Biomechanical Constraints ---
            script_log("Applying biomechanical constraints...")
            # Get frame-by-frame biomechanical data
            biometrics_frames = {}
            for frame_num, frame_data in transformed_frames_data.items():

                biometrics_frame = {}

                # Shoulder to shoulder distance
                left_shoulder = frame_data.get('LEFT_SHOULDER')
                right_shoulder = frame_data.get('RIGHT_SHOULDER')
                if left_shoulder and right_shoulder:
                    shoulder_distance = calculate_3d_distance(left_shoulder, right_shoulder)
                    biometrics_frame['shoulder_distance'] = shoulder_distance

                # Hip to hip distance
                left_hip = frame_data.get('LEFT_HIP')
                right_hip = frame_data.get('RIGHT_HIP')
                if left_hip and right_hip:
                    hip_distance = calculate_3d_distance(left_hip, right_hip)
                    biometrics_frame['hip_distance'] = hip_distance

                # Arm lengths (upper and forearm)
                left_elbow = frame_data.get('LEFT_ELBOW')
                left_wrist = frame_data.get('LEFT_WRIST')
                left_shoulder = frame_data.get('LEFT_SHOULDER')
                if left_shoulder and left_elbow:
                    biometrics_frame['upper_arm_left_length'] = calculate_3d_distance(left_shoulder, left_elbow)
                if left_elbow and left_wrist:
                    biometrics_frame['forearm_left_length'] = calculate_3d_distance(left_elbow, left_wrist)

                right_elbow = frame_data.get('RIGHT_ELBOW')
                right_wrist = frame_data.get('RIGHT_WRIST')
                right_shoulder = frame_data.get('RIGHT_SHOULDER')
                if right_shoulder and right_elbow:
                    biometrics_frame['upper_arm_right_length'] = calculate_3d_distance(right_shoulder, right_elbow)
                if right_elbow and right_wrist:
                    biometrics_frame['forearm_right_length'] = calculate_3d_distance(right_elbow, right_wrist)

                # Leg lengths
                left_knee = frame_data.get('LEFT_KNEE')
                left_ankle = frame_data.get('LEFT_ANKLE')
                if left_hip and left_knee:
                    biometrics_frame['left_leg_length'] = calculate_3d_distance(left_hip, left_knee)
                if left_knee and left_ankle:
                    biometrics_frame['left_shin_length'] = calculate_3d_distance(left_knee, left_ankle)

                right_knee = frame_data.get('RIGHT_KNEE')
                right_ankle = frame_data.get('RIGHT_ANKLE')
                if right_hip and right_knee:
                    biometrics_frame['right_leg_length'] = calculate_3d_distance(right_hip, right_knee)
                if right_knee and right_ankle:
                    biometrics_frame['right_shin_length'] = calculate_3d_distance(right_knee, right_ankle)

                # Torso length (mid-hip to mid-shoulder)
                if left_hip and right_hip and left_shoulder and right_shoulder:
                    hip_mid = {
                        'x': (left_hip['x'] + right_hip['x']) / 2,
                        'y': (left_hip['y'] + right_hip['y']) / 2,
                        'z': (left_hip['z'] + right_hip['z']) / 2
                    }
                    shoulder_mid = {
                        'x': (left_shoulder['x'] + right_shoulder['x']) / 2,
                        'y': (left_shoulder['y'] + right_shoulder['y']) / 2,
                        'z': (left_shoulder['z'] + right_shoulder['z']) / 2
                    }
                    biometrics_frame['torso_length'] = calculate_3d_distance(hip_mid, shoulder_mid)

                biometrics_frames[frame_num] = biometrics_frame

            # Calculate average biomechanics for the entire sequence
            frame_count = len(biometrics_frames)
            if frame_count > 0:
                total_shoulder_d = sum(
                    f['shoulder_distance'] for f in biometrics_frames.values() if 'shoulder_distance' in f)
                total_hip_d = sum(f['hip_distance'] for f in biometrics_frames.values() if 'hip_distance' in f)
                total_upper_arm_l_len = sum(
                    f['upper_arm_left_length'] for f in biometrics_frames.values() if 'upper_arm_left_length' in f)
                total_upper_arm_r_len = sum(
                    f['upper_arm_right_length'] for f in biometrics_frames.values() if 'upper_arm_right_length' in f)
                total_forearm_l_len = sum(
                    f['forearm_left_length'] for f in biometrics_frames.values() if 'forearm_left_length' in f)
                total_forearm_r_len = sum(
                    f['forearm_right_length'] for f in biometrics_frames.values() if 'forearm_right_length' in f)
                total_torso_len = sum(f['torso_length'] for f in biometrics_frames.values() if 'torso_length' in f)
                total_left_leg_len = sum(
                    f['left_leg_length'] for f in biometrics_frames.values() if 'left_leg_length' in f)
                total_right_leg_len = sum(
                    f['right_leg_length'] for f in biometrics_frames.values() if 'right_leg_length' in f)

                count_shoulder = sum(1 for f in biometrics_frames.values() if 'shoulder_distance' in f)
                count_hip = sum(1 for f in biometrics_frames.values() if 'hip_distance' in f)
                count_upper_arm_l = sum(1 for f in biometrics_frames.values() if 'upper_arm_left_length' in f)
                count_upper_arm_r = sum(1 for f in biometrics_frames.values() if 'upper_arm_right_length' in f)
                count_forearm_l = sum(1 for f in biometrics_frames.values() if 'forearm_left_length' in f)
                count_forearm_r = sum(1 for f in biometrics_frames.values() if 'forearm_right_length' in f)
                count_torso = sum(1 for f in biometrics_frames.values() if 'torso_length' in f)
                count_left_leg = sum(1 for f in biometrics_frames.values() if 'left_leg_length' in f)
                count_right_leg = sum(1 for f in biometrics_frames.values() if 'right_leg_length' in f)

                avg_shoulder_d = total_shoulder_d / count_shoulder if count_shoulder > 0 else 0
                avg_hip_d = total_hip_d / count_hip if count_hip > 0 else 0
                avg_upper_arm_l_length = total_upper_arm_l_len / count_upper_arm_l if count_upper_arm_l > 0 else 0
                avg_upper_arm_r_length = total_upper_arm_r_len / count_upper_arm_r if count_upper_arm_r > 0 else 0
                avg_forearm_l_length = total_forearm_l_len / count_forearm_l if count_forearm_l > 0 else 0
                avg_forearm_r_length = total_forearm_r_len / count_forearm_r if count_forearm_r > 0 else 0
                avg_torso_length = total_torso_len / count_torso if count_torso > 0 else 0
                avg_left_leg_length = total_left_leg_len / count_left_leg if count_left_leg > 0 else 0
                avg_right_leg_length = total_right_leg_len / count_right_leg if count_right_leg > 0 else 0

            # Create the final biometric data object
            biometric_data = {
                "version": "1.0",
                "date_created": datetime.now().isoformat(),
                "average_shoulder_distance": avg_shoulder_d,
                "average_hip_distance": avg_hip_d,
                "average_upper_arm_left_length": avg_upper_arm_l_length,
                "average_upper_arm_right_length": avg_upper_arm_r_length,
                "average_forearm_left_length": avg_forearm_l_length,
                "average_forearm_right_length": avg_forearm_r_length,
                "average_torso_length": avg_torso_length,
                "average_left_leg_length": avg_left_leg_length,
                "average_right_leg_length": avg_right_leg_length,
                # Add head motion limits
                "head_roll_limit_degrees": CONFIG['biomechanical_constraints']['head_roll_limit_degrees'],
                "head_nod_limit_degrees": CONFIG['biomechanical_constraints']['head_nod_limit_degrees'],
                "head_y_translation_limit_percent_of_height": CONFIG['biomechanical_constraints'][
                    'head_y_translation_limit_percent'],
                # Add head motion sensitivities
                "head_roll_sensitivity": CONFIG['biomechanical_constraints']['head_roll_sensitivity'],
                "head_nod_sensitivity": CONFIG['biomechanical_constraints']['head_nod_sensitivity']
            }

            with open(output_biometrics_file, 'w') as f:
                json.dump(biometric_data, f, indent=4)
            script_log(f"Successfully estimated biometrics and saved to '{output_biometrics_file}'.")
        else:
            script_log("Forward biometric constraints skipped (turn_off_biomechanical_constraints is True).")

        # Save transformed data
        with open(output_file, 'w') as f:
            json.dump(transformed_frames_data, f, indent=4)
        script_log(f"Successfully processed data and saved to '{output_file}'.")

    except Exception as e:
        script_log(f"Error transforming data from '{input_file}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply physics transformations to pose data")
    parser.add_argument('--show', type=str, default=None, help='Override the default show name')
    parser.add_argument('--scene', type=str, default=None, help='Override the current scene name')
    args = parser.parse_args()

    show_name = args.show if args.show else get_current_show_name()
    scene_name = args.scene if args.scene else get_current_scene_name(show_name)

    script_log(f"=== PHYSICS APPLICATION STARTED ===")
    script_log(f"Show: {show_name}")
    script_log(f"Scene: {scene_name}")

    # Get processing step paths from scene-config.json using utils
    step_paths = get_processing_step_paths(show_name, scene_name, "apply_physics")
    input_json_file_data = step_paths['input_file']
    output_json_file = step_paths['output_file']
    output_biometrics_file = step_paths['biometrics_output_file']  # From additional_outputs

    script_log(f"Input: {input_json_file_data}")
    script_log(f"Output: {output_json_file}")
    script_log(f"Biometrics: {output_biometrics_file}")

    # Run the physics application
    apply_physics(input_json_file_data, output_json_file, output_biometrics_file)
    script_log("=== PHYSICS APPLICATION COMPLETE ===")