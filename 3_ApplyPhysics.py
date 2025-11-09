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
ENABLE_Z_UP_TRANSFORMATION = debug_flags.get('enable_z_up_transformation', True)
ENABLE_HEAD_OVER_HEELS = debug_flags.get('enable_head_over_heels', True)
ENABLE_FORWARD_TRANSFORMATION = debug_flags.get('enable_forward_transformation', True)
ENABLE_SHOULDER_SHIFT = debug_flags.get('enable_shoulder_shift', True)
ENABLE_HIP_SHIFT = debug_flags.get('enable_hip_shift', True)
ENABLE_BIOMECHANICAL_CONSTRAINTS = debug_flags.get('enable_biomechanical_constraints', True)
ENABLE_HIP_HEEL_FLOOR_SHIFT = debug_flags.get('enable_hip_heel_floor_shift', True)
ENABLE_DEPTH_ADJUSTMENT = debug_flags.get('enable_depth_adjustment', True)
ENABLE_SHOULDER_STABILIZATION = debug_flags.get('enable_shoulder_stabilization', True)
ENABLE_HEAD_STABILIZATION = debug_flags.get('enable_head_stabilization', True)
ENABLE_FOOT_FLATTENING = debug_flags.get('enable_foot_flattening', True)


def head_over_heels(all_frames_data):
    """Detect if figure is inverted and flip Y coordinates if needed.

    Compares the average Y value of HEAD_TOP landmark to the average Y value
    of LEFT_FOOT_INDEX landmark to determine if the figure is inverted. If inverted,
    changes the sign of the Y coordinates on all landmarks in all frames.

    Args:
        all_frames_data: Dictionary of frame data with landmark coordinates

    Returns:
        Dictionary of frame data with Y coordinates flipped if inverted
    """
    script_log("Checking for inverted figure (head over heels)...")

    # Calculate average Y values for HEAD_TOP and LEFT_FOOT_INDEX across all frames
    head_top_y_values = []
    left_foot_y_values = []

    for frame_num, frame_data in all_frames_data.items():
        head_top = frame_data.get('HEAD_TOP')
        left_foot = frame_data.get('LEFT_FOOT_INDEX')

        if head_top and 'y' in head_top and head_top['y'] is not None:
            head_top_y_values.append(head_top['y'])

        if left_foot and 'y' in left_foot and left_foot['y'] is not None:
            left_foot_y_values.append(left_foot['y'])

    # If we don't have enough data, return original frames
    if not head_top_y_values or not left_foot_y_values:
        script_log("WARNING: Insufficient data to detect inversion - HEAD_TOP or LEFT_FOOT_INDEX landmarks missing")
        return all_frames_data

    # Calculate average Y positions
    avg_head_top_y = sum(head_top_y_values) / len(head_top_y_values)
    avg_left_foot_y = sum(left_foot_y_values) / len(left_foot_y_values)

    script_log(f"Average HEAD_TOP Y: {avg_head_top_y:.4f}")
    script_log(f"Average LEFT_FOOT_INDEX Y: {avg_left_foot_y:.4f}")

    # Check if figure is inverted (head lower than feet in Y-axis)
    # In MediaPipe coordinates: +Y is downward, so head should have lower Y values than feet normally
    # If head has higher Y values than feet, the figure is inverted
    is_inverted = avg_head_top_y > avg_left_foot_y

    script_log(f"Figure inverted: {is_inverted}")

    # If not inverted, return original data
    if not is_inverted:
        script_log("Figure is properly oriented - no inversion correction needed")
        return all_frames_data

    # If inverted, flip Y coordinates for all landmarks in all frames
    script_log("Figure is inverted - flipping Y coordinates for all landmarks...")

    minus_one = -1
    for frame_num, frame_data in all_frames_data.items():
        for landmark_name, coords in frame_data.items():
            if (isinstance(coords, dict) and
                    'y' in coords and
                    coords['y'] is not None):
                # Flip Y coordinate
                coords['y'] = minus_one * coords['y']

    script_log("Successfully flipped Y coordinates to correct inverted figure")
    return all_frames_data

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


class FootFlattener:
    """Detects and corrects unnatural foot positioning in Blender coordinate system"""

    def __init__(self, landmark_config=None, motion_detector=None):
        script_log("Initializing FootFlattener for Blender coordinates")
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})

        # Use provided motion_detector or create new one
        self.motion_detector = motion_detector or MotionPhaseDetector(landmark_config=landmark_config)

        # Load config with new structure
        self.config = CONFIG.get('foot_flattener', {})
        self.grounding_controls = self.config.get('grounding_controls', {})
        self.smoothing_controls = self.config.get('smoothing_controls', {})
        self.biomechanical_constraints = self.config.get('biomechanical_constraints', {})

        self.epsilon = CONFIG['general']['epsilon']
        self.state_correction_history = deque(maxlen=10)

        # Debug log config values
        script_log(f"FootFlattener config loaded:")
        script_log(f"  Enable: {self.config.get('enable_foot_flattening', True)}")
        script_log(f"  Max foot lift grounded: {self.grounding_controls.get('max_foot_lift_grounded', 0.02)}")
        script_log(f"  Max foot lift jumping: {self.grounding_controls.get('max_foot_lift_jumping', 0.15)}")
        script_log(f"  Using shared motion detector: {motion_detector is not None}")

    def analyze_foot_contact(self, frame):
        """Analyze foot positioning and return correction parameters"""
        left_heel = frame.get('LEFT_HEEL')
        right_heel = frame.get('RIGHT_HEEL')
        left_foot_index = frame.get('LEFT_FOOT_INDEX')
        right_foot_index = frame.get('RIGHT_FOOT_INDEX')
        left_ankle = frame.get('LEFT_ANKLE')
        right_ankle = frame.get('RIGHT_ANKLE')

        if not all([left_heel, right_heel, left_foot_index, right_foot_index, left_ankle, right_ankle]):
            return None

        # Calculate current foot positions
        left_heel_z = left_heel.get('z', 0)
        right_heel_z = right_heel.get('z', 0)
        left_toe_z = left_foot_index.get('z', 0)
        right_toe_z = right_foot_index.get('z', 0)

        # Calculate foot angles and relationships
        left_foot_angle = self._calculate_foot_angle(left_heel, left_foot_index, left_ankle)
        right_foot_angle = self._calculate_foot_angle(right_heel, right_foot_index, right_ankle)

        # Determine if correction is needed
        max_heel_elevation = self.grounding_controls.get('max_foot_lift_grounded', 0.05)
        heels_elevated = (left_heel_z > max_heel_elevation or
                          right_heel_z > max_heel_elevation)

        min_toe_contact = 0.05  # Fixed value
        toes_elevated = (left_toe_z > min_toe_contact or
                         right_toe_z > min_toe_contact)

        needs_correction = heels_elevated or toes_elevated

        # Calculate confidence based on foot positioning consistency
        confidence = self._calculate_foot_confidence(
            left_heel_z, right_heel_z, left_toe_z, right_toe_z,
            left_foot_angle, right_foot_angle
        )

        return {
            'needs_correction': needs_correction,
            'confidence': confidence,
            'left_heel_z': left_heel_z,
            'right_heel_z': right_heel_z,
            'left_toe_z': left_toe_z,
            'right_toe_z': right_toe_z,
            'left_foot_angle': left_foot_angle,
            'right_foot_angle': right_foot_angle,
            'heels_elevated': heels_elevated,
            'toes_elevated': toes_elevated
        }

    def _calculate_foot_angle(self, heel, foot_index, ankle):
        """Calculate the angle of the foot relative to horizontal"""
        if not all([heel, foot_index, ankle]):
            return 0.0

        # Vector from heel to toe (foot direction)
        foot_vector = np.array([
            foot_index['x'] - heel['x'],
            foot_index['y'] - heel['y'],
            foot_index['z'] - heel['z']
        ])

        # Horizontal plane normal (up vector)
        horizontal_normal = np.array([0, 0, 1])

        # Project foot vector onto horizontal plane
        horizontal_foot = foot_vector - np.dot(foot_vector, horizontal_normal) * horizontal_normal

        if np.linalg.norm(horizontal_foot) < self.epsilon:
            return 0.0

        # Calculate angle between foot vector and horizontal projection
        dot_product = np.dot(foot_vector, horizontal_foot)
        magnitudes = np.linalg.norm(foot_vector) * np.linalg.norm(horizontal_foot)

        if magnitudes < self.epsilon:
            return 0.0

        cos_angle = dot_product / magnitudes
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        # Determine if angle is positive (toes up) or negative (heels up)
        if foot_vector[2] > horizontal_foot[2]:
            angle_deg = -angle_deg  # Heels up

        return angle_deg

    def _calculate_foot_confidence(self, left_heel_z, right_heel_z, left_toe_z, right_toe_z, left_angle, right_angle):
        """Calculate confidence score for foot contact analysis"""
        confidence = 1.0

        # Penalize if feet are at very different heights (unlikely in normal poses)
        heel_height_diff = abs(left_heel_z - right_heel_z)
        if heel_height_diff > 0.2:  # More than 20cm difference
            confidence *= 0.5

        # Penalize if foot angles are very different
        angle_diff = abs(left_angle - right_angle)
        if angle_diff > 30:  # More than 30 degrees difference
            confidence *= 0.7

        # Penalize if one foot is flat and other is highly elevated
        if (left_heel_z < 0.05 and right_heel_z > 0.15) or (right_heel_z < 0.05 and left_heel_z > 0.15):
            confidence *= 0.6

        return max(0.0, min(1.0, confidence))

    def should_apply_corrections(self, frame, motion_state):
        """Motion-state-aware decision making for foot corrections"""
        if not self.config.get('enable_foot_flattening', True):
            return False

        # Never correct during these states
        if motion_state in [MotionState.JUMPING, MotionState.FALLING, MotionState.FEET_OFF_FLOOR]:
            return False

        # State-specific correction logic
        if motion_state == MotionState.STANDING:
            return self._should_correct_standing(frame)

        elif motion_state == MotionState.WALKING:
            return self._should_correct_walking(frame)

        elif motion_state == MotionState.RUNNING:
            return self._should_correct_running(frame)

        elif motion_state == MotionState.LANDING:
            return self._should_correct_landing(frame)

        elif motion_state == MotionState.TRANSITION:
            return self._should_correct_transition(frame)

        return False

    def _should_correct_standing(self, frame):
        """Correction logic for STANDING state - less aggressive for extended feet"""
        foot_analysis = self.analyze_foot_contact(frame)
        if not foot_analysis:
            return False

        # Check if feet have been extended (longer than typical)
        left_foot_length = calculate_3d_distance(frame.get('LEFT_HEEL', {}), frame.get('LEFT_FOOT_INDEX', {}))
        right_foot_length = calculate_3d_distance(frame.get('RIGHT_HEEL', {}), frame.get('RIGHT_FOOT_INDEX', {}))

        # Typical unextended foot length is ~0.12m, extended is >0.15m
        typical_foot_length = 0.12
        feet_extended = (left_foot_length > typical_foot_length or
                         right_foot_length > typical_foot_length)

        # Be more lenient with extended feet
        if feet_extended:
            max_standing_heel_elevation = 0.04  # 4cm tolerance instead of 2cm
        else:
            max_standing_heel_elevation = self.grounding_controls.get('max_foot_lift_grounded', 0.02)

        heels_elevated = (foot_analysis['left_heel_z'] > max_standing_heel_elevation or
                          foot_analysis['right_heel_z'] > max_standing_heel_elevation)

        return heels_elevated and foot_analysis['confidence'] > 0.7

    def _should_correct_walking(self, frame):
        """Correction logic for WALKING state"""
        foot_analysis = self.analyze_foot_contact(frame)
        if not foot_analysis:
            return False

        # Walking: Only correct if foot is in stance phase (heel contact)
        is_stance_phase = self._is_stance_phase(frame)
        max_walking_heel_elevation = self.grounding_controls.get('max_foot_lift_grounded', 0.08)  # Slightly higher tolerance for walking
        heels_elevated = (foot_analysis['left_heel_z'] > max_walking_heel_elevation or
                          foot_analysis['right_heel_z'] > max_walking_heel_elevation)

        return is_stance_phase and heels_elevated and foot_analysis['confidence'] > 0.6

    def _should_correct_running(self, frame):
        """Correction logic for RUNNING state - very limited"""
        foot_analysis = self.analyze_foot_contact(frame)
        if not foot_analysis:
            return False

        # Running: Only correct for severe elevation issues
        max_running_heel_elevation = self.grounding_controls.get('max_foot_lift_jumping', 0.15)
        severe_elevation = (foot_analysis['left_heel_z'] > max_running_heel_elevation or
                            foot_analysis['right_heel_z'] > max_running_heel_elevation)

        return severe_elevation and foot_analysis['confidence'] > 0.8

    def _should_correct_landing(self, frame):
        """Correction logic for LANDING state - wait for stabilization"""
        # Check if we've been in landing state long enough
        landing_frames = sum(1 for state in self.state_correction_history
                             if state == MotionState.LANDING)
        min_landing_frames = self.smoothing_controls.get('min_frames_for_ground_contact', 3)

        if landing_frames < min_landing_frames:
            return False  # Wait for stabilization

        foot_analysis = self.analyze_foot_contact(frame)
        if not foot_analysis:
            return False

        return foot_analysis['needs_correction'] and foot_analysis['confidence'] > 0.7

    def _should_correct_transition(self, frame):
        """Very conservative for TRANSITION states"""
        foot_analysis = self.analyze_foot_contact(frame)
        if not foot_analysis:
            return False

        # Only correct if feet are extremely elevated during transition
        max_transition_elevation = 0.20  # Fixed value for extreme cases
        extreme_elevation = (foot_analysis['left_heel_z'] > max_transition_elevation or
                             foot_analysis['right_heel_z'] > max_transition_elevation)

        return extreme_elevation and foot_analysis['confidence'] > 0.9

    def _is_stance_phase(self, frame):
        """Detect if foot is in stance phase (heel contact) during walking"""
        left_heel_z = frame.get('LEFT_HEEL', {}).get('z', float('inf'))
        right_heel_z = frame.get('RIGHT_HEEL', {}).get('z', float('inf'))

        # In walking, one foot is usually lower (stance) and one higher (swing)
        min_heel_z = min(left_heel_z, right_heel_z)
        stance_threshold = 0.1  # Fixed threshold

        return min_heel_z < stance_threshold

    def _extend_foot_to_proportional_length(self, frame, foot_side):
        """Extend foot to be proportional to shin length if too short"""
        min_ratio = self.biomechanical_constraints.get('min_foot_length_to_shin_ratio', 0.0)

        # If ratio is 0, skip foot extension
        if min_ratio <= 0:
            return frame

        heel_key = f'{foot_side}_HEEL'
        foot_index_key = f'{foot_side}_FOOT_INDEX'
        ankle_key = f'{foot_side}_ANKLE'
        knee_key = f'{foot_side}_KNEE'

        heel = frame.get(heel_key)
        foot_index = frame.get(foot_index_key)
        ankle = frame.get(ankle_key)
        knee = frame.get(knee_key)

        if not all([heel, foot_index, ankle, knee]):
            return frame

        # Calculate current foot length (heel to foot_index)
        current_foot_length = calculate_3d_distance(heel, foot_index)

        # Calculate shin length (knee to ankle)
        shin_length = calculate_3d_distance(knee, ankle)

        if shin_length is None or shin_length < self.epsilon:
            return frame

        # Calculate minimum required foot length
        min_foot_length = shin_length * min_ratio

        # If foot is already long enough, no extension needed
        if current_foot_length >= min_foot_length:
            return frame

        # Calculate direction vector from heel to foot_index
        foot_vector = np.array([
            foot_index['x'] - heel['x'],
            foot_index['y'] - heel['y'],
            foot_index['z'] - heel['z']
        ])

        # Normalize and extend
        current_length = np.linalg.norm(foot_vector)
        if current_length > self.epsilon:
            foot_direction = foot_vector / current_length
            extension_vector = foot_direction * (min_foot_length - current_length)

            # Apply extension to foot_index only (keep heel grounded)
            frame[foot_index_key]['x'] += extension_vector[0]
            frame[foot_index_key]['y'] += extension_vector[1]
            frame[foot_index_key]['z'] += extension_vector[2]

            # Only log if actual extension occurred
            if abs(min_foot_length - current_length) > 0.001:
                comment(
                    f"DEBUG: Extended {foot_side} foot from {current_foot_length:.3f} to {min_foot_length:.3f} (shin: {shin_length:.3f})")

        return frame

    def apply_foot_correction(self, frame, analysis, motion_state):
        """Apply foot positioning corrections with motion state awareness"""
        if not analysis or not analysis['needs_correction']:
            return frame

        # State-specific correction parameters
        if motion_state == MotionState.STANDING:
            heel_strength = 0.4
            toe_strength = 0.3
        elif motion_state == MotionState.WALKING:
            heel_strength = 0.3
            toe_strength = 0.2
        elif motion_state == MotionState.RUNNING:
            heel_strength = 0.1
            toe_strength = 0.1
        else:
            heel_strength = 0.4
            toe_strength = 0.3

        max_correction = 0.15

        # Apply corrections to heels
        for heel_landmark in ['LEFT_HEEL', 'RIGHT_HEEL']:
            if heel_landmark in frame:
                current_z = frame[heel_landmark].get('z', 0)
                target_z = 0.0  # Ground level

                correction = (target_z - current_z) * heel_strength
                correction = max(-max_correction, min(max_correction, correction))

                frame[heel_landmark]['z'] = current_z + correction

        # Apply proportional corrections to toes to maintain foot shape
        for toe_landmark in ['LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']:
            if toe_landmark in frame:
                current_z = frame[toe_landmark].get('z', 0)
                # Toes get less correction to maintain natural foot angle
                correction = (0.0 - current_z) * toe_strength
                correction = max(-max_correction, min(max_correction, correction))

                frame[toe_landmark]['z'] = current_z + correction

        return frame

    def debug_foot_analysis(self, frame, frame_num):
        """Debug method to see what FootFlattener is detecting"""
        foot_analysis = self.analyze_foot_contact(frame)
        if not foot_analysis:
            script_log(f"DEBUG Frame {frame_num}: No foot analysis possible")
            return

        motion_state = self.motion_detector.detect_state(frame, None)
        should_correct = self.should_apply_corrections(frame, motion_state)

        script_log(f"DEBUG Frame {frame_num}:")
        script_log(f"  Motion State: {motion_state.name}")
        script_log(f"  Left Heel Z: {foot_analysis['left_heel_z']:.3f}")
        script_log(f"  Right Heel Z: {foot_analysis['right_heel_z']:.3f}")
        script_log(f"  Left Toe Z: {foot_analysis['left_toe_z']:.3f}")
        script_log(f"  Right Toe Z: {foot_analysis['right_toe_z']:.3f}")
        script_log(f"  Needs Correction: {foot_analysis['needs_correction']}")
        script_log(f"  Confidence: {foot_analysis['confidence']:.2f}")
        script_log(f"  Should Correct: {should_correct}")
        script_log(f"  Max Grounded Lift: {self.grounding_controls.get('max_foot_lift_grounded', 0.02)}")

    def process_frames(self, all_frames):
        """Main processing method using shared motion detector"""
        if not self.config.get('enable_foot_flattening', True):
            script_log("Foot flattening disabled via config")
            return all_frames

        script_log("Applying motion-aware foot flattening corrections...")
        corrected_frames = {}

        # Store original frames for comparison
        original_frames = all_frames.copy()

        # First: Apply foot length extension if configured
        min_ratio = self.biomechanical_constraints.get('min_foot_length_to_shin_ratio', 0.0)
        if min_ratio > 0:
            script_log(f"Applying foot length extension with ratio: {min_ratio}")
            for frame_num, frame in all_frames.items():
                # Create a copy to avoid modifying original during iteration
                frame_copy = {k: v.copy() for k, v in frame.items()}
                frame_copy = self._extend_foot_to_proportional_length(frame_copy, 'LEFT')
                frame_copy = self._extend_foot_to_proportional_length(frame_copy, 'RIGHT')
                all_frames[frame_num] = frame_copy

        # Use the shared motion detector to get states for all frames
        motion_states = {}
        prev_frame = None

        # First pass: detect motion states using shared detector
        for frame_num, frame in all_frames.items():
            motion_state = self.motion_detector.detect_state(frame, prev_frame)
            motion_states[frame_num] = motion_state
            self.state_correction_history.append(motion_state)
            prev_frame = frame

        # Second pass: apply state-aware corrections
        correction_count = 0
        for frame_num, frame in all_frames.items():
            motion_state = motion_states[frame_num]

            # DEBUG: See what's happening
            if int(frame_num) < 3:  # Only log first 3 frames to avoid spam
                self.debug_foot_analysis(frame, frame_num)

            if self.should_apply_corrections(frame, motion_state):
                analysis = self.analyze_foot_contact(frame)
                corrected_frame = self.apply_foot_correction(frame, analysis, motion_state)
                corrected_frames[frame_num] = corrected_frame
                correction_count += 1

                # Debug logging for first few corrections
                if correction_count <= 5:
                    script_log(f"Frame {frame_num} ({motion_state.name}): Corrected feet - "
                               f"LH: {analysis['left_heel_z']:.3f}->{corrected_frame['LEFT_HEEL']['z']:.3f}, "
                               f"RH: {analysis['right_heel_z']:.3f}->{corrected_frame['RIGHT_HEEL']['z']:.3f}")
            else:
                corrected_frames[frame_num] = frame

        script_log(f"Foot flattening completed: {correction_count}/{len(all_frames)} frames corrected")

        # Log correction distribution by motion state
        state_corrections = {}
        for frame_num, frame in corrected_frames.items():
            if frame_num in motion_states:
                state = motion_states[frame_num]
                if frame != all_frames[frame_num]:  # Frame was corrected
                    state_corrections[state] = state_corrections.get(state, 0) + 1

        if state_corrections:
            script_log("Corrections by motion state:")
            for state, count in state_corrections.items():
                script_log(f"  {state.name}: {count} frames")

        return corrected_frames


class HeadStabilizer:
    """Detects and corrects head position relative to body in Blender coordinate system"""

    def __init__(self, landmark_config=None):
        script_log("Initializing IMPROVED head stabilization for Blender coordinates")
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})

        # Add epsilon for numerical stability
        self.epsilon = CONFIG['general']['epsilon']

        self.config = CONFIG.get('head_stabilization', {})

        # Load all config sections
        self.distance_controls = self.config.get('distance_controls', {})
        self.height_controls = self.config.get('height_controls', {})
        self.forward_controls = self.config.get('forward_position_controls', {})
        self.body_plane = self.config.get('body_plane_alignment', {})
        self.proportional = self.config.get('proportional_corrections', {})
        self.smoothing = self.config.get('smoothing_controls', {})

        # Debug log config values
        script_log(f"IMPROVED Head stabilization config loaded:")
        script_log(
            f"  Distance: min={self.distance_controls.get('min_head_shoulder_distance')}, target={self.distance_controls.get('target_head_shoulder_distance')}")
        script_log(
            f"  Height: min={self.height_controls.get('min_head_height')}, target={self.height_controls.get('target_head_height')}")
        script_log(
            f"  Forward: min={self.forward_controls.get('min_head_forward_offset')}, target={self.forward_controls.get('target_head_forward_offset')}")

    ### METHOD ####################################################################################

    def calculate_body_plane(self, frame):
        """Calculate body plane using hips and shoulders

        HEAD_TOP should be on or in front of the body plane (plane_distance ≥ 0)
        NOSE should typically be on or in front of the body plane (plane_distance ≥ 0)
        Only in extreme head rotations (>90°) would the nose be behind the plane

        We're measuring how much each landmark projects onto the body's forward direction,
        not their actual perpendicular distance to an infinite plane.
        """
        left_shoulder = frame.get('LEFT_SHOULDER')
        right_shoulder = frame.get('RIGHT_SHOULDER')
        left_hip = frame.get('LEFT_HIP')

        if not all([left_shoulder, right_shoulder, left_hip]):
            return None

        ls = np.array([left_shoulder['x'], left_shoulder['y'], left_shoulder['z']])
        rs = np.array([right_shoulder['x'], right_shoulder['y'], right_shoulder['z']])
        lh = np.array([left_hip['x'], left_hip['y'], left_hip['z']])

        # Calculate shoulder midpoint
        shoulder_mid = (ls + rs) / 2

        # Define plane using three points: left_shoulder, right_shoulder, left_hip
        shoulder_vector = rs - ls  # Right direction
        hip_vector = lh - ls  # Down direction
        body_normal = np.cross(shoulder_vector, hip_vector)  # Forward

        # Normalize
        normal_length = np.linalg.norm(body_normal)
        if normal_length < self.epsilon:
            return None
        body_normal = body_normal / normal_length

        return {
            'body_normal': body_normal,
            'shoulder_mid': shoulder_mid,  # Keep this for compatibility
            'plane_reference': shoulder_mid
        }

    ### METHOD ####################################################################################

    def analyze_head_body_relationship(self, frame, body_plane):
        """Analyze head position relative to body plane with proper naming"""
        head_top = frame.get('HEAD_TOP')
        nose = frame.get('NOSE')

        if not all([head_top, nose, body_plane]):
            return None

        ht = np.array([head_top['x'], head_top['y'], head_top['z']])
        ns = np.array([nose['x'], nose['y'], nose['z']])
        body_normal = body_plane['body_normal']
        shoulder_mid = body_plane['shoulder_mid']

        # Calculate dot products with body normal (not "plane distances")
        head_to_ref_vector = ht - shoulder_mid
        nose_to_ref_vector = ns - shoulder_mid

        head_dot_product = np.dot(head_to_ref_vector, body_normal)
        nose_dot_product = np.dot(nose_to_ref_vector, body_normal)

        # Nose behind body plane is the primary issue to detect
        is_behind_body_plane = nose_dot_product < 0

        # Calculate vertical distance from shoulders (Z component in Blender)
        shoulder_to_head = ht - shoulder_mid
        head_vertical = shoulder_to_head[2]

        # Calculate head height (HEAD_TOP to NOSE distance)
        head_height = np.linalg.norm(ht - ns)

        # Calculate nose forward advantage relative to HEAD_TOP in body forward direction
        head_to_nose = ns - ht
        nose_forward = np.dot(head_to_nose, body_normal)  # Use body normal, not Y component

        # DEBUG: Log the actual vectors and calculations
        script_log(f"DEBUG HEAD ANALYSIS:")
        script_log(f"  Shoulder mid: {shoulder_mid}")
        script_log(f"  HEAD_TOP: {ht}")
        script_log(f"  NOSE: {ns}")
        script_log(f"  Body normal: {body_normal}")
        script_log(f"  Head to shoulder vector: {head_to_ref_vector}")
        script_log(f"  Nose to shoulder vector: {nose_to_ref_vector}")
        script_log(f"  Head dot: {head_dot_product:.3f}")
        script_log(f"  Nose dot: {nose_dot_product:.3f}")

        return {
            'head_dot_product': head_dot_product,
            'nose_dot_product': nose_dot_product,
            'head_vertical': head_vertical,
            'head_height': head_height,
            'nose_forward': nose_forward,
            'is_behind_body_plane': is_behind_body_plane,
            'head_top_position': ht,
            'nose_position': ns,
            'body_normal': body_normal
        }

    ### METHOD ####################################################################################

    def detect_head_issues(self, head_analysis, frame, body_plane):
        """Detect if head needs correction using proper dot product naming"""
        if not head_analysis:
            return None

        # Use config values
        MIN_HEAD_HEIGHT = self.height_controls.get('min_head_height', 0.15)
        MIN_HEAD_VERTICAL = self.distance_controls.get('min_head_shoulder_distance', 0.10)
        NOSE_FORWARD_ADVANTAGE = self.forward_controls.get('nose_forward_advantage', 0.02)
        MIN_NOSE_LENGTH = self.forward_controls.get('minimum_nose_length', 0.07)
        MAX_NOSE_LENGTH = self.forward_controls.get('maximum_nose_length', 0.12)

        issues = []

        # Check if NOSE is behind body plane (primary detection)
        if head_analysis['is_behind_body_plane']:
            issues.append(f"nose_behind_body_plane_{head_analysis['nose_dot_product']:.3f}")

        # Check if HEAD_TOP is behind body plane (secondary detection)
        if head_analysis['head_dot_product'] < 0:
            issues.append(f"head_behind_body_plane_{head_analysis['head_dot_product']:.3f}")

        # Check head height
        if head_analysis['head_height'] < MIN_HEAD_HEIGHT:
            issues.append(f"head_squished_{head_analysis['head_height']:.3f}")

        # Check if head is too low relative to shoulders
        if head_analysis['head_vertical'] < MIN_HEAD_VERTICAL:
            issues.append(f"head_too_low_{head_analysis['head_vertical']:.3f}")

        # Check if nose is sufficiently in front of HEAD_TOP in body forward direction
        if head_analysis['nose_forward'] < NOSE_FORWARD_ADVANTAGE:
            issues.append(f"nose_not_forward_enough_{head_analysis['nose_forward']:.3f}")

        # Check if nose is too long (unrealistic)
        if head_analysis['nose_forward'] > MAX_NOSE_LENGTH:
            issues.append(f"nose_too_long_{head_analysis['nose_forward']:.3f}")

        # Check if nose is too short
        if head_analysis['nose_forward'] < MIN_NOSE_LENGTH:
            issues.append(f"nose_too_short_{head_analysis['nose_forward']:.3f}")

        return issues if issues else None

    ### METHOD ####################################################################################

    ### METHOD ####################################################################################

    def correct_head_position(self, frame, head_analysis, body_plane):
        """Apply corrections using head_dot_product for proper head stabilization"""
        # Ensure analysis was performed and necessary landmarks exist
        if not head_analysis:
            script_log("DEBUG: No head_analysis - returning frame unchanged")
            return frame

        head_top = frame.get('HEAD_TOP')
        nose = frame.get('NOSE')

        if not all([head_top, nose]):
            script_log("DEBUG: Missing HEAD_TOP or NOSE - returning frame unchanged")
            return frame

        # Get config values (NOSE_CORRECTION_RATIO is preserved for potential future use or context)
        NOSE_CORRECTION_RATIO = self.proportional.get('nose_correction_ratio', 0.7)
        MAX_NOSE_LENGTH = self.forward_controls.get('maximum_nose_length', 0.12)
        MIN_NOSE_LENGTH = self.forward_controls.get('minimum_nose_length', 0.07)

        body_normal = body_plane['body_normal']
        # Note: head_dot_product is assumed to be calculated in a preceding analysis step
        head_dot_product = head_analysis['head_dot_product']
        nose_forward = head_analysis['nose_forward']

        # DEBUG: Log positions before correction
        script_log(f"DEBUG PRE-CORRECTION:")
        script_log(f"  HEAD_TOP: ({head_top['x']:.3f}, {head_top['y']:.3f}, {head_top['z']:.3f})")
        script_log(f"  NOSE: ({nose['x']:.3f}, {nose['y']:.3f}, {nose['z']:.3f})")
        script_log(f"  Head dot product: {head_dot_product:.3f}")
        script_log(f"  Nose forward: {nose_forward:.3f}")
        script_log(f"  Max nose length: {MAX_NOSE_LENGTH:.3f}")

        # 1. Correction for NOSE being too long (forward but stretched)
        # This original logic is preserved: it only moves the NOSE backward.
        if nose_forward > MAX_NOSE_LENGTH and not head_analysis['is_behind_body_plane']:
            # Nose is too long but in front of plane - move it backward toward HEAD_TOP
            excess_length = nose_forward - MAX_NOSE_LENGTH
            correction_vector = -body_normal * excess_length  # Move backward

            # Apply correction to NOSE only (keep HEAD_TOP fixed)
            nose['x'] += correction_vector[0]
            nose['y'] += correction_vector[1]
            nose['z'] += correction_vector[2]

            # DEBUG: Log this specific correction
            script_log(f"DEBUG NOSE-TOO-LONG CORRECTION:")
            script_log(f"  Correction vector: {correction_vector}")

        # 2. Rigid-Body Correction: Triggered ONLY if HEAD_TOP is behind the body plane.
        # This implements your requested rigid-body logic.
        elif head_dot_product < 0:

            # Calculate correction distance needed to move HEAD_TOP onto the plane (dot=0).
            correction_distance = -head_dot_product
            correction_vector = body_normal * correction_distance

            # Apply full correction to HEAD_TOP (Moving it exactly onto the plane)
            head_top['x'] += correction_vector[0]
            head_top['y'] += correction_vector[1]
            head_top['z'] += correction_vector[2]

            # Apply the same full correction to NOSE (rigid-body move).
            # This ensures NOSE moves forward by the same amount as HEAD_TOP.
            nose['x'] += correction_vector[0]
            nose['y'] += correction_vector[1]
            nose['z'] += correction_vector[2]

            # DEBUG: Log positions after correction
            script_log(f"DEBUG HEAD-ANCHORED CORRECTION:")
            script_log(f"  HEAD_TOP: ({head_top['x']:.3f}, {head_top['y']:.3f}, {head_top['z']:.3f})")
            script_log(f"  NOSE: ({nose['x']:.3f}, {nose['y']:.3f}, {nose['z']:.3f})")
            script_log(f"  Correction applied: {np.linalg.norm(correction_vector):.3f} units")

        else:
            # If HEAD_TOP is on or in front of the plane, and NOSE is not too long, no correction is needed.
            script_log("DEBUG: No head stabilization correction needed")

        return frame

    ### METHOD ####################################################################################

    def process_head_frame(self, frame):
        """Main method to detect and correct head position for a single frame"""

        # Debug: Check if correct_head_position is what we expect
        current_method = self.correct_head_position
        script_log(f"DEBUG: correct_head_position method: {current_method}")
        script_log(f"DEBUG: correct_head_position module: {getattr(current_method, '__module__', 'unknown')}")
        script_log(f"DEBUG: correct_head_position name: {getattr(current_method, '__name__', 'unknown')}")

        # Calculate body reference frame using correct body plane
        body_plane = self.calculate_body_plane(frame)

        if not body_plane:
            script_log("WARNING: No valid body plane - skipping head stabilization")
            return frame

        # Analyze head position relative to body plane
        head_analysis = self.analyze_head_body_relationship(frame, body_plane)

        if not head_analysis:
            return frame

        # DEBUG: Log the analysis results with new naming convention
        script_log(f"DEBUG Head Analysis: head_dot={head_analysis['head_dot_product']:.3f}, "
                   f"nose_dot={head_analysis['nose_dot_product']:.3f}, "
                   f"vertical={head_analysis['head_vertical']:.3f}, height={head_analysis['head_height']:.3f}, "
                   f"nose_forward={head_analysis['nose_forward']:.3f}, "
                   f"is_behind_plane={head_analysis['is_behind_body_plane']}")

        # Detect issues
        issues = self.detect_head_issues(head_analysis, frame, body_plane)

        script_log(f"DEBUG: Issues detected: {issues}")

        if issues:
            script_log(f"Head stabilization issues: {issues}")

            # ✅ Save original values BEFORE correction
            original_head_y = frame['HEAD_TOP']['y']
            original_nose_y = frame['NOSE']['y']
            original_head_x = frame['HEAD_TOP']['x']
            original_nose_x = frame['NOSE']['x']
            original_head_z = frame['HEAD_TOP']['z']
            original_nose_z = frame['NOSE']['z']

            # Apply correction
            frame = self.correct_head_position(frame, head_analysis, body_plane)

            # Now we can log the changes in all coordinates
            script_log(
                f"DEBUG: Head changed from ({original_head_x:.3f}, {original_head_y:.3f}, {original_head_z:.3f}) "
                f"to ({frame['HEAD_TOP']['x']:.3f}, {frame['HEAD_TOP']['y']:.3f}, {frame['HEAD_TOP']['z']:.3f})")
            script_log(
                f"DEBUG: Nose changed from ({original_nose_x:.3f}, {original_nose_y:.3f}, {original_nose_z:.3f}) "
                f"to ({frame['NOSE']['x']:.3f}, {frame['NOSE']['y']:.3f}, {frame['NOSE']['z']:.3f})")
        else:
            script_log("DEBUG: No head stabilization issues detected")

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
        self.previous_foot_z = None
        self.state_cache = {}  # Cache for frame states to improve performance

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
        # Create a simple hash for caching
        frame_hash = self._frame_hash(frame)
        if frame_hash in self.state_cache:
            return self.state_cache[frame_hash]

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
        foot_vel = 0.0
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

        # Calculate hip velocity for additional motion context
        hip_vel = self._calculate_hip_velocity(frame, prev_frame)

        # State detection logic
        new_state = self._determine_primary_state(
            foot_min_z, hand_min_z, head_min_z, foot_vel, hip_vel, frame
        )

        # State transition smoothing
        self.state_history.append(new_state)

        # Require consistent state for window_size frames before changing
        if len(self.state_history) == self.window_size:
            if all(s == new_state for s in self.state_history):
                if new_state != self.current_state:
                    script_log(f"Motion state transition: {self.current_state.name} -> {new_state.name}")
                self.current_state = new_state
            else:
                # If inconsistent, stay in transition state
                self.current_state = MotionState.TRANSITION
        else:
            # Until we have enough history, use the new state directly
            self.current_state = new_state

        # Cache the result
        self.state_cache[frame_hash] = self.current_state

        # Clean cache if it gets too large (prevent memory issues)
        if len(self.state_cache) > 1000:
            self.state_cache.clear()

        return self.current_state

    def _determine_primary_state(self, foot_min_z, hand_min_z, head_min_z, foot_vel, hip_vel, frame):
        """Determine the primary motion state based on all available data"""
        feet_off_ground_threshold = CONFIG['motion_detection']['feet_off_ground_threshold']
        upward_velocity_threshold = CONFIG['motion_detection']['upward_velocity_threshold']
        downward_velocity_threshold = CONFIG['motion_detection']['downward_velocity_threshold']
        running_velocity_threshold = CONFIG['motion_detection']['running_velocity_threshold']
        walking_velocity_threshold = CONFIG['motion_detection']['walking_velocity_threshold']

        # Check for FEET_OFF_FLOOR state (handstand, headstand, etc.)
        min_body_z = self.lowest_z_of_body_parts(frame)
        if (min_body_z < foot_min_z and
                (hand_min_z == min_body_z or head_min_z == min_body_z)):
            return MotionState.FEET_OFF_FLOOR

        # Check for jumping/landing based on foot position and velocity
        if foot_min_z > feet_off_ground_threshold:
            if foot_vel > upward_velocity_threshold:
                return MotionState.JUMPING
            elif foot_vel < downward_velocity_threshold:
                return MotionState.LANDING
            else:
                return MotionState.FALLING  # Mid-air with neutral velocity

        # Check for walking/running when feet are near ground
        elif 0 < foot_min_z <= feet_off_ground_threshold:
            # Use both foot and hip velocity for better motion detection
            overall_velocity = max(abs(foot_vel), abs(hip_vel))

            if overall_velocity > running_velocity_threshold:
                return MotionState.RUNNING
            elif overall_velocity > walking_velocity_threshold:
                return MotionState.WALKING
            else:
                return MotionState.STANDING

        # Default to standing if feet are on ground with no significant motion
        else:
            return MotionState.STANDING

    def _calculate_hip_velocity(self, current_frame, previous_frame):
        """Calculate vertical velocity of hips between frames"""
        if not previous_frame:
            return 0.0

        left_hip_current = current_frame.get('LEFT_HIP')
        right_hip_current = current_frame.get('RIGHT_HIP')
        left_hip_prev = previous_frame.get('LEFT_HIP')
        right_hip_prev = previous_frame.get('RIGHT_HIP')

        if not all([left_hip_current, right_hip_current, left_hip_prev, right_hip_prev]):
            return 0.0

        # Calculate hip midpoint for both frames
        current_hip_mid = (
                                  left_hip_current.get('z', 0) + right_hip_current.get('z', 0)
                          ) / 2

        previous_hip_mid = (
                                   left_hip_prev.get('z', 0) + right_hip_prev.get('z', 0)
                           ) / 2

        return current_hip_mid - previous_hip_mid

    def _frame_hash(self, frame):
        """Create a simple hash of frame data for caching"""
        # Use a subset of key landmarks for hashing to balance performance and accuracy
        hash_landmarks = ['LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_HIP', 'RIGHT_HIP', 'HEAD_TOP']
        hash_values = []

        for landmark in hash_landmarks:
            coords = frame.get(landmark, {})
            if isinstance(coords, dict):
                # Round to 3 decimal places to handle floating point variations
                x = round(coords.get('x', 0), 3)
                z = round(coords.get('z', 0), 3)
                hash_values.extend([x, z])

        return tuple(hash_values)

    def get_motion_confidence(self, frame):
        """Calculate confidence score for the current motion state classification"""
        foot_landmarks = []
        for landmark_name, landmark_data in self.landmarks.items():
            if "FOOT" in landmark_name and landmark_data.get("motion_detection", False):
                foot_landmarks.append(landmark_name)

        # Calculate foot position consistency
        foot_z_values = []
        for foot_landmark in foot_landmarks:
            foot_z = frame.get(foot_landmark, {}).get('z', None)
            if foot_z is not None:
                foot_z_values.append(foot_z)

        if not foot_z_values:
            return 0.0

        # Confidence based on foot position spread (more consistent = higher confidence)
        foot_spread = max(foot_z_values) - min(foot_z_values)
        spread_confidence = max(0.0, 1.0 - (foot_spread * 10.0))  # Normalize

        # Confidence based on proximity to state thresholds
        state_threshold_confidence = self._calculate_threshold_confidence(frame)

        return (spread_confidence + state_threshold_confidence) / 2.0

    def _calculate_threshold_confidence(self, frame):
        """Calculate confidence based on distance from state transition thresholds"""
        current_state = self.current_state

        if current_state in [MotionState.STANDING, MotionState.WALKING, MotionState.RUNNING]:
            # For ground states, confidence is high if feet are clearly on ground
            foot_landmarks = []
            for landmark_name, landmark_data in self.landmarks.items():
                if "FOOT" in landmark_name and landmark_data.get("motion_detection", False):
                    foot_landmarks.append(landmark_name)

            foot_z_values = [frame.get(foot, {}).get('z', 1.0) for foot in foot_landmarks]
            if foot_z_values:
                min_foot_z = min(foot_z_values)
                # Higher confidence when feet are clearly on ground (low Z)
                return max(0.0, 1.0 - (min_foot_z * 5.0))

        elif current_state in [MotionState.JUMPING, MotionState.FALLING]:
            # For aerial states, confidence is high when feet are clearly elevated
            foot_landmarks = []
            for landmark_name, landmark_data in self.landmarks.items():
                if "FOOT" in landmark_name and landmark_data.get("motion_detection", False):
                    foot_landmarks.append(landmark_name)

            foot_z_values = [frame.get(foot, {}).get('z', 0.0) for foot in foot_landmarks]
            if foot_z_values:
                min_foot_z = min(foot_z_values)
                threshold = CONFIG['motion_detection']['feet_off_ground_threshold']
                # Higher confidence when feet are clearly above threshold
                return min(1.0, max(0.0, (min_foot_z - threshold) * 10.0))

        return 0.5  # Default medium confidence

    def reset_state(self):
        """Reset the motion state detector to initial conditions"""
        self.state_history.clear()
        self.current_state = MotionState.STANDING
        self.previous_foot_z = None
        self.state_cache.clear()
        script_log("MotionPhaseDetector state reset to STANDING")

    def get_state_history(self):
        """Get the recent state history for analysis"""
        return list(self.state_history)

    def is_state_stable(self, state=None):
        """Check if the current (or specified) state is stable"""
        target_state = state or self.current_state
        return all(s == target_state for s in self.state_history)

    def get_state_duration(self):
        """Get how many frames the current state has been active"""
        if not self.state_history:
            return 0

        duration = 0
        for state in reversed(self.state_history):
            if state == self.current_state:
                duration += 1
            else:
                break
        return duration


class HeightAdjuster:
    def __init__(self, landmark_config=None, motion_detector=None):
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})

        # Use provided motion_detector or create new one
        self.motion_detector = motion_detector or MotionPhaseDetector(landmark_config=landmark_config)

        self.previous_frames = deque(maxlen=CONFIG['motion_detection']['jump_detection_min_frames'])
        self.ground_history = deque(maxlen=CONFIG['height_adjustment']['ground_history_size'])
        self.jump_trajectory = None

        script_log(f"HeightAdjuster initialized with shared motion detector: {motion_detector is not None}")

    def process_frames(self, all_frames):
        """Main processing method using shared motion detector"""
        # First pass: Detect motion phases using shared detector
        states = {}
        prev_frame = None
        for frame_num, frame in sorted(all_frames.items(), key=lambda x: int(x[0])):
            state = self.motion_detector.detect_state(frame, prev_frame)
            states[frame_num] = state
            prev_frame = frame
            self.previous_frames.append(frame)

        # Second pass: Apply phase-specific corrections
        correction_stats = {
            'feet_off_floor': 0,
            'foot_contact': 0,
            'jumping': 0,
            'landing': 0,
            'falling': 0
        }

        for frame_num, frame in all_frames.items():
            state = states[frame_num]

            if state == MotionState.FEET_OFF_FLOOR:
                self.adjust_feet_off_floor(frame)
                correction_stats['feet_off_floor'] += 1
            elif state in [MotionState.WALKING, MotionState.RUNNING, MotionState.STANDING]:
                self.adjust_foot_contact(frame)
                correction_stats['foot_contact'] += 1
            elif state == MotionState.JUMPING:
                self.adjust_jumping(frame, frame_num, all_frames)
                correction_stats['jumping'] += 1
            elif state == MotionState.LANDING:
                self.adjust_landing(frame)
                correction_stats['landing'] += 1
            elif state == MotionState.FALLING:
                self.adjust_falling(frame)
                correction_stats['falling'] += 1

        # Log correction statistics
        script_log(f"Height adjustment completed - Corrections by state:")
        script_log(f"  FEET_OFF_FLOOR: {correction_stats['feet_off_floor']} frames")
        script_log(f"  FOOT_CONTACT: {correction_stats['foot_contact']} frames")
        script_log(f"  JUMPING: {correction_stats['jumping']} frames")
        script_log(f"  LANDING: {correction_stats['landing']} frames")
        script_log(f"  FALLING: {correction_stats['falling']} frames")

        # Third pass: Physics-based smoothing
        self.apply_physics_smoothing(all_frames)

        return all_frames

    def adjust_feet_off_floor(self, frame):
        """Adjust height when other joints are lowest point"""
        # Find lowest body part
        min_z = self.motion_detector.lowest_z_of_body_parts(frame)

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
            script_log(f"INFO: Insufficient samples ({len(foot_heights)}) for jump trajectory fitting")
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
            script_log(f"WARNING: Jump trajectory fitting failed: {str(e)}")

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
        # This is a placeholder for more sophisticated physics-based smoothing
        # Currently, the motion state detection and state-specific adjustments
        # provide sufficient smoothing for most cases

        # Future enhancement could include:
        # - Velocity continuity between frames
        # - Acceleration-based smoothing
        # - Spring-damper simulation for landings
        # - Momentum preservation during jumps

        pass

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

    def get_foot_landmarks(self):
        """Get all foot-related landmarks from config"""
        foot_landmarks = []
        for landmark_name, landmark_data in self.landmarks.items():
            if any(foot_term in landmark_name for foot_term in ["FOOT", "ANKLE", "HEEL"]):
                foot_landmarks.append(landmark_name)
        return foot_landmarks

    def calculate_vertical_velocity(self, current_frame, previous_frame):
        """Calculate vertical velocity between frames"""
        if not previous_frame:
            return 0.0

        # Use hip midpoint for velocity calculation
        left_hip = current_frame.get('LEFT_HIP')
        right_hip = current_frame.get('RIGHT_HIP')
        prev_left_hip = previous_frame.get('LEFT_HIP')
        prev_right_hip = previous_frame.get('RIGHT_HIP')

        if all([left_hip, right_hip, prev_left_hip, prev_right_hip]):
            current_hip_z = (left_hip['z'] + right_hip['z']) / 2
            previous_hip_z = (prev_left_hip['z'] + prev_right_hip['z']) / 2
            return current_hip_z - previous_hip_z

        return 0.0

    def is_balanced_stance(self, frame):
        """Check if the pose represents a balanced standing stance"""
        left_heel = frame.get('LEFT_HEEL')
        right_heel = frame.get('RIGHT_HEEL')
        left_foot_index = frame.get('LEFT_FOOT_INDEX')
        right_foot_index = frame.get('RIGHT_FOOT_INDEX')

        if not all([left_heel, right_heel, left_foot_index, right_foot_index]):
            return False

        # Check if feet are reasonably close to ground
        max_heel_height = 0.1  # 10cm max heel elevation for balanced stance
        left_heel_z = left_heel.get('z', float('inf'))
        right_heel_z = right_heel.get('z', float('inf'))

        heels_low = left_heel_z < max_heel_height and right_heel_z < max_heel_height

        # Check if feet are roughly symmetrical
        heel_height_diff = abs(left_heel_z - right_heel_z)
        symmetrical = heel_height_diff < 0.05  # Less than 5cm difference

        return heels_low and symmetrical


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
    """
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

        script_log("Starting Z-up transformation...")

        if ENABLE_Z_UP_TRANSFORMATION:
            transformed_frames_data = {}

            if ENABLE_HEAD_OVER_HEELS:
                all_frames_data = head_over_heels(all_frames_data)

            shoulder_stabilizer = None
            if ENABLE_SHOULDER_STABILIZATION:
                shoulder_stabilizer = ShoulderStabilizer(landmark_config=landmark_config)
                script_log("Shoulder stabilization enabled")
            else:
                script_log("Shoulder stabilization disabled")

            head_stabilizer = None
            if ENABLE_HEAD_STABILIZATION:
                head_stabilizer = HeadStabilizer(landmark_config=landmark_config)
                script_log("Head stabilization enabled")
            else:
                script_log("Head stabilization disabled")

            # SHARED MOTION DETECTOR - Create once for FootFlattener and HeightAdjuster
            shared_motion_detector = None
            if ENABLE_HIP_HEEL_FLOOR_SHIFT or ENABLE_FOOT_FLATTENING:
                shared_motion_detector = MotionPhaseDetector(landmark_config=landmark_config)
                script_log("Created shared motion detector for foot and height processing")

            script_log("Looping through all_frames_data.items()")
            for frame_num, frame_data in all_frames_data.items():
                if int(frame_num) == 0:  # Just check first frame
                    left_heel_original = frame_data.get('LEFT_HEEL', {})
                    script_log(
                        f"DEBUG Original LEFT_HEEL: ({left_heel_original.get('x')}, {left_heel_original.get('y')}, {left_heel_original.get('z')})")

                print_debug = False
                if random.random() < 0.03:
                    print_debug = True

                if print_debug:
                    comment(f"Loopy frame_num: {frame_num}")
                transformed_frame_data = {}

                if print_debug:
                    comment(f"Frame {frame_num}: Coord transform")

                for landmark_name, coords in frame_data.items():
                    if print_debug:
                        comment(f"Loopier frame_num: {frame_num}, landmark_name: {landmark_name}, coords: {coords}")

                    if all(k in coords for k in ['x', 'y', 'z']) and coords['x'] is not None:
                        minus_one = -1
                        transformed_frame_data[landmark_name] = {
                            'x': round(coords['x'], 4),
                            'y': round(coords['z'], 4),
                            'z': minus_one * round(coords['y'], 4),
                            'visibility': round(coords.get('visibility', 0.0), 4)
                        }

                if print_debug:
                    comment(f"Second Pass: Frame {frame_num}: Shoulder Stab")

                if shoulder_stabilizer:
                    frame_analysis = shoulder_stabilizer.analyze_shoulder_hip_relationship(transformed_frame_data)

                    if frame_analysis:
                        scaling_issues = shoulder_stabilizer.detect_scaling_mismatch(frame_analysis)

                        if scaling_issues:
                            script_log(
                                f"Frame {frame_num}: Shoulder scaling issues detected - {', '.join(scaling_issues)}")
                            transformed_frame_data = shoulder_stabilizer.stabilize_shoulders(transformed_frame_data,
                                                                                             frame_analysis)

                            new_analysis = shoulder_stabilizer.analyze_shoulder_hip_relationship(transformed_frame_data)
                            if new_analysis:
                                script_log(
                                    f"Frame {frame_num}: Corrected vertical offset from {frame_analysis['vertical_offset']:.3f} to {new_analysis['vertical_offset']:.3f}")

                transformed_frames_data[frame_num] = transformed_frame_data

                if int(frame_num) == 0:
                    left_heel_transformed = transformed_frame_data.get('LEFT_HEEL', {})
                    script_log(
                        f"DEBUG Transformed LEFT_HEEL: ({left_heel_transformed.get('x')}, {left_heel_transformed.get('y')}, {left_heel_transformed.get('z')})")

                if print_debug:
                    comment(f"Frame {frame_num}: Transformed frame data")

            script_log("Applied coordinate transformation")

            if ENABLE_FORWARD_TRANSFORMATION:
                script_log("Determining forward orientation...")
                avg_foot_forward_vector = np.array([0.0, 0.0, 0.0])
                count_foot_vectors = 0
                avg_body_normal_vector = np.array([0.0, 0.0, 0.0])
                count_body_normals = 0

                for frame_data in transformed_frames_data.values():
                    left_foot_index = frame_data.get("LEFT_FOOT_INDEX")
                    right_foot_index = frame_data.get("RIGHT_FOOT_INDEX")
                    left_heel = frame_data.get("LEFT_HEEL")
                    right_heel = frame_data.get("RIGHT_HEEL")
                    left_shoulder = frame_data.get("LEFT_SHOULDER")
                    right_shoulder = frame_data.get("RIGHT_SHOULDER")
                    left_hip = frame_data.get("LEFT_HIP")
                    right_hip = frame_data.get("RIGHT_HIP")

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
                    avg_foot_forward_vector[2] = 0.0
                    if np.linalg.norm(avg_foot_forward_vector) > 1e-6:
                        avg_foot_forward_vector /= np.linalg.norm(avg_foot_forward_vector)

                if count_body_normals > 0:
                    avg_body_normal_vector /= count_body_normals
                    avg_body_normal_vector[2] = 0.0
                    if np.linalg.norm(avg_body_normal_vector) > 1e-6:
                        avg_body_normal_vector /= np.linalg.norm(avg_body_normal_vector)

                if np.linalg.norm(avg_foot_forward_vector) > 1e-6:
                    forward_vector_figure = avg_foot_forward_vector
                elif np.linalg.norm(avg_body_normal_vector) > 1e-6:
                    forward_vector_figure = np.cross(np.array([0, 0, 1]), avg_body_normal_vector)
                else:
                    forward_vector_figure = np.array([0, 1, 0])

                desired_forward_vector = np.array([0, 1, 0])
                angle_forward = math.atan2(desired_forward_vector[1], desired_forward_vector[0]) - \
                                math.atan2(forward_vector_figure[1], forward_vector_figure[0])

                forward_rotation_matrix = np.array([[np.cos(angle_forward), -np.sin(angle_forward), 0],
                                                    [np.sin(angle_forward), np.cos(angle_forward), 0],
                                                    [0, 0, 1]])

                script_log(f"Debug: Forward Rotation Matrix (Z-axis rotation): \n{forward_rotation_matrix}")

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
                script_log("Forward transformation skipped (ENABLE_forward_transformation is False).")

            script_log("Grounding feet at Z=0...")

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

            # MOVED: Foot flattening is now at the END, after all other transformations

            if ENABLE_DEPTH_ADJUSTMENT:
                script_log("Starting depth adjustment for exaggerated depth estimates...")
                depth_adjuster = DepthAdjuster(landmark_config=landmark_config)
                transformed_frames_data = depth_adjuster.process_frames(transformed_frames_data)
                script_log("Completed depth adjustment")
            else:
                script_log("Depth adjustment skipped (ENABLE_depth_adjustment is False).")

            if head_stabilizer and ENABLE_HEAD_STABILIZATION:
                script_log("Starting head stabilization...")
                stabilized_frames_data = {}
                for frame_num, frame_data in transformed_frames_data.items():
                    stabilized_frame = head_stabilizer.process_head_frame(frame_data)
                    stabilized_frames_data[frame_num] = stabilized_frame
                transformed_frames_data = stabilized_frames_data
                script_log("Completed head stabilization")

            # HEIGHT ADJUSTMENT (with shared motion detector)
            if ENABLE_HIP_HEEL_FLOOR_SHIFT:
                script_log("Applying enhanced HIP_HEEL_FLOOR_SHIFT with state machine...")
                height_adjuster = HeightAdjuster(
                    landmark_config=landmark_config,
                    motion_detector=shared_motion_detector
                )
                transformed_frames_data = height_adjuster.process_frames(transformed_frames_data)
                script_log("Completed HIP_HEEL_FLOOR_SHIFT transformation")
            else:
                script_log("HIP_HEEL_FLOOR_SHIFT skipped (ENABLE_hip_heel_floor_shift is False).")

            script_log("Applying global scaling and centering...")
            min_z = float('inf')
            max_z = float('-inf')
            all_x = []
            all_y = []

            for frame_data in transformed_frames_data.values():
                for coords in frame_data.values():
                    if (isinstance(coords, dict) and
                            'x' in coords and 'y' in coords and 'z' in coords):

                        try:
                            x_val = float(coords['x'])
                            y_val = float(coords['y'])
                            z_val = float(coords['z'])

                            if (np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val)):
                                min_z = min(min_z, z_val)
                                max_z = max(max_z, z_val)
                                all_x.append(x_val)
                                all_y.append(y_val)
                        except (ValueError, TypeError):
                            continue

            if not all_x or min_z == float('inf'):
                script_log("Warning: No valid coordinates found for scaling. Skipping scaling step.")
            else:
                initial_height = max_z - min_z
                desired_height = CONFIG['height_adjustment']['desired_height']
                scale_factor = desired_height / initial_height if initial_height > CONFIG['general']['epsilon'] else 1.0

                center_x = (min(all_x) + max(all_x)) / 2 if all_x else 0.0
                center_y = (min(all_y) + max(all_y)) / 2 if all_y else 0.0

                script_log(
                    f"Debug: Scaling factors - Height: {initial_height:.3f}m, Scale: {scale_factor:.3f}, Center: ({center_x:.3f}, {center_y:.3f})")

                for frame_data in transformed_frames_data.values():
                    for landmark_name, coords in frame_data.items():
                        if (isinstance(coords, dict) and
                                'x' in coords and 'y' in coords and 'z' in coords):

                            try:
                                x_val = float(coords['x'])
                                y_val = float(coords['y'])
                                z_val = float(coords['z'])

                                if (np.isfinite(x_val) and np.isfinite(y_val) and np.isfinite(z_val)):
                                    coords['x'] = x_val * scale_factor
                                    coords['y'] = y_val * scale_factor
                                    coords['z'] = z_val * scale_factor

                                    coords['x'] -= (center_x * scale_factor)
                                    coords['y'] -= (center_y * scale_factor)
                            except (ValueError, TypeError):
                                continue

                script_log(f"Data scaled to a height of {desired_height}m and centered at (0,0).")

            for frame_data in transformed_frames_data.values():
                frame_min_z = float('inf')
                for coords in frame_data.values():
                    if (isinstance(coords, dict) and 'z' in coords):
                        try:
                            z_val = float(coords['z'])
                            if np.isfinite(z_val):
                                frame_min_z = min(frame_min_z, z_val)
                        except (ValueError, TypeError):
                            continue

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

            # BIOMECHANICAL CONSTRAINTS - KEEP THIS IN ORIGINAL POSITION
            if ENABLE_BIOMECHANICAL_CONSTRAINTS:
                script_log("Applying biomechanical constraints...")
                biometrics_frames = {}
                for frame_num, frame_data in transformed_frames_data.items():

                    biometrics_frame = {}

                    left_shoulder = frame_data.get('LEFT_SHOULDER')
                    right_shoulder = frame_data.get('RIGHT_SHOULDER')
                    if left_shoulder and right_shoulder:
                        shoulder_distance = calculate_3d_distance(left_shoulder, right_shoulder)
                        biometrics_frame['shoulder_distance'] = shoulder_distance

                    left_hip = frame_data.get('LEFT_HIP')
                    right_hip = frame_data.get('RIGHT_HIP')
                    if left_hip and right_hip:
                        hip_distance = calculate_3d_distance(left_hip, right_hip)
                        biometrics_frame['hip_distance'] = hip_distance

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
                    "head_roll_limit_degrees": CONFIG['biomechanical_constraints']['head_roll_limit_degrees'],
                    "head_nod_limit_degrees": CONFIG['biomechanical_constraints']['head_nod_limit_degrees'],
                    "head_y_translation_limit_percent_of_height": CONFIG['biomechanical_constraints'][
                        'head_y_translation_limit_percent'],
                    "head_roll_sensitivity": CONFIG['biomechanical_constraints']['head_roll_sensitivity'],
                    "head_nod_sensitivity": CONFIG['biomechanical_constraints']['head_nod_sensitivity']
                }

                with open(output_biometrics_file, 'w') as f:
                    json.dump(biometric_data, f, indent=4)
                script_log(f"Successfully estimated biometrics and saved to '{output_biometrics_file}'.")
            else:
                script_log("Forward biometric constraints skipped (ENABLE_biomechanical_constraints is False).")

            # FOOT FLATTENING (with shared motion detector) - MOVED TO THE VERY END
            if ENABLE_FOOT_FLATTENING:
                script_log("Applying foot flattening corrections as FINAL step...")
                foot_flattener = FootFlattener(
                    landmark_config=landmark_config,
                    motion_detector=shared_motion_detector
                )
                transformed_frames_data = foot_flattener.process_frames(transformed_frames_data)
                script_log("Completed FINAL foot flattening")
            else:
                script_log("Foot flattening skipped (ENABLE_foot_flattening is False).")
        else:
            script_log("Z-up transformation skipped (ENABLE_z_up_transformation is True).")
            transformed_frames_data = all_frames_data

        if transformed_frames_data:
            first_frame_key = list(transformed_frames_data.keys())[0]
            first_frame = transformed_frames_data[first_frame_key]
            script_log(f"Debug: First frame keys: {list(first_frame.keys())}")

            first_landmark_key = list(first_frame.keys())[0]
            first_landmark = first_frame[first_landmark_key]
            script_log(f"Debug: First landmark ({first_landmark_key}): {first_landmark}")
            script_log(f"Debug: First landmark type: {type(first_landmark)}")

            if isinstance(first_landmark, dict):
                script_log(f"Debug: First landmark keys: {list(first_landmark.keys())}")

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