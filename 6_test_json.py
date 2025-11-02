# 6_test_json.py (version 2.0 - Config-driven mocap pipeline analysis)

import os
import sys
import json
import math
import numpy as np
from collections import deque
from enum import Enum, auto
from datetime import datetime

# Add project root to path to import utils
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
    script_log, get_current_show_name, get_current_scene_name,
    get_processing_step_paths, load_landmark_config, get_project_root,
    load_global_config, load_show_config, get_scene_config, get_scene_folder_name,
    get_show_path, get_scene_paths
)


# Motion State Enum and Detector (from 3_ApplyPhysics.py)
class MotionState(Enum):
    STANDING = auto()
    WALKING = auto()
    RUNNING = auto()
    JUMPING = auto()
    LANDING = auto()
    FALLING = auto()
    FEET_OFF_FLOOR = auto()
    TRANSITION = auto()


class MotionPhaseDetector:
    def __init__(self, window_size=5, landmark_config=None):
        self.window_size = window_size
        self.state_history = deque(maxlen=window_size)
        self.current_state = MotionState.STANDING
        self.landmark_config = landmark_config or {}
        self.landmarks = self.landmark_config.get("landmarks", {})
        self.previous_foot_z = None
        self.state_cache = {}

    def lowest_z_of_body_parts(self, frame):
        """Find the lowest body part using landmarks from config"""
        min_z = float('inf')
        for landmark_name in self.landmarks.keys():
            landmark_data = frame.get(landmark_name, {})
            if isinstance(landmark_data, dict) and 'z' in landmark_data:
                z_val = landmark_data['z']
                if z_val is not None and z_val < min_z:
                    min_z = z_val
        return min_z if min_z != float('inf') else 0.0

    def detect_state(self, frame, prev_frame=None):
        """Determine the current motion state based on body positions"""
        frame_hash = self._frame_hash(frame)
        if frame_hash in self.state_cache:
            return self.state_cache[frame_hash]

        # Get relevant landmarks from config
        foot_landmarks = []
        hand_landmarks = []
        head_landmarks = []

        for landmark_name, landmark_data in self.landmarks.items():
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

        # State detection logic (simplified for testing)
        new_state = self._determine_primary_state(
            foot_min_z, hand_min_z, head_min_z, foot_vel, hip_vel, frame
        )

        # State transition smoothing
        self.state_history.append(new_state)

        if len(self.state_history) == self.window_size:
            if all(s == new_state for s in self.state_history):
                if new_state != self.current_state:
                    pass  # Don't log during analysis
                self.current_state = new_state
            else:
                self.current_state = MotionState.TRANSITION
        else:
            self.current_state = new_state

        self.state_cache[frame_hash] = self.current_state

        if len(self.state_cache) > 1000:
            self.state_cache.clear()

        return self.current_state

    def _determine_primary_state(self, foot_min_z, hand_min_z, head_min_z, foot_vel, hip_vel, frame):
        """Determine the primary motion state based on all available data"""
        feet_off_ground_threshold = 0.1
        upward_velocity_threshold = 0.05
        downward_velocity_threshold = -0.05
        running_velocity_threshold = 0.1
        walking_velocity_threshold = 0.02

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
                return MotionState.FALLING

        # Check for walking/running when feet are near ground
        elif 0 < foot_min_z <= feet_off_ground_threshold:
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

        current_hip_mid = (
                                  left_hip_current.get('z', 0) + right_hip_current.get('z', 0)
                          ) / 2

        previous_hip_mid = (
                                   left_hip_prev.get('z', 0) + right_hip_prev.get('z', 0)
                           ) / 2

        return current_hip_mid - previous_hip_mid

    def _frame_hash(self, frame):
        """Create a simple hash of frame data for caching"""
        hash_landmarks = ['LEFT_HEEL', 'RIGHT_HEEL', 'LEFT_HIP', 'RIGHT_HIP', 'HEAD_TOP']
        hash_values = []
        for landmark in hash_landmarks:
            coords = frame.get(landmark, {})
            if isinstance(coords, dict):
                x = round(coords.get('x', 0), 3)
                z = round(coords.get('z', 0), 3)
                hash_values.extend([x, z])
        return tuple(hash_values)


# Configuration Management
def load_test_json_config():
    """
    Load the 6_test_json.py specific configuration
    """
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_script_dir, "6_test_json_config.json")

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            script_log(f"Loaded 6_test_json configuration from: {config_path}")
            return config.get("6_test_json_config", {})
        else:
            script_log(f"WARNING: 6_test_json_config.json not found at {config_path}")
            return create_default_test_json_config()

    except Exception as e:
        script_log(f"ERROR loading 6_test_json configuration: {e}")
        return create_default_test_json_config()


def create_default_test_json_config():
    """
    Create a default configuration if the config file doesn't exist
    """
    return {
        "coordinate_systems": {
            "filter_data": "MEDIAPIPE",
            "apply_physics": "MEDIAPIPE",
            "export_to_blender": "BLENDER"
        },
        "report_settings": {
            "generate_json_structure_reports": True,
            "generate_landmark_quality_reports": True,
            "generate_coordinate_system_reports": True,
            "generate_comprehensive_reports": True,
            "generate_comparison_report": True,
            "generate_config_report": True,
            "concatenate_all_reports": False,
            "master_report_filename": "mocap_analysis_master_report.txt",
            "include_in_master_report": {
                "config_report": True,
                "json_structure_reports": True,
                "landmark_quality_reports": True,
                "coordinate_system_reports": True,
                "comprehensive_reports": True,
                "comparison_report": True
            }
        },
        "analysis_settings": {
            "enable_motion_state_analysis": True,
            "enable_acrobatic_pose_detection": True,
            "enable_orientation_stability_analysis": True
        }
    }


def get_coordinate_system_v2(step_name, test_json_config):
    """
    Determine coordinate system using the config file
    """
    coordinate_systems = test_json_config.get("coordinate_systems", {})

    # Exact match first
    if step_name in coordinate_systems:
        return coordinate_systems[step_name]

    # Then try case-insensitive partial matches
    step_lower = step_name.lower()
    for config_step_name, coord_system in coordinate_systems.items():
        if config_step_name.lower() in step_lower:
            return coord_system

    # Fallback to pattern matching
    if any(pattern in step_lower for pattern in ["step2", "step_2", "filter", "extract"]):
        return "MEDIAPIPE"
    elif any(pattern in step_lower for pattern in ["step3", "step_3", "physics"]):
        return "MEDIAPIPE"
    elif any(pattern in step_lower for pattern in ["step4", "step_4", "final", "export", "blender"]):
        return "BLENDER"

    return "UNKNOWN"


# Utility functions
def calculate_3d_distance(p1, p2):
    """Calculate 3D Euclidean distance between two points"""
    if not (p1 and p2 and
            p1.get('x') is not None and p1.get('y') is not None and p1.get('z') is not None and
            p2.get('x') is not None and p2.get('y') is not None and p2.get('z') is not None):
        return None
    return math.sqrt((p2['x'] - p1['x']) ** 2 +
                     (p2['y'] - p1['y']) ** 2 +
                     (p2['z'] - p1['z']) ** 2)


def detect_hand_floor_contact(frame, coord_system, floor_threshold=0.05):
    """Detect if hands are touching or near the floor"""
    left_hand = frame.get('LEFT_WRIST')
    right_hand = frame.get('RIGHT_WRIST')

    if not left_hand or not right_hand:
        return "UNKNOWN - Missing hand landmarks"

    # Determine which coordinate represents "up" based on coordinate system
    if coord_system == "MEDIAPIPE":  # -Y up
        left_hand_height = -left_hand['y']  # Convert to positive-up
        right_hand_height = -right_hand['y']
    elif coord_system == "BLENDER":  # +Z up
        left_hand_height = left_hand['z']
        right_hand_height = right_hand['z']
    else:
        return "UNKNOWN - Invalid coordinate system"

    # Check if hands are near floor level
    left_touching = left_hand_height <= floor_threshold
    right_touching = right_hand_height <= floor_threshold

    if left_touching and right_touching:
        return "BOTH_HANDS_ON_FLOOR"
    elif left_touching:
        return "LEFT_HAND_ON_FLOOR"
    elif right_touching:
        return "RIGHT_HAND_ON_FLOOR"
    else:
        return "HANDS_OFF_FLOOR"


def detect_acrobatic_poses(frame, coord_system):
    """Detect intentional acrobatic poses with coordinate system awareness"""
    left_hand = frame.get('LEFT_WRIST')
    right_hand = frame.get('RIGHT_WRIST')
    left_foot = frame.get('LEFT_ANKLE') or frame.get('LEFT_HEEL')
    right_foot = frame.get('RIGHT_ANKLE') or frame.get('RIGHT_HEEL')
    head = frame.get('HEAD_TOP') or frame.get('NOSE')

    if not all([left_hand, right_hand, left_foot, right_foot, head]):
        return "UNKNOWN - Missing landmarks"

    # Convert heights based on coordinate system
    if coord_system == "MEDIAPIPE":  # -Y up
        hand_avg = (-left_hand['y'] + -right_hand['y']) / 2
        foot_avg = (-left_foot['y'] + -right_foot['y']) / 2
        head_height = -head['y']
    elif coord_system == "BLENDER":  # +Z up
        hand_avg = (left_hand['z'] + right_hand['z']) / 2
        foot_avg = (left_foot['z'] + right_foot['z']) / 2
        head_height = head['z']
    else:
        return "UNKNOWN - Invalid coordinate system"

    # Also get hand-floor contact info
    hand_contact = detect_hand_floor_contact(frame, coord_system)

    # Detect headstand/handstand patterns
    if hand_avg < foot_avg and hand_avg < head_height:
        # Hands are lowest point
        if "BOTH_HANDS" in hand_contact:
            if abs(hand_avg - head_height) < 0.2:  # Head and hands roughly same height
                return "HANDSTAND"
            else:
                return "ACROBATIC_HANDS_LOW"
        else:
            return "INVERTED_HANDS_LOW"

    elif head_height < hand_avg and head_height < foot_avg:
        # Head is lowest point
        if abs(head_height - hand_avg) < 0.3:  # Head and hands roughly same height
            return "HEADSTAND"
        else:
            return "ACROBATIC_HEAD_LOW"

    elif foot_avg > hand_avg + 0.4 and foot_avg > head_height + 0.4:
        # Feet significantly higher than hands/head
        return "INVERTED_POSE"

    # Check for specific hand-floor interactions
    if "BOTH_HANDS" in hand_contact:
        return "HANDS_ON_FLOOR"
    elif "LEFT_HAND" in hand_contact or "RIGHT_HAND" in hand_contact:
        return "ONE_HAND_ON_FLOOR"

    return "NORMAL"


def is_figure_upside_down(frame, coord_system):
    """Determine if figure is upside down with coordinate system awareness"""
    left_shoulder = frame.get('LEFT_SHOULDER')
    right_shoulder = frame.get('RIGHT_SHOULDER')
    left_hip = frame.get('LEFT_HIP')
    right_hip = frame.get('RIGHT_HIP')

    if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
        return "UNKNOWN - Missing landmarks"

    # Convert heights based on coordinate system
    if coord_system == "MEDIAPIPE":  # -Y up
        shoulder_mid = (-left_shoulder['y'] + -right_shoulder['y']) / 2
        hip_mid = (-left_hip['y'] + -right_hip['y']) / 2
    elif coord_system == "BLENDER":  # +Z up
        shoulder_mid = (left_shoulder['z'] + right_shoulder['z']) / 2
        hip_mid = (left_hip['z'] + right_hip['z']) / 2
    else:
        return "UNKNOWN - Invalid coordinate system"

    # In normal upright pose, shoulders are above hips
    if shoulder_mid > hip_mid:
        return "RIGHT-SIDE UP (shoulders above hips)"
    else:
        return "UPSIDE DOWN (shoulders below hips)"


def analyze_figure_facing(frame, coord_system):
    """Analyze which direction the figure is facing with coordinate awareness"""
    left_hip = frame.get('LEFT_HIP')
    right_hip = frame.get('RIGHT_HIP')
    left_shoulder = frame.get('LEFT_SHOULDER')
    right_shoulder = frame.get('RIGHT_SHOULDER')
    nose = frame.get('NOSE')
    left_ankle = frame.get('LEFT_ANKLE')
    right_ankle = frame.get('RIGHT_ANKLE')
    left_foot_index = frame.get('LEFT_FOOT_INDEX')
    right_foot_index = frame.get('RIGHT_FOOT_INDEX')

    analysis = {}
    analysis['coordinate_system'] = coord_system

    # 1. Hip positions
    if left_hip and right_hip:
        hip_x_diff = right_hip['x'] - left_hip['x']
        hip_y_diff = right_hip['y'] - left_hip['y']
        analysis['hip_x_diff'] = hip_x_diff
        analysis['hip_y_diff'] = hip_y_diff

        if abs(hip_x_diff) > abs(hip_y_diff):
            analysis['hips_facing'] = "SIDEWAYS (X-dominant)"
        else:
            analysis['hips_facing'] = "FORWARD/BACK (Y-dominant)"

    # 2. Shoulder positions
    if left_shoulder and right_shoulder:
        shoulder_x_diff = right_shoulder['x'] - left_shoulder['x']
        shoulder_y_diff = right_shoulder['y'] - left_shoulder['y']
        analysis['shoulder_x_diff'] = shoulder_x_diff
        analysis['shoulder_y_diff'] = shoulder_y_diff

        if abs(shoulder_x_diff) > abs(shoulder_y_diff):
            analysis['shoulders_facing'] = "SIDEWAYS (X-dominant)"
        else:
            analysis['shoulders_facing'] = "FORWARD/BACK (Y-dominant)"

    # 3. Nose position relative to body center
    if left_shoulder and right_shoulder and left_hip and right_hip and nose:
        shoulder_mid_x = (left_shoulder['x'] + right_shoulder['x']) / 2
        shoulder_mid_y = (left_shoulder['y'] + right_shoulder['y']) / 2
        hip_mid_x = (left_hip['x'] + right_hip['x']) / 2
        hip_mid_y = (left_hip['y'] + right_hip['y']) / 2
        body_center_x = (shoulder_mid_x + hip_mid_x) / 2
        body_center_y = (shoulder_mid_y + hip_mid_y) / 2

        nose_offset_x = nose['x'] - body_center_x
        nose_offset_y = nose['y'] - body_center_y

        analysis['nose_offset_x'] = nose_offset_x
        analysis['nose_offset_y'] = nose_offset_y

        # Determine facing direction based on nose position
        if abs(nose_offset_x) > abs(nose_offset_y):
            if nose_offset_x > 0:
                analysis['nose_facing'] = "RIGHT"
            else:
                analysis['nose_facing'] = "LEFT"
        else:
            if nose_offset_y > 0:
                analysis['nose_facing'] = "FORWARD (+Y)"
            else:
                analysis['nose_facing'] = "BACKWARD (-Y)"

    # 4. Foot direction
    if left_ankle and left_foot_index:
        foot_vec_x = left_foot_index['x'] - left_ankle['x']
        foot_vec_y = left_foot_index['y'] - left_ankle['y']
        analysis['left_foot_vec_x'] = foot_vec_x
        analysis['left_foot_vec_y'] = foot_vec_y

        if abs(foot_vec_x) > abs(foot_vec_y):
            analysis['left_foot_facing'] = "SIDEWAYS"
        else:
            if foot_vec_y > 0:
                analysis['left_foot_facing'] = "FORWARD"
            else:
                analysis['left_foot_facing'] = "BACKWARD"

    if right_ankle and right_foot_index:
        foot_vec_x = right_foot_index['x'] - right_ankle['x']
        foot_vec_y = right_foot_index['y'] - right_ankle['y']
        analysis['right_foot_vec_x'] = foot_vec_x
        analysis['right_foot_vec_y'] = foot_vec_y

        if abs(foot_vec_x) > abs(foot_vec_y):
            analysis['right_foot_facing'] = "SIDEWAYS"
        else:
            if foot_vec_y > 0:
                analysis['right_foot_facing'] = "FORWARD"
            else:
                analysis['right_foot_facing'] = "BACKWARD"

    return analysis


def calculate_body_plane(frame):
    """Calculate body plane using hips and shoulders (from 3_ApplyPhysics.py)"""
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
    if normal_length < 1e-6:
        return None
    body_normal = body_normal / normal_length

    return {
        'body_normal': body_normal,
        'shoulder_mid': shoulder_mid
    }


def analyze_head_body_relationship(frame, body_plane):
    """Analyze head position relative to body plane"""
    head_top = frame.get('HEAD_TOP')
    nose = frame.get('NOSE')

    if not all([head_top, nose, body_plane]):
        return None

    ht = np.array([head_top['x'], head_top['y'], head_top['z']])
    ns = np.array([nose['x'], nose['y'], nose['z']])
    body_normal = body_plane['body_normal']
    shoulder_mid = body_plane['shoulder_mid']

    # Calculate dot products with body normal
    head_to_ref_vector = ht - shoulder_mid
    nose_to_ref_vector = ns - shoulder_mid

    head_dot_product = np.dot(head_to_ref_vector, body_normal)
    nose_dot_product = np.dot(nose_to_ref_vector, body_normal)

    # Determine if behind body plane
    is_behind_body_plane = nose_dot_product < 0

    return {
        'head_dot_product': head_dot_product,
        'nose_dot_product': nose_dot_product,
        'is_behind_body_plane': is_behind_body_plane,
        'head_position': "BEHIND body plane" if head_dot_product < 0 else "IN FRONT of body plane",
        'nose_position': "BEHIND body plane" if is_behind_body_plane else "IN FRONT of body plane"
    }


def analyze_orientation_stability(frames, coord_system):
    """
    Analyze orientation stability across multiple frames
    """
    orientation_changes = []
    previous_orientation = None

    for i, frame in enumerate(frames):
        current_orientation = is_figure_upside_down(frame, coord_system)

        if previous_orientation is not None and current_orientation != previous_orientation:
            orientation_changes.append({
                'frame': i,
                'from': previous_orientation,
                'to': current_orientation,
                'type': 'ORIENTATION_CHANGE'
            })

        previous_orientation = current_orientation

    return {
        'total_frames': len(frames),
        'orientation_changes': orientation_changes,
        'stability_score': 1.0 - (len(orientation_changes) / len(frames)) if frames else 1.0,
        'is_stable': len(orientation_changes) <= max(1, len(frames) * 0.1)  # Allow 10% changes
    }


def report_orientation_stability(stability_analysis):
    """Generate a human-readable report from orientation stability analysis"""
    report = []
    report.append("=== ORIENTATION STABILITY ANALYSIS ===")
    report.append(f"Total Frames Analyzed: {stability_analysis['total_frames']}")
    report.append(f"Orientation Changes: {len(stability_analysis['orientation_changes'])}")
    report.append(f"Stability Score: {stability_analysis['stability_score']:.3f}")
    report.append(f"Orientation Stable: {'YES' if stability_analysis['is_stable'] else 'NO'}")

    if stability_analysis['orientation_changes']:
        report.append("\nOrientation Change Events:")
        for change in stability_analysis['orientation_changes']:
            report.append(f"  Frame {change['frame']}: {change['from']} -> {change['to']}")

    return "\n".join(report)


def find_acrobatic_sequences(frames, coord_system, min_duration_frames=5):
    """
    Find sustained acrobatic poses across multiple frames
    """
    acrobatic_sequences = []
    current_sequence = None

    for i, frame in enumerate(frames):
        acrobatic_pose = detect_acrobatic_poses(frame, coord_system)
        is_acrobatic = acrobatic_pose not in ["NORMAL", "UNKNOWN", "HANDS_OFF_FLOOR"]

        if is_acrobatic and current_sequence is None:
            # Start new sequence
            current_sequence = {
                'start_frame': i,
                'end_frame': i,
                'pose_type': acrobatic_pose,
                'frames': [i]
            }
        elif is_acrobatic and current_sequence is not None:
            # Continue current sequence
            current_sequence['end_frame'] = i
            current_sequence['frames'].append(i)
            # Update pose type if it changes (use most frequent)
            if acrobatic_pose != current_sequence['pose_type']:
                current_sequence['pose_type'] = f"MIXED({current_sequence['pose_type']}+{acrobatic_pose})"
        elif not is_acrobatic and current_sequence is not None:
            # End current sequence if it meets minimum duration
            if len(current_sequence['frames']) >= min_duration_frames:
                acrobatic_sequences.append(current_sequence)
            current_sequence = None

    # Don't forget the last sequence
    if current_sequence is not None and len(current_sequence['frames']) >= min_duration_frames:
        acrobatic_sequences.append(current_sequence)

    return acrobatic_sequences


def analyze_motion_states(frames, landmark_config):
    """
    Analyze motion states across all frames using MotionPhaseDetector
    """
    detector = MotionPhaseDetector(landmark_config=landmark_config)
    motion_states = []
    state_transitions = []
    previous_state = None

    for i, frame in enumerate(frames):
        prev_frame = frames[i - 1] if i > 0 else None
        current_state = detector.detect_state(frame, prev_frame)

        motion_states.append({
            'frame': i,
            'state': current_state,
            'state_name': current_state.name
        })

        if previous_state is not None and current_state != previous_state:
            state_transitions.append({
                'frame': i,
                'from': previous_state.name,
                'to': current_state.name
            })

        previous_state = current_state

    # Calculate state statistics
    state_counts = {}
    for state in motion_states:
        state_name = state['state_name']
        state_counts[state_name] = state_counts.get(state_name, 0) + 1

    return {
        'motion_states': motion_states,
        'state_transitions': state_transitions,
        'state_counts': state_counts,
        'total_frames': len(frames),
        'primary_state': max(state_counts.items(), key=lambda x: x[1])[0] if state_counts else 'UNKNOWN'
    }


def analyze_json_file_multi_frame(file_path, step_name, landmark_config, test_json_config):
    """
    Main analysis function that processes entire JSON files frame-by-frame
    """
    script_log(f"Analyzing file: {file_path}")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        script_log(f"ERROR: Could not load JSON file {file_path}: {e}")
        return None

    frames = extract_frames_from_data(data, file_path)

    if not frames:
        return None

    script_log(f"Found {len(frames)} frames to analyze")

    # Check if we have valid frame data
    if len(frames) == 0:
        script_log("WARNING: No frames found in data")
        return None

    # Check first frame structure
    first_frame = frames[0]
    if not isinstance(first_frame, dict):
        script_log(f"WARNING: First frame is not a dictionary (type: {type(first_frame).__name__})")
        return None

    # Determine coordinate system using config
    coord_system = get_coordinate_system_v2(step_name, test_json_config)
    script_log(f"Coordinate system: {coord_system}")

    # Perform all analyses based on config
    analyses = {}
    analysis_settings = test_json_config.get("analysis_settings", {})

    try:
        # 1. Orientation stability
        if analysis_settings.get("enable_orientation_stability_analysis", True):
            script_log("Analyzing orientation stability...")
            analyses['orientation_stability'] = analyze_orientation_stability(frames, coord_system)

        # 2. Acrobatic sequences
        if analysis_settings.get("enable_acrobatic_pose_detection", True):
            script_log("Finding acrobatic sequences...")
            min_duration = test_json_config.get("report_settings", {}).get("min_acrobatic_sequence_duration", 5)
            analyses['acrobatic_sequences'] = find_acrobatic_sequences(frames, coord_system, min_duration)

        # 3. Motion states
        if analysis_settings.get("enable_motion_state_analysis", True):
            script_log("Analyzing motion states...")
            analyses['motion_states'] = analyze_motion_states(frames, landmark_config)

        # 4. Frame-by-frame detailed analysis
        script_log("Performing detailed frame analysis...")
        sample_frames = test_json_config.get("report_settings", {}).get("sample_frames_for_detailed_analysis", 10)
        sample_frames = min(sample_frames, len(frames))
        analyses['frame_samples'] = []

        for i in range(sample_frames):
            frame_analysis = {
                'frame': i,
                'orientation': is_figure_upside_down(frames[i], coord_system),
                'acrobatic_pose': detect_acrobatic_poses(frames[i], coord_system),
                'hand_floor_contact': detect_hand_floor_contact(frames[i], coord_system)
            }

            # Add facing analysis if enabled
            if analysis_settings.get("enable_facing_analysis", True):
                frame_analysis['facing_analysis'] = analyze_figure_facing(frames[i], coord_system)

            # Add body plane analysis if enabled and possible
            if analysis_settings.get("enable_body_plane_analysis", True):
                body_plane = calculate_body_plane(frames[i])
                if body_plane:
                    frame_analysis['head_body_relationship'] = analyze_head_body_relationship(frames[i], body_plane)

            analyses['frame_samples'].append(frame_analysis)

        analyses['metadata'] = {
            'file_path': file_path,
            'step_name': step_name,
            'coord_system': coord_system,
            'total_frames': len(frames),
            'analysis_timestamp': datetime.now().isoformat()
        }

        return analyses

    except Exception as e:
        script_log(f"ERROR during analysis of {file_path}: {e}")
        import traceback
        script_log(f"Analysis traceback: {traceback.format_exc()}")
        return None


def generate_comprehensive_report(analyses, output_file="mocap_json_report.txt"):
    """
    Generate a comprehensive report from all analyses
    """
    report_lines = []

    # Header
    report_lines.append("=" * 80)
    report_lines.append("MOCAP PIPELINE DATA ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"File: {analyses['metadata']['file_path']}")
    report_lines.append(f"Processing Step: {analyses['metadata']['step_name']}")
    report_lines.append(f"Coordinate System: {analyses['metadata']['coord_system']}")
    report_lines.append(f"Total Frames: {analyses['metadata']['total_frames']}")
    report_lines.append("")

    # 1. Orientation Stability Summary
    if 'orientation_stability' in analyses:
        orientation = analyses['orientation_stability']
        report_lines.append(report_orientation_stability(orientation))
        report_lines.append("")

    # 2. Motion State Analysis
    if 'motion_states' in analyses:
        motion = analyses['motion_states']
        report_lines.append("=== MOTION STATE ANALYSIS ===")
        report_lines.append(f"Primary Motion State: {motion['primary_state']}")
        report_lines.append(f"Total State Transitions: {len(motion['state_transitions'])}")
        report_lines.append("")
        report_lines.append("Motion State Distribution:")
        for state, count in sorted(motion['state_counts'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / motion['total_frames']) * 100
            report_lines.append(f"  {state}: {count} frames ({percentage:.1f}%)")
        report_lines.append("")

        if motion['state_transitions']:
            max_transitions = 10  # Default value
            report_lines.append("Significant State Transitions:")
            for transition in motion['state_transitions'][:max_transitions]:
                report_lines.append(f"  Frame {transition['frame']}: {transition['from']} -> {transition['to']}")
            if len(motion['state_transitions']) > max_transitions:
                report_lines.append(f"  ... and {len(motion['state_transitions']) - max_transitions} more transitions")
        report_lines.append("")

    # 3. Acrobatic Sequences
    if 'acrobatic_sequences' in analyses:
        acro_seqs = analyses['acrobatic_sequences']
        report_lines.append("=== ACROBATIC POSE SEQUENCES ===")
        report_lines.append(f"Found {len(acro_seqs)} sustained acrobatic sequences")

        for i, seq in enumerate(acro_seqs):
            duration = seq['end_frame'] - seq['start_frame'] + 1
            report_lines.append(f"Sequence {i + 1}: Frames {seq['start_frame']}-{seq['end_frame']} "
                                f"({duration} frames) - {seq['pose_type']}")
        report_lines.append("")

    # 4. Frame Samples
    if 'frame_samples' in analyses:
        report_lines.append("=== DETAILED FRAME SAMPLES (First 10 frames) ===")
        for sample in analyses['frame_samples']:
            report_lines.append(f"Frame {sample['frame']}:")
            report_lines.append(f"  Orientation: {sample['orientation']}")
            report_lines.append(f"  Acrobatic Pose: {sample['acrobatic_pose']}")
            report_lines.append(f"  Hand-Floor Contact: {sample['hand_floor_contact']}")

            if 'facing_analysis' in sample:
                facing = sample['facing_analysis']
                report_lines.append(f"  Facing Analysis:")
                if 'nose_facing' in facing:
                    report_lines.append(f"    Nose Facing: {facing['nose_facing']}")
                if 'hips_facing' in facing:
                    report_lines.append(f"    Hips Facing: {facing['hips_facing']}")

            if 'head_body_relationship' in sample:
                rel = sample['head_body_relationship']
                report_lines.append(f"  Head Position: {rel['nose_position']}")
        report_lines.append("")

    # 5. Quality Assessment
    report_lines.append("=== DATA QUALITY ASSESSMENT ===")

    # Orientation stability check
    if 'orientation_stability' in analyses:
        orientation = analyses['orientation_stability']
        if orientation['is_stable']:
            report_lines.append("[OK] Orientation: STABLE (minimal flipping)")
        else:
            report_lines.append("[ISSUE] Orientation: UNSTABLE (frequent orientation changes)")

    # Motion state distribution check
    if 'motion_states' in analyses:
        motion = analyses['motion_states']
        primary_percentage = (motion['state_counts'].get(motion['primary_state'], 0) / motion['total_frames']) * 100
        if primary_percentage > 60:
            report_lines.append(
                f"[OK] Motion: CLEAR PRIMARY STATE ({motion['primary_state']} - {primary_percentage:.1f}%)")
        else:
            report_lines.append(
                f"[WARNING] Motion: NO CLEAR DOMINANT STATE (primary: {motion['primary_state']} - {primary_percentage:.1f}%)")

    # Acrobatic content check
    if 'acrobatic_sequences' in analyses:
        acro_seqs = analyses['acrobatic_sequences']
        total_acrobatic_frames = sum(len(seq['frames']) for seq in acro_seqs)
        acro_percentage = (total_acrobatic_frames / analyses['metadata']['total_frames']) * 100
        if acro_percentage > 20:
            report_lines.append(f"[OK] Acrobatic Content: SIGNIFICANT ({acro_percentage:.1f}% of frames)")
        elif acro_percentage > 5:
            report_lines.append(f"[WARNING] Acrobatic Content: MODERATE ({acro_percentage:.1f}% of frames)")
        else:
            report_lines.append(f"[OK] Acrobatic Content: MINIMAL ({acro_percentage:.1f}% of frames)")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Write report to file with UTF-8 encoding
    report_content = "\n".join(report_lines)
    report_path = _get_log_path(output_file)

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        script_log(f"Report written to: {report_path}")
    except Exception as e:
        script_log(f"ERROR: Could not write report to {report_path}: {e}")
        # Fallback: print to console
        print(report_content)

    return report_content


def create_minimal_landmark_config():
    """
    Create a minimal landmark configuration as last resort fallback
    """
    script_log("WARNING: Creating minimal landmark configuration")

    minimal_config = {
        "landmarks": {
            "NOSE": {"index": 0, "motion_detection": True},
            "HEAD_TOP": {"index": 10, "motion_detection": True},
            "LEFT_SHOULDER": {"index": 11, "motion_detection": True},
            "RIGHT_SHOULDER": {"index": 12, "motion_detection": True},
            "LEFT_WRIST": {"index": 15, "motion_detection": True},
            "RIGHT_WRIST": {"index": 16, "motion_detection": True},
            "LEFT_HIP": {"index": 23, "motion_detection": True},
            "RIGHT_HIP": {"index": 24, "motion_detection": True},
            "LEFT_ANKLE": {"index": 27, "motion_detection": True},
            "RIGHT_ANKLE": {"index": 28, "motion_detection": True},
            "LEFT_HEEL": {"index": 29, "motion_detection": True},
            "RIGHT_HEEL": {"index": 30, "motion_detection": True},
            "LEFT_FOOT_INDEX": {"index": 31, "motion_detection": True},
            "RIGHT_FOOT_INDEX": {"index": 32, "motion_detection": True}
        },
        "motion_detection_settings": {
            "window_size": 5,
            "feet_off_ground_threshold": 0.1,
            "upward_velocity_threshold": 0.05,
            "downward_velocity_threshold": -0.05,
            "running_velocity_threshold": 0.1,
            "walking_velocity_threshold": 0.02
        }
    }

    return minimal_config


def load_landmark_config_with_fallback(config_path):
    """
    Load landmark config with fallback to the existing pipeline landmarks file
    """
    try:
        return load_landmark_config(config_path)
    except Exception as e:
        script_log(f"WARNING: Could not load landmark config from {config_path}: {e}")

        # Try to load from the pipeline landmarks file in the same directory
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_landmarks_path = os.path.join(current_script_dir, "0_RunMocapAnimPipeline_LANDMARKS.json")

        if os.path.exists(pipeline_landmarks_path):
            script_log(f"Loading landmarks from pipeline config: {pipeline_landmarks_path}")
            return load_landmark_config(pipeline_landmarks_path)
        else:
            script_log(f"WARNING: Pipeline landmarks file not found at {pipeline_landmarks_path}")
            script_log("Using minimal default landmark configuration")
            return create_minimal_landmark_config()


def generate_config_report(show_name, scene_name, landmark_config):
    """
    Generate a report showing all configuration files and paths being used
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CONFIGURATION AND PATHS REPORT")
    report_lines.append("=" * 80)

    try:
        # Project structure
        project_root = get_project_root()
        report_lines.append(f"Project Root: {project_root}")
        report_lines.append("")

        # Landmark config source
        report_lines.append("=== LANDMARK CONFIGURATION ===")
        landmark_config_path = os.path.join(project_root, "config", "landmark_config.json")
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_landmarks_path = os.path.join(current_script_dir, "0_RunMocapAnimPipeline_LANDMARKS.json")

        if os.path.exists(landmark_config_path):
            report_lines.append(f"Primary Source: {landmark_config_path} [EXISTS]")
        elif os.path.exists(pipeline_landmarks_path):
            report_lines.append(f"Fallback Source: {pipeline_landmarks_path} [EXISTS]")
        else:
            report_lines.append("Source: MINIMAL DEFAULT CONFIGURATION (no files found)")

        report_lines.append(f"Landmarks Configured: {len(landmark_config.get('landmarks', {}))}")

        # List motion detection landmarks
        motion_landmarks = []
        for name, data in landmark_config.get('landmarks', {}).items():
            if data.get('motion_detection', False):
                motion_landmarks.append(name)
        report_lines.append(f"Motion Detection Landmarks: {', '.join(motion_landmarks)}")
        report_lines.append("")

        # Global config
        global_config = load_global_config()
        report_lines.append("=== GLOBAL CONFIG ===")
        report_lines.append(f"File: {os.path.join(project_root, 'config', 'global_config.json')}")
        report_lines.append(f"Blender Path: {global_config.get('blender_exe_path', 'NOT SET')}")
        report_lines.append(f"Shows Root: {global_config.get('shows_root_dir', 'NOT SET')}")
        report_lines.append("")

        # Show config
        show_config = load_show_config(show_name)
        show_path = get_show_path(show_name)
        report_lines.append("=== SHOW CONFIG ===")
        report_lines.append(f"Show: {show_name}")
        report_lines.append(f"Show Path: {show_path}")
        report_lines.append(f"Config File: {os.path.join(show_path, 'config', 'show_config.json')}")
        report_lines.append(f"Inputs Root: {show_config['paths'].get('inputs_root', 'DEFAULT')}")
        report_lines.append(f"Outputs Root: {show_config['paths'].get('outputs_root', 'DEFAULT')}")
        report_lines.append("")

        # Scene config
        scene_config = get_scene_config(show_name, scene_name)
        scene_folder_name = get_scene_folder_name(show_name, scene_name)

        # Build the scene directory path properly
        scene_dir = os.path.join(show_path, "inputs", "scenes", scene_folder_name)
        scene_config_path = os.path.join(scene_dir, "scene-config.json")

        report_lines.append("=== SCENE CONFIG ===")
        report_lines.append(f"Scene: {scene_name}")
        report_lines.append(f"Scene Folder: {scene_folder_name}")
        report_lines.append(f"Scene Directory: {scene_dir} {'[EXISTS]' if os.path.exists(scene_dir) else '[MISSING]'}")
        report_lines.append(
            f"Scene Config: {scene_config_path} {'[EXISTS]' if os.path.exists(scene_config_path) else '[MISSING]'}")

        # FIXED: Scene paths - only check actual file paths, not folder names
        scene_files_to_check = {
            "input_video": os.path.join(scene_dir, "step_1_input.mp4"),
            "output_pose_data": os.path.join(scene_dir, "step_2_input.json"),
            "input_blender_rig": os.path.join(scene_dir, "rig.blend"),
            "step_3_input": os.path.join(scene_dir, "step_3_input.json"),
            "step_4_input": os.path.join(scene_dir, "step_4_input.json"),
            "final_animation_data": os.path.join(scene_dir, "final_animation_data.json")
        }

        report_lines.append("Scene Files:")
        for key, file_path in scene_files_to_check.items():
            exists = "[EXISTS]" if os.path.exists(file_path) else "[MISSING]"
            report_lines.append(f"  {exists} {key}: {file_path}")
        report_lines.append("")

        # Processing steps
        report_lines.append("=== PROCESSING STEPS ===")
        processing_steps = scene_config.get("processing_steps", {})
        for step_name, step_config in processing_steps.items():
            step_paths = get_processing_step_paths(show_name, scene_name, step_name)
            report_lines.append(f"Step: {step_name}")
            report_lines.append(f"  Input: {step_paths.get('input_file', 'NOT FOUND')}")
            report_lines.append(f"  Output: {step_paths.get('output_file', 'NOT FOUND')}")

            # Check file existence
            input_exists = os.path.exists(step_paths.get('input_file', ''))
            output_exists = os.path.exists(step_paths.get('output_file', ''))
            report_lines.append(
                f"  Status: Input {'[EXISTS]' if input_exists else '[MISSING]'}, Output {'[EXISTS]' if output_exists else '[MISSING]'}")
        report_lines.append("")

    except Exception as e:
        report_lines.append(f"ERROR generating config report: {e}")

    return "\n".join(report_lines)


def generate_json_structure_report(file_path, step_name, sample_size=3):
    """
    Analyze and report on the JSON file structure with meaningful information
    Handles video files and other non-JSON files appropriately
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"FILE STRUCTURE REPORT: {step_name}")
    report_lines.append("=" * 80)
    report_lines.append(f"File: {file_path}")

    # Check file type first
    filename_lower = file_path.lower()

    if filename_lower.endswith('.mp4') or filename_lower.endswith('.avi') or filename_lower.endswith('.mov'):
        # Video file - handle separately
        report_lines.append("File Type: VIDEO")
        report_lines.append("This is a video file, not JSON data.")

        try:
            file_size = os.path.getsize(file_path)
            report_lines.append(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            report_lines.append("Video files contain binary data and cannot be analyzed as JSON.")
        except Exception as e:
            report_lines.append(f"Error getting file info: {e}")

        return "\n".join(report_lines)

    elif filename_lower.endswith('.blend'):
        # Blender file
        report_lines.append("File Type: BLENDER FILE")
        report_lines.append("This is a Blender binary file, not JSON data.")

        try:
            file_size = os.path.getsize(file_path)
            report_lines.append(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
            report_lines.append("Blender files contain binary data and cannot be analyzed as JSON.")
        except Exception as e:
            report_lines.append(f"Error getting file info: {e}")

        return "\n".join(report_lines)

    # For JSON files, proceed with the analysis
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        report_lines.append("File Type: JSON")
        report_lines.append(f"JSON Structure: {type(data).__name__}")

        if isinstance(data, list):
            report_lines.append(f"Structure: List with {len(data)} frames")
            if data:
                report_lines.append(f"First frame type: {type(data[0]).__name__}")
                if isinstance(data[0], dict):
                    report_lines.append(f"Landmarks in first frame: {len(data[0])}")
                    # Show first few landmarks
                    landmarks = list(data[0].keys())[:10]
                    report_lines.append(f"Sample landmarks: {', '.join(landmarks)}" +
                                        ("..." if len(data[0]) > 10 else ""))

        elif isinstance(data, dict):
            # Check if it's a frame-based structure (numeric keys)
            numeric_keys = []
            string_keys = []
            for key in data.keys():
                try:
                    int(key)
                    numeric_keys.append(key)
                except (ValueError, TypeError):
                    string_keys.append(key)

            if numeric_keys and len(numeric_keys) > 0:
                # This is a frame-based structure
                sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
                total_frames = len(numeric_keys)
                report_lines.append(f"Structure: Frame-based dictionary with {total_frames} frames")

                # Analyze first frame
                first_frame_key = sorted_keys[0]
                first_frame = data[first_frame_key]
                report_lines.append(f"Frame {first_frame_key} landmarks: {len(first_frame)}")

                # Show landmark statistics
                landmark_names = list(first_frame.keys())
                report_lines.append(f"All landmarks: {', '.join(sorted(landmark_names))}")

                # Show sample data from first few frames
                report_lines.append("")
                report_lines.append("=== SAMPLE FRAME DATA ===")

                for i in range(min(sample_size, len(sorted_keys))):
                    frame_key = sorted_keys[i]
                    frame = data[frame_key]
                    report_lines.append(f"Frame {frame_key}:")

                    # Show a few key landmarks with coordinates
                    key_landmarks = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP']
                    for landmark in key_landmarks:
                        if landmark in frame:
                            coords = frame[landmark]
                            if isinstance(coords, dict):
                                report_lines.append(f"  {landmark}: "
                                                    f"x={coords.get('x', 'N/A'):.3f}, "
                                                    f"y={coords.get('y', 'N/A'):.3f}, "
                                                    f"z={coords.get('z', 'N/A'):.3f}")

                    if i < sample_size - 1:  # Add spacing between frames, but not after last one
                        report_lines.append("")

                # Data statistics
                report_lines.append("")
                report_lines.append("=== DATA STATISTICS ===")

                # Check coordinate completeness
                complete_frames = 0
                frames_with_missing_coords = 0

                sample_frames = [data[key] for key in sorted_keys[:min(10, len(sorted_keys))]]
                for frame in sample_frames:
                    frame_complete = True
                    for landmark, coords in frame.items():
                        if isinstance(coords, dict):
                            if not all(coord in coords for coord in ['x', 'y', 'z']):
                                frame_complete = False
                                frames_with_missing_coords += 1
                                break
                    if frame_complete:
                        complete_frames += 1

                report_lines.append(f"Frames with complete 3D coordinates: {complete_frames}/{len(sample_frames)}")
                if frames_with_missing_coords > 0:
                    report_lines.append(
                        f"Frames with missing coordinates: {frames_with_missing_coords}/{len(sample_frames)}")

                # Coordinate ranges
                coord_ranges = {'x': {'min': float('inf'), 'max': float('-inf')},
                                'y': {'min': float('inf'), 'max': float('-inf')},
                                'z': {'min': float('inf'), 'max': float('-inf')}}

                for frame in sample_frames:
                    for landmark, coords in frame.items():
                        if isinstance(coords, dict):
                            for coord in ['x', 'y', 'z']:
                                if coord in coords and coords[coord] is not None:
                                    coord_ranges[coord]['min'] = min(coord_ranges[coord]['min'], coords[coord])
                                    coord_ranges[coord]['max'] = max(coord_ranges[coord]['max'], coords[coord])

                report_lines.append("Coordinate ranges in sample:")
                for coord in ['x', 'y', 'z']:
                    if coord_ranges[coord]['min'] != float('inf'):
                        report_lines.append(
                            f"  {coord}: [{coord_ranges[coord]['min']:.3f}, {coord_ranges[coord]['max']:.3f}]")
                    else:
                        report_lines.append(f"  {coord}: NO DATA")

            else:
                # Regular dictionary structure (non-frame-based)
                report_lines.append(f"Structure: Dictionary with {len(data)} keys")
                if string_keys:
                    report_lines.append("Key types:")
                    report_lines.append(f"  String keys: {len(string_keys)}")
                    if string_keys:
                        report_lines.append(f"  Sample string keys: {', '.join(string_keys[:10])}" +
                                            ("..." if len(string_keys) > 10 else ""))

        # File size info
        file_size = os.path.getsize(file_path)
        report_lines.append("")
        report_lines.append(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")

    except UnicodeDecodeError:
        report_lines.append("File Type: BINARY (Non-JSON)")
        report_lines.append("This file appears to be binary data, not JSON.")
        try:
            file_size = os.path.getsize(file_path)
            report_lines.append(f"File size: {file_size} bytes ({file_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            report_lines.append(f"Error getting file info: {e}")

    except Exception as e:
        report_lines.append(f"ERROR analyzing file structure: {e}")

    return "\n".join(report_lines)


def generate_landmark_quality_report(frames, coord_system, landmark_config):
    """
    Analyze data quality for each landmark across frames
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LANDMARK DATA QUALITY REPORT")
    report_lines.append("=" * 80)

    if not frames:
        report_lines.append("No frames to analyze")
        return "\n".join(report_lines)

    # Analyze landmark presence and data quality
    landmark_stats = {}
    total_frames = len(frames)

    for frame in frames:
        if not isinstance(frame, dict):
            continue

        for landmark, coords in frame.items():
            if landmark not in landmark_stats:
                landmark_stats[landmark] = {
                    'present_count': 0,
                    'complete_count': 0,
                    'has_x': 0, 'has_y': 0, 'has_z': 0,
                    'x_values': [], 'y_values': [], 'z_values': []
                }

            landmark_stats[landmark]['present_count'] += 1

            if isinstance(coords, dict):
                has_x = 'x' in coords and coords['x'] is not None
                has_y = 'y' in coords and coords['y'] is not None
                has_z = 'z' in coords and coords['z'] is not None

                if has_x:
                    landmark_stats[landmark]['has_x'] += 1
                    landmark_stats[landmark]['x_values'].append(coords['x'])
                if has_y:
                    landmark_stats[landmark]['has_y'] += 1
                    landmark_stats[landmark]['y_values'].append(coords['y'])
                if has_z:
                    landmark_stats[landmark]['has_z'] += 1
                    landmark_stats[landmark]['z_values'].append(coords['z'])

                if has_x and has_y and has_z:
                    landmark_stats[landmark]['complete_count'] += 1

    # Generate report
    report_lines.append(f"Analyzed {total_frames} frames, {len(landmark_stats)} unique landmarks")
    report_lines.append("")
    report_lines.append("=== LANDMARK QUALITY SUMMARY ===")

    for landmark, stats in sorted(landmark_stats.items()):
        presence_pct = (stats['present_count'] / total_frames) * 100
        completeness_pct = (stats['complete_count'] / total_frames) * 100

        report_lines.append(f"\n{landmark}:")
        report_lines.append(f"  Presence: {stats['present_count']}/{total_frames} ({presence_pct:.1f}%)")
        report_lines.append(f"  Complete 3D: {stats['complete_count']}/{total_frames} ({completeness_pct:.1f}%)")
        report_lines.append(f"  Coordinate coverage: "
                            f"X({stats['has_x']}/{total_frames}), "
                            f"Y({stats['has_y']}/{total_frames}), "
                            f"Z({stats['has_z']}/{total_frames})")

        # Calculate variability for complete landmarks
        if stats['complete_count'] > 0:
            if stats['x_values']:
                x_range = max(stats['x_values']) - min(stats['x_values'])
                report_lines.append(f"  X range: {x_range:.3f}")
            if stats['y_values']:
                y_range = max(stats['y_values']) - min(stats['y_values'])
                report_lines.append(f"  Y range: {y_range:.3f}")
            if stats['z_values']:
                z_range = max(stats['z_values']) - min(stats['z_values'])
                report_lines.append(f"  Z range: {z_range:.3f}")

    # Identify problematic landmarks
    report_lines.append("")
    report_lines.append("=== PROBLEMATIC LANDMARKS ===")
    problematic = []

    for landmark, stats in landmark_stats.items():
        presence_pct = (stats['present_count'] / total_frames) * 100
        completeness_pct = (stats['complete_count'] / total_frames) * 100

        if presence_pct < 80:
            problematic.append(f"{landmark}: Low presence ({presence_pct:.1f}%)")
        elif completeness_pct < 80:
            problematic.append(f"{landmark}: Incomplete data ({completeness_pct:.1f}%)")

    if problematic:
        for issue in problematic:
            report_lines.append(f"  {issue}")
    else:
        report_lines.append("  No major issues detected")

    return "\n".join(report_lines)


def generate_coordinate_system_analysis(frames, step_name, test_json_config):
    """
    Analyze coordinate system characteristics and detect potential issues
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("COORDINATE SYSTEM ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append(f"Processing Step: {step_name}")

    # Use config-based coordinate system detection
    expected_system = get_coordinate_system_v2(step_name, test_json_config)
    report_lines.append(f"Expected System: {expected_system}")
    report_lines.append("")

    if not frames or len(frames) == 0:
        report_lines.append("No frames to analyze")
        return "\n".join(report_lines)

    # Analyze a few sample frames to determine coordinate system characteristics
    sample_frames = frames[:min(5, len(frames))]

    ground_level_analysis = []
    vertical_analysis = []

    for i, frame in enumerate(sample_frames):
        if not isinstance(frame, dict):
            continue

        # Analyze foot positions for ground level
        foot_landmarks = ['LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL']
        foot_heights = []

        for landmark in foot_landmarks:
            if landmark in frame and isinstance(frame[landmark], dict):
                coords = frame[landmark]
                # Try to determine which coordinate represents height
                if 'y' in coords:  # MediaPipe: -Y up
                    foot_heights.append(-coords['y'])
                elif 'z' in coords:  # Blender: Z up
                    foot_heights.append(coords['z'])

        if foot_heights:
            avg_foot_height = sum(foot_heights) / len(foot_heights)
            ground_level_analysis.append(f"Frame {i}: Avg foot height = {avg_foot_height:.3f}")

        # Analyze vertical distribution
        all_heights = []
        for landmark, coords in frame.items():
            if isinstance(coords, dict):
                if 'y' in coords:
                    all_heights.append(-coords['y'])  # Convert to positive-up
                elif 'z' in coords:
                    all_heights.append(coords['z'])

        if all_heights:
            height_range = max(all_heights) - min(all_heights)
            vertical_analysis.append(f"Frame {i}: Height range = {height_range:.3f}")

    report_lines.append("=== GROUND LEVEL ANALYSIS ===")
    if ground_level_analysis:
        for analysis in ground_level_analysis:
            report_lines.append(f"  {analysis}")
    else:
        report_lines.append("  Could not determine ground level")

    report_lines.append("")
    report_lines.append("=== VERTICAL DISTRIBUTION ===")
    if vertical_analysis:
        for analysis in vertical_analysis:
            report_lines.append(f"  {analysis}")
    else:
        report_lines.append("  Could not analyze vertical distribution")

    # Detect potential coordinate system issues
    report_lines.append("")
    report_lines.append("=== COORDINATE SYSTEM ASSESSMENT ===")

    if expected_system == "MEDIAPIPE":
        report_lines.append("Expected: MEDIAPIPE (-Y up)")
        report_lines.append("Check: Negative Y values should increase downward")
    elif expected_system == "BLENDER":
        report_lines.append("Expected: BLENDER (+Z up)")
        report_lines.append("Check: Positive Z values should increase upward")
    else:
        report_lines.append("Expected: UNKNOWN - Manual inspection required")

    return "\n".join(report_lines)


def generate_comparison_report(all_analyses):
    """
    Generate a comparison report across all processing steps
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MOCAP PIPELINE COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    for step_name, analysis in all_analyses.items():
        report_lines.append(f"--- {step_name} ---")

        orientation = analysis.get('orientation_stability', {})
        motion = analysis.get('motion_states', {})
        acro_seqs = analysis.get('acrobatic_sequences', [])

        if orientation:
            report_lines.append(f"Orientation Stability: {orientation.get('stability_score', 0):.3f}")
        if motion:
            report_lines.append(f"Primary Motion State: {motion.get('primary_state', 'UNKNOWN')}")
            report_lines.append(f"State Transitions: {len(motion.get('state_transitions', []))}")
        report_lines.append(f"Acrobatic Sequences: {len(acro_seqs)}")
        report_lines.append("")

    report_content = "\n".join(report_lines)
    report_path = _get_log_path("mocap_pipeline_comparison.txt")

    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        script_log(f"Comparison report written to: {report_path}")
    except Exception as e:
        script_log(f"ERROR: Could not write comparison report: {e}")


def diagnose_scene_files(show_name, scene_name):
    """
    Diagnostic function to see what files actually exist in the scene directory
    """
    script_log("=== SCENE FILE DIAGNOSIS ===")

    try:
        # Get the scene directory path
        project_root = get_project_root()
        scene_folder_name = get_scene_folder_name(show_name, scene_name)
        scene_dir = os.path.join(project_root, "shows", show_name, "inputs", "scenes", scene_folder_name)

        script_log(f"Scene directory: {scene_dir}")

        if not os.path.exists(scene_dir):
            script_log(f"ERROR: Scene directory does not exist: {scene_dir}")
            return

        # List all files in the scene directory
        all_files = os.listdir(scene_dir)
        json_files = [f for f in all_files if f.endswith('.json')]

        script_log(f"All JSON files in scene directory ({len(json_files)} files):")
        for json_file in sorted(json_files):
            full_path = os.path.join(scene_dir, json_file)
            file_size = os.path.getsize(full_path) if os.path.exists(full_path) else 0
            script_log(f"  {json_file} ({file_size} bytes)")

        # Check for common naming patterns
        common_patterns = [
            "step_2", "step2", "filtered", "physics", "final",
            "step_3", "step3", "step_4", "step4"
        ]

        script_log("Files matching common patterns:")
        for pattern in common_patterns:
            matches = [f for f in json_files if pattern.lower() in f.lower()]
            for match in matches:
                script_log(f"  {pattern}: {match}")

        # Check scene config for processing steps
        scene_config = get_scene_config(show_name, scene_name)
        processing_steps = scene_config.get("processing_steps", {})

        script_log("Processing steps from scene config:")
        for step_name, step_config in processing_steps.items():
            script_log(f"  {step_name}:")
            script_log(f"    input_file: {step_config.get('input_file', 'NOT SET')}")
            script_log(f"    output_file: {step_config.get('output_file', 'NOT SET')}")

    except Exception as e:
        script_log(f"ERROR during scene file diagnosis: {e}")


def extract_frames_from_data(data, file_path):
    """
    Extract frames from various JSON data structures
    """
    frames = None

    if isinstance(data, list):
        frames = data
        script_log(f"Found {len(frames)} frames in list structure")
    elif isinstance(data, dict) and 'frames' in data:
        frames = data['frames']
        script_log(f"Found {len(frames)} frames in 'frames' key")
    elif isinstance(data, dict):
        # Try to find any list value that might contain frames
        for key, value in data.items():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    frames = value
                    script_log(f"Found {len(frames)} frames in key: {key}")
                    break
                elif isinstance(value[0], (int, float)):
                    script_log(f"Found numeric list in key: {key} - not frame data")

        # NEW: Handle the case where dictionary keys are numeric frame indices
        if not frames:
            # Check if keys are numeric (frame indices)
            numeric_keys = []
            for key in data.keys():
                try:
                    int(key)
                    numeric_keys.append(key)
                except (ValueError, TypeError):
                    pass

            if numeric_keys and len(numeric_keys) > 0:
                # Sort numeric keys to maintain frame order
                sorted_keys = sorted(numeric_keys, key=lambda x: int(x))
                frames = []
                for key in sorted_keys:
                    frame_data = data[key]
                    if isinstance(frame_data, dict):
                        frames.append(frame_data)

                if frames:
                    script_log(f"Found {len(frames)} frames in numeric key structure")
                    # Add frame numbers to each frame for tracking
                    for i, frame in enumerate(frames):
                        frame['_frame_number'] = i

    if not frames:
        script_log(f"WARNING: Could not find frame data in {file_path}")
        script_log(f"JSON structure: {type(data).__name__}")
        if isinstance(data, dict):
            script_log(f"Top-level keys sample: {list(data.keys())[:10]}...")  # Show first 10 keys only

    return frames


def find_alternative_json_file(show_name, scene_name, step_name):
    """
    Try to find JSON files using alternative naming patterns
    """
    try:
        project_root = get_project_root()
        scene_folder_name = get_scene_folder_name(show_name, scene_name)
        scene_dir = os.path.join(project_root, "shows", show_name, "inputs", "scenes", scene_folder_name)

        if not os.path.exists(scene_dir):
            return None

        all_files = os.listdir(scene_dir)
        json_files = [f for f in all_files if f.endswith('.json')]

        # Map step names to search patterns
        step_patterns = {
            "Step 2 - Filtered Pose Data": ["step2", "step_2", "filtered", "processed"],
            "Step 3 - Physics Applied": ["step3", "step_3", "physics", "applied"],
            "Step 4 - Final Pose Data": ["step4", "step_4", "final", "output"],
            "Step2": ["step2", "step_2", "filtered"],
            "Step3": ["step3", "step_3", "physics"],
            "Step4": ["step4", "step_4", "final"],
            "Filtered Pose Data": ["filtered", "step2", "step_2"],
            "Physics Applied": ["physics", "step3", "step_3"],
            "Final Pose Data": ["final", "step4", "step_4"]
        }

        patterns = step_patterns.get(step_name, [step_name.lower().replace(' ', '')])

        for pattern in patterns:
            for json_file in json_files:
                if pattern.lower() in json_file.lower():
                    full_path = os.path.join(scene_dir, json_file)
                    # Check if file has reasonable size (not empty)
                    if os.path.getsize(full_path) > 100:  # At least 100 bytes
                        return full_path

        return None

    except Exception as e:
        script_log(f"ERROR in find_alternative_json_file: {e}")
        return None


def find_any_json_files(show_name, scene_name):
    """
    Find any JSON files in the scene directory that might contain pose data
    """
    try:
        project_root = get_project_root()
        scene_folder_name = get_scene_folder_name(show_name, scene_name)
        scene_dir = os.path.join(project_root, "shows", show_name, "inputs", "scenes", scene_folder_name)

        if not os.path.exists(scene_dir):
            return []

        all_files = os.listdir(scene_dir)
        json_files = [os.path.join(scene_dir, f) for f in all_files if f.endswith('.json')]

        # Filter out very small files and config files
        valid_files = []
        for json_file in json_files:
            file_size = os.path.getsize(json_file)
            filename = os.path.basename(json_file).lower()

            # Skip config files and very small files
            if (file_size > 1000 and  # At least 1KB
                    'config' not in filename and
                    'landmark' not in filename):
                valid_files.append(json_file)

        return valid_files

    except Exception as e:
        script_log(f"ERROR in find_any_json_files: {e}")
        return []


def _get_log_path(log_file_name):
    """
    Calculates the absolute path for the log file in the project logs folder.
    Logs will always go to: [project_root]/logs/[log_file_name]
    """
    try:
        # Get the project root (where utils.py lives)
        project_root = get_project_root()

        # Create logs directory if it doesn't exist
        logs_dir = os.path.join(project_root, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        return os.path.join(logs_dir, log_file_name)

    except Exception as e:
        # Fallback to current working directory if everything fails
        fallback_path = os.path.join(os.getcwd(), log_file_name)
        print(f"WARNING: Could not determine project logs directory. Using fallback: {fallback_path}")
        return fallback_path


def diagnose_project_structure():
    """
    Diagnostic function to show the complete project structure
    """
    script_log("=== PROJECT STRUCTURE DIAGNOSIS ===")

    try:
        project_root = get_project_root()
        script_log(f"Project Root: {project_root}")

        # Key directories to check
        key_dirs = [
            project_root,
            os.path.join(project_root, "config"),
            os.path.join(project_root, "logs"),
            os.path.join(project_root, "shows"),
            os.path.join(project_root, "shows", "aae"),
            os.path.join(project_root, "shows", "aae", "inputs"),
            os.path.join(project_root, "shows", "aae", "inputs", "scenes"),
            os.path.join(project_root, "shows", "aae", "inputs", "scenes", "10_tavern_scene"),
            os.path.join(project_root, "shows", "aae", "config"),
        ]

        script_log("Key directory status:")
        for dir_path in key_dirs:
            exists = os.path.exists(dir_path)
            status = "EXISTS" if exists else "MISSING"
            script_log(f"  {status}: {dir_path}")

            if exists:
                # Show contents for critical directories
                if "10_tavern_scene" in dir_path or "config" in dir_path:
                    try:
                        items = os.listdir(dir_path)
                        script_log(f"    Contents: {', '.join(items[:10])}" + ("..." if len(items) > 10 else ""))
                    except Exception as e:
                        script_log(f"    Could not list contents: {e}")

        # Check specific critical files
        critical_files = [
            os.path.join(project_root, "config", "global_config.json"),
            os.path.join(project_root, "shows", "aae", "config", "show_config.json"),
            os.path.join(project_root, "shows", "aae", "inputs", "scenes", "10_tavern_scene", "scene-config.json"),
        ]

        script_log("Critical file status:")
        for file_path in critical_files:
            exists = os.path.exists(file_path)
            status = "EXISTS" if exists else "MISSING"
            script_log(f"  {status}: {file_path}")

    except Exception as e:
        script_log(f"ERROR during project structure diagnosis: {e}")


def diagnose_processing_step_paths(show_name, scene_name):
    """
    Diagnostic function to debug processing step path resolution
    """
    script_log("=== PROCESSING STEP PATHS DIAGNOSIS ===")

    try:
        scene_config = get_scene_config(show_name, scene_name)
        processing_steps = scene_config.get("processing_steps", {})

        for step_name, step_config in processing_steps.items():
            script_log(f"Step: {step_name}")
            script_log(f"  Config input_file: {step_config.get('input_file', 'NOT SET')}")
            script_log(f"  Config output_file: {step_config.get('output_file', 'NOT SET')}")

            step_paths = get_processing_step_paths(show_name, scene_name, step_name)
            script_log(f"  Resolved input_file: {step_paths.get('input_file', 'NOT FOUND')}")
            script_log(f"  Resolved output_file: {step_paths.get('output_file', 'NOT FOUND')}")

            input_exists = os.path.exists(step_paths.get('input_file', ''))
            output_exists = os.path.exists(step_paths.get('output_file', ''))
            script_log(f"  Exists: Input {'YES' if input_exists else 'NO'}, Output {'YES' if output_exists else 'NO'}")

    except Exception as e:
        script_log(f"ERROR in processing step paths diagnosis: {e}")


def concatenate_all_reports(test_json_config):
    """
    Concatenate all generated reports into a single master report file
    """
    try:
        report_settings = test_json_config.get("report_settings", {})

        if not report_settings.get("concatenate_all_reports", False):
            script_log("Report concatenation is disabled in config")
            return

        master_filename = report_settings.get("master_report_filename", "mocap_analysis_master_report.txt")
        master_path = _get_log_path(master_filename)

        include_settings = report_settings.get("include_in_master_report", {})

        script_log(f"Concatenating reports into master file: {master_path}")

        master_content = []
        master_content.append("=" * 80)
        master_content.append("MOCAP ANALYSIS MASTER REPORT")
        master_content.append("=" * 80)
        master_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        master_content.append("")

        # 1. Config Report
        if include_settings.get("config_report", True):
            config_report_path = _get_log_path("mocap_config_report.txt")
            if os.path.exists(config_report_path):
                master_content.append("=== CONFIGURATION REPORT ===")
                master_content.append("")
                with open(config_report_path, 'r', encoding='utf-8') as f:
                    master_content.append(f.read())
                master_content.append("")
                master_content.append("")
            else:
                master_content.append("=== CONFIGURATION REPORT ===")
                master_content.append("Config report not found")
                master_content.append("")

        # 2. JSON Structure Reports
        if include_settings.get("json_structure_reports", True):
            json_structure_files = []
            for file in os.listdir(_get_log_path("")):
                if file.startswith("json_structure_") and file.endswith(".txt"):
                    json_structure_files.append(file)

            if json_structure_files:
                master_content.append("=== JSON STRUCTURE REPORTS ===")
                master_content.append("")

                for structure_file in sorted(json_structure_files):
                    structure_path = _get_log_path(structure_file)
                    master_content.append(
                        f"--- {structure_file.replace('json_structure_', '').replace('.txt', '')} ---")
                    with open(structure_path, 'r', encoding='utf-8') as f:
                        master_content.append(f.read())
                    master_content.append("")

        # 3. Landmark Quality Reports
        if include_settings.get("landmark_quality_reports", True):
            quality_files = []
            for file in os.listdir(_get_log_path("")):
                if file.startswith("landmark_quality_") and file.endswith(".txt"):
                    quality_files.append(file)

            if quality_files:
                master_content.append("=== LANDMARK QUALITY REPORTS ===")
                master_content.append("")

                for quality_file in sorted(quality_files):
                    quality_path = _get_log_path(quality_file)
                    master_content.append(
                        f"--- {quality_file.replace('landmark_quality_', '').replace('.txt', '')} ---")
                    with open(quality_path, 'r', encoding='utf-8') as f:
                        master_content.append(f.read())
                    master_content.append("")

        # 4. Coordinate System Reports
        if include_settings.get("coordinate_system_reports", True):
            coord_files = []
            for file in os.listdir(_get_log_path("")):
                if file.startswith("coordinate_system_") and file.endswith(".txt"):
                    coord_files.append(file)

            if coord_files:
                master_content.append("=== COORDINATE SYSTEM REPORTS ===")
                master_content.append("")

                for coord_file in sorted(coord_files):
                    coord_path = _get_log_path(coord_file)
                    master_content.append(f"--- {coord_file.replace('coordinate_system_', '').replace('.txt', '')} ---")
                    with open(coord_path, 'r', encoding='utf-8') as f:
                        master_content.append(f.read())
                    master_content.append("")

        # 5. Comprehensive Analysis Reports
        if include_settings.get("comprehensive_reports", True):
            comp_files = []
            for file in os.listdir(_get_log_path("")):
                if file.startswith("mocap_report_") and file.endswith(".txt"):
                    comp_files.append(file)

            if comp_files:
                master_content.append("=== COMPREHENSIVE ANALYSIS REPORTS ===")
                master_content.append("")

                for comp_file in sorted(comp_files):
                    comp_path = _get_log_path(comp_file)
                    master_content.append(f"--- {comp_file.replace('mocap_report_', '').replace('.txt', '')} ---")
                    with open(comp_path, 'r', encoding='utf-8') as f:
                        master_content.append(f.read())
                    master_content.append("")

        # 6. Comparison Report
        if include_settings.get("comparison_report", True):
            comparison_path = _get_log_path("mocap_pipeline_comparison.txt")
            if os.path.exists(comparison_path):
                master_content.append("=== PIPELINE COMPARISON REPORT ===")
                master_content.append("")
                with open(comparison_path, 'r', encoding='utf-8') as f:
                    master_content.append(f.read())
                master_content.append("")

        master_content.append("=" * 80)
        master_content.append("END OF MASTER REPORT")
        master_content.append("=" * 80)

        # Write the master report
        with open(master_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(master_content))

        script_log(f"Master report written to: {master_path}")

        # Report statistics
        total_reports_included = (
                (1 if include_settings.get("config_report", True) and os.path.exists(
                    _get_log_path("mocap_config_report.txt")) else 0) +
                len([f for f in os.listdir(_get_log_path("")) if
                     f.startswith("json_structure_") and f.endswith(".txt")]) +
                len([f for f in os.listdir(_get_log_path("")) if
                     f.startswith("landmark_quality_") and f.endswith(".txt")]) +
                len([f for f in os.listdir(_get_log_path("")) if
                     f.startswith("coordinate_system_") and f.endswith(".txt")]) +
                len([f for f in os.listdir(_get_log_path("")) if
                     f.startswith("mocap_report_") and f.endswith(".txt")]) +
                (1 if include_settings.get("comparison_report", True) and os.path.exists(
                    _get_log_path("mocap_pipeline_comparison.txt")) else 0)
        )

        script_log(f"Master report includes {total_reports_included} individual reports")

    except Exception as e:
        script_log(f"ERROR concatenating reports: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")


def main():
    """
    Main execution function for the mocap pipeline analysis tool
    """
    script_log("Starting mocap pipeline JSON analysis...")

    try:
        # Load the test_json configuration first
        test_json_config = load_test_json_config()
        script_log("6_test_json configuration loaded successfully")

        # Get current show and scene
        show_name = get_current_show_name()
        scene_name = get_current_scene_name(show_name)
        script_log(f"Analyzing data for: {show_name} - {scene_name}")

        # Run diagnostics based on config
        diagnostic_settings = test_json_config.get("diagnostic_settings", {})
        if diagnostic_settings.get("enable_project_structure_diagnosis", True):
            diagnose_project_structure()

        if diagnostic_settings.get("enable_scene_file_diagnosis", True):
            diagnose_scene_files(show_name, scene_name)

        if diagnostic_settings.get("enable_processing_step_diagnosis", True):
            diagnose_processing_step_paths(show_name, scene_name)

        # Load landmark configuration with fallback to pipeline landmarks
        project_root = get_project_root()
        landmark_config_path = os.path.join(project_root, "config", "landmark_config.json")
        landmark_config = load_landmark_config_with_fallback(landmark_config_path)

        # Log which config was loaded
        if os.path.exists(landmark_config_path):
            script_log(f"Loaded landmark configuration from: {landmark_config_path}")
        else:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            pipeline_landmarks_path = os.path.join(current_script_dir, "0_RunMocapAnimPipeline_LANDMARKS.json")
            if os.path.exists(pipeline_landmarks_path):
                script_log(f"Loaded landmark configuration from pipeline file: {pipeline_landmarks_path}")
            else:
                script_log("Using minimal default landmark configuration")

        script_log(f"Configuration contains {len(landmark_config.get('landmarks', {}))} landmarks")

        # Log motion detection landmarks
        motion_landmarks = []
        for name, data in landmark_config.get('landmarks', {}).items():
            if data.get('motion_detection', False):
                motion_landmarks.append(name)
        script_log(f"Motion detection enabled for {len(motion_landmarks)} landmarks: {', '.join(motion_landmarks)}")

        # Generate configuration report if enabled
        report_settings = test_json_config.get("report_settings", {})
        if report_settings.get("generate_config_report", True):
            config_report = generate_config_report(show_name, scene_name, landmark_config)
            config_report_path = _get_log_path("mocap_config_report.txt")
            with open(config_report_path, 'w', encoding='utf-8') as f:
                f.write(config_report)
            script_log(f"Configuration report written to: {config_report_path}")

        # Get processing steps from scene config instead of hard-coded ones
        scene_config = get_scene_config(show_name, scene_name)
        processing_steps_config = scene_config.get("processing_steps", {})

        # Build processing steps list from scene config
        processing_steps_to_try = []
        for step_name, step_config in processing_steps_config.items():
            processing_steps_to_try.append(step_name)

        # Also try the file-based steps we found in diagnosis
        processing_steps_to_try.extend([
            "Step 2 - Filtered Pose Data",
            "Step 3 - Physics Applied",
            "Step 4 - Final Pose Data",
            "Step2",
            "Step3",
            "Step4"
        ])

        all_analyses = {}
        analyzed_files = set()  # Track which files we've already analyzed

        # Try direct file access as fallback
        scene_folder_name = get_scene_folder_name(show_name, scene_name)
        scene_dir = os.path.join(get_project_root(), "shows", show_name, "inputs", "scenes", scene_folder_name)
        direct_json_files = find_any_json_files(show_name, scene_name)

        for step_name in processing_steps_to_try:
            script_log(f"\n--- Trying step: {step_name} ---")

            try:
                # Get file paths for this processing step
                step_paths = get_processing_step_paths(show_name, scene_name, step_name)
                input_file = step_paths.get('input_file')

                if not input_file:
                    script_log(f"WARNING: No input file configured for {step_name}")
                    continue

                if not os.path.exists(input_file):
                    script_log(f"WARNING: Input file not found for {step_name}: {input_file}")

                    # Try to find the file with a different approach
                    alternative_file = find_alternative_json_file(show_name, scene_name, step_name)
                    if alternative_file and alternative_file not in analyzed_files:
                        script_log(f"Found alternative file: {alternative_file}")
                        input_file = alternative_file
                    else:
                        # Try direct file matching from scene directory
                        script_log("Trying direct file matching...")
                        for json_file in direct_json_files:
                            if step_name.lower() in os.path.basename(json_file).lower():
                                if json_file not in analyzed_files:
                                    script_log(f"Found matching file: {json_file}")
                                    input_file = json_file
                                    break
                        else:
                            continue

                # Skip if we've already analyzed this file
                if input_file in analyzed_files:
                    script_log(f"Already analyzed: {input_file}")
                    continue

                analyzed_files.add(input_file)

                # Generate JSON structure report if enabled
                if report_settings.get("generate_json_structure_reports", True):
                    structure_report = generate_json_structure_report(input_file, step_name)
                    structure_report_path = _get_log_path(f"json_structure_{step_name.replace(' ', '_')}.txt")
                    with open(structure_report_path, 'w', encoding='utf-8') as f:
                        f.write(structure_report)
                    script_log(f"JSON structure report written to: {structure_report_path}")

                # Load frames for additional analyses
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                frames = extract_frames_from_data(data, input_file)

                if not frames:
                    script_log(f"WARNING: Could not find frame data in {input_file}")
                    continue

                # Generate landmark quality report if enabled
                if report_settings.get("generate_landmark_quality_reports", True):
                    coord_system = get_coordinate_system_v2(step_name, test_json_config)
                    quality_report = generate_landmark_quality_report(frames, coord_system, landmark_config)
                    quality_report_path = _get_log_path(f"landmark_quality_{step_name.replace(' ', '_')}.txt")
                    with open(quality_report_path, 'w', encoding='utf-8') as f:
                        f.write(quality_report)
                    script_log(f"Landmark quality report written to: {quality_report_path}")

                # Generate coordinate system analysis if enabled
                if report_settings.get("generate_coordinate_system_reports", True):
                    coord_report = generate_coordinate_system_analysis(frames, step_name, test_json_config)
                    coord_report_path = _get_log_path(f"coordinate_system_{step_name.replace(' ', '_')}.txt")
                    with open(coord_report_path, 'w', encoding='utf-8') as f:
                        f.write(coord_report)
                    script_log(f"Coordinate system report written to: {coord_report_path}")

                # Analyze the JSON file for motion and orientation
                analysis = analyze_json_file_multi_frame(input_file, step_name, landmark_config, test_json_config)

                if analysis:
                    all_analyses[step_name] = analysis
                    script_log(f"[OK] Completed analysis for {step_name}")

                    # Generate comprehensive report if enabled
                    if report_settings.get("generate_comprehensive_reports", True):
                        report_filename = f"mocap_report_{step_name.replace(' ', '_').replace('-', '')}.txt"
                        script_log(f"Generating comprehensive report for {step_name}...")
                        generate_comprehensive_report(analysis, report_filename)
                else:
                    script_log(f"[FAILED] Failed to analyze {step_name}")

            except Exception as e:
                script_log(f"ERROR analyzing {step_name}: {e}")
                import traceback
                script_log(f"Traceback: {traceback.format_exc()}")
                continue

        # If no analyses found with configured steps, try to find any JSON files
        if len(all_analyses) == 0:
            script_log("\n--- Searching for any JSON files to analyze ---")
            any_json_files = find_any_json_files(show_name, scene_name)

            for json_file in any_json_files:
                if json_file not in analyzed_files:
                    script_log(f"Analyzing found file: {json_file}")
                    try:
                        # Use a generic step name based on the filename
                        step_name = os.path.basename(json_file).replace('.json', '')
                        analysis = analyze_json_file_multi_frame(json_file, step_name, landmark_config,
                                                                 test_json_config)
                        if analysis:
                            all_analyses[step_name] = analysis
                            script_log(f"[OK] Completed analysis for {step_name}")

                            # Generate comprehensive report if enabled
                            if report_settings.get("generate_comprehensive_reports", True):
                                report_filename = f"mocap_report_{step_name.replace(' ', '_').replace('-', '')}.txt"
                                generate_comprehensive_report(analysis, report_filename)
                    except Exception as e:
                        script_log(f"ERROR analyzing {json_file}: {e}")

        # Generate summary report comparing all steps if enabled
        if report_settings.get("generate_comparison_report", True) and len(all_analyses) > 1:
            generate_comparison_report(all_analyses)
        elif len(all_analyses) == 0:
            script_log("WARNING: No analyses completed successfully - no JSON files found with pose data")
        else:
            script_log("Single step analyzed - no comparison report generated")

        # Concatenate all reports into master report if enabled
        concatenate_all_reports(test_json_config)

        script_log("Mocap pipeline analysis completed!")

    except Exception as e:
        script_log(f"FATAL ERROR in main execution: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()