# 4D_magic_inner.py (Version 3.0 - Using config file)

import bpy
import json
import os
import sys
import math
import argparse

from mathutils import Vector, Euler, Quaternion, Matrix
from datetime import datetime


def parse_arguments():
    """Parse command line arguments passed from 4D_magic.py"""
    parser = argparse.ArgumentParser(description='4D Magic Inner Script')
    parser.add_argument('--project-root', required=True, help='Path to project root')

    # Parse only the arguments after '--'
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    return parser.parse_args(argv)


# Parse arguments FIRST
args = parse_arguments()

# Add project_root to sys.path so we can import utils
if args.project_root not in sys.path:
    sys.path.append(args.project_root)

# NOW import project utilities
from utils import (
    script_log,
    get_current_show_name,
    get_current_scene_name,
    get_scene_paths,
    get_scene_config
)

# Attempt to import numpy, with a fallback if not available
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ==================================================================================================
# === Video Plane Settings (Hard-coded) ===
# ==================================================================================================

VIDEO_SETTINGS = {
    "show_video": True,
    "video_file_name": "step_1_input.mp4",  # Will look for this in the same directory as the script
    "video_plane": "XZ",  # Options: "XY", "XZ", "YZ"
    "video_size": 1.6,
    "video_rotation": [180, 0, 0]  # Degrees
}


# ==================================================================================================
# === Local Copies of _4D_magic_utils.py ===
#
# These are basic logging utilities.
# ==================================================================================================

# Global settings for debug logging.
_GLOBAL_DEBUG_SETTINGS = {}
# This dictionary will contain specific settings for joints if provided.
_GLOBAL_DEBUG_JOINT_SETTINGS = {}

# Add script directory to sys.path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# ==================================================================================================
# === Local Copies of _4D_magic_blender_utils.py ===
#
# These are Blender-specific utility functions.
# ==================================================================================================

def find_blender_object(object_name, obj_type):
    """Finds a Blender object by name and type."""
    try:
        obj = bpy.data.objects[object_name]
        if obj.type == obj_type:
            return obj
        else:
            script_log(f"Error: Object '{object_name}' exists but is not of type '{obj_type}'.", force_log=True)
            return None
    except KeyError:
        script_log(f"Error: Object '{object_name}' not found in the scene.", force_log=True)
        return None


# ==================================================================================================
# === Video Plane Functions ===
# ==================================================================================================

def create_video_plane():
    """Create a plane and apply the video texture to it"""
    # VIDEO_SETTINGS is now updated from scene-config.json in main()
    if not VIDEO_SETTINGS.get("show_video", False):
        script_log("Video plane creation disabled in settings", force_log=True)
        return

    video_file = VIDEO_SETTINGS.get("video_file_name")
    if not video_file:
        script_log("Warning: No video file name specified. Skipping video plane creation.", force_log=True)
        return

    # Look for video file in the scene inputs directory
    show_name = get_current_show_name()
    scene_name = get_current_scene_name(show_name)
    scene_paths = get_scene_paths(show_name, scene_name)
    inputs_dir = os.path.dirname(scene_paths["output_pose_data"])

    video_file_path = os.path.join(inputs_dir, video_file)
    if not os.path.exists(video_file_path):
        script_log(f"Warning: Video file '{video_file_path}' not found. Skipping video plane creation.", force_log=True)
        return

    # Create the plane
    video_plane_type = VIDEO_SETTINGS.get("video_plane", "XZ").upper()
    rotation_axis = (0, 0, 0)
    if video_plane_type == "XY":
        rotation_axis = (0, 0, 0)
    elif video_plane_type == "XZ":
        rotation_axis = (math.radians(90), 0, 0)
    elif video_plane_type == "YZ":
        rotation_axis = (0, math.radians(90), 0)

    bpy.ops.mesh.primitive_plane_add(
        size=VIDEO_SETTINGS.get("video_size", 1.0),
        enter_editmode=False,
        align='WORLD',
        location=(0, 0, 0),
        rotation=rotation_axis
    )
    video_plane = bpy.context.active_object
    video_plane.name = "VideoPlane"

    # Create material and texture
    mat = bpy.data.materials.new(name="VideoMaterial")
    video_plane.data.materials.append(mat)
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        mat.node_tree.nodes.remove(principled_bsdf)

    # Add a Texture Coordinate and an Image Texture node
    texture_coord = mat.node_tree.nodes.new('ShaderNodeTexCoord')
    image_texture = mat.node_tree.nodes.new('ShaderNodeTexImage')

    # Check if the video file exists before trying to load it
    try:
        image_texture.image = bpy.data.images.load(video_file_path, check_existing=True)
    except RuntimeError:
        script_log(f"Error: Could not load video file '{video_file_path}'. Check file path and format.", force_log=True)
        return

    # Set the image as a movie source (Blender 4.0+ compatible)
    video_image = image_texture.image
    video_image.source = 'MOVIE'

    # Add an Emission shader to the material
    emission_shader = mat.node_tree.nodes.new('ShaderNodeEmission')

    # Connect the nodes
    links = mat.node_tree.links
    links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
    links.new(image_texture.outputs['Color'], emission_shader.inputs['Color'])
    links.new(emission_shader.outputs['Emission'], mat.node_tree.nodes['Material Output'].inputs['Surface'])

    # Set the rotation from the settings
    video_rot_deg = VIDEO_SETTINGS.get("video_rotation", [0, 0, 0])
    video_plane.rotation_euler = Euler(
        (math.radians(video_rot_deg[0]), math.radians(video_rot_deg[1]), math.radians(video_rot_deg[2])), 'XYZ')

    video_plane.location = (0, 0, 0)

    script_log("Video plane created and configured.", force_log=True)


def set_global_xyz(armature_obj, mp_joint_name, mapping_data, bone_type, target_world_location,
                   debug_joint_settings=None):
    """
    Sets the global world coordinates of a bone.

    Args:
        armature_obj (bpy.types.Object): The armature object.
        mp_joint_name (str): The MediaPipe joint name (e.g., "LEFT_WRIST").
        mapping_data (dict): The mapping data.
        bone_type (str): "DEFORM_BONE" or "IK_CONTROLLER".
        target_world_location (Vector): The target world coordinates.
    """
    global _GLOBAL_DEBUG_SETTINGS

    bone_name = mapping_data.get(mp_joint_name, {}).get(bone_type)
    if not bone_name:
        script_log(f"Error: Missing bone name for {mp_joint_name} ({bone_type}) in mapping data.", force_log=True)
        return

    try:
        pose_bone = armature_obj.pose.bones[bone_name]

        # Convert the world location to local space relative to the bone's parent
        if pose_bone.parent:
            # The bone's matrix is its world matrix. We need the parent's world inverse.
            parent_matrix_world_inverted = pose_bone.parent.matrix.inverted()
            local_location = parent_matrix_world_inverted @ target_world_location
        else:
            # If no parent, the local space is the armature's object space
            local_location = armature_obj.matrix_world.inverted() @ target_world_location

        # Set the location of the bone. For IK controllers, this is the final step.
        pose_bone.location = local_location
        script_log(
            f"  set_global_xyz for {mp_joint_name} ({bone_type}): Set local location to {list(pose_bone.location)}",
            mp_joint_name, debug_joint_settings)

    except KeyError as e:
        script_log(f"Error: Bone '{bone_name}' not found in armature '{armature_obj.name}'.", force_log=True,
                   mp_joint_name=mp_joint_name, debug_joint_settings=debug_joint_settings)
    except Exception as e:
        script_log(f"An unexpected error occurred in set_global_xyz for bone '{bone_name}': {e}", force_log=True,
                   mp_joint_name=mp_joint_name, debug_joint_settings=debug_joint_settings)


def get_global_xyz(armature_obj, mp_joint_name, mapping_data, bone_type, debug_joint_settings=None):
    """
    Gets the global world coordinates of a bone's head.

    Args:
        armature_obj (bpy.types.Object): The armature object.
        mp_joint_name (str): The MediaPipe joint name (e.g., "LEFT_WRIST").
        mapping_data (dict): The mapping data.
        bone_type (str): "DEFORM_BONE" or "IK_CONTROLLER".

    Returns:
        Vector: The global coordinates of the bone head, or None if not found.
    """
    global _GLOBAL_DEBUG_SETTINGS

    bone_name = mapping_data.get(mp_joint_name, {}).get(bone_type)
    if not bone_name:
        script_log(f"Error: Missing bone name for {mp_joint_name} ({bone_type}) in mapping data.", force_log=True)
        return None

    try:
        pose_bone = armature_obj.pose.bones[bone_name]
        world_location = armature_obj.matrix_world @ pose_bone.matrix.translation
        script_log(f"  get_global_xyz for {mp_joint_name} ({bone_type}): World Location = [{world_location[0]:.4f}, {world_location[1]:.4f}, {world_location[2]:.4f}]",
                   mp_joint_name, debug_joint_settings)
        return world_location
    except KeyError:
        script_log(f"Error: Bone '{bone_name}' not found in armature '{armature_obj.name}'.", force_log=True,
                   mp_joint_name=mp_joint_name, debug_joint_settings=debug_joint_settings)
        return None


# ==================================================================================================
# === Local Copies of _4D_magic_inner_blender_setup.py ===
#
# These are Blender-specific environment setup functions.
# ==================================================================================================

def setup_blender_environment(armature_name):
    """
    Ensures the specified armature is selected and in Pose Mode.
    Returns the armature object if successful, None otherwise.
    """
    # Find the armature object
    armature_obj = find_blender_object(armature_name, 'ARMATURE')
    if not armature_obj:
        return None

    # Ensure the armature is the active object and selected
    bpy.context.view_layer.objects.active = armature_obj
    armature_obj.select_set(True)

    # Switch to Pose Mode
    if bpy.context.object.mode != 'POSE':
        try:
            bpy.ops.object.mode_set(mode='POSE')
            script_log(f"Entered Pose Mode for armature '{armature_name}'.", force_log=True)
        except Exception as e:
            script_log(f"Error: Could not switch to Pose Mode for '{armature_name}'. {e}", force_log=True)
            return None

    return armature_obj


# ==================================================================================================
# === Local Copies of _4D_magic_inner_params.py ===
#
# These are parameter handling functions.
# ==================================================================================================

def load_and_validate_parameters(params_json_path):
    """
    Loads and validates parameters from JSON file.
    This version expects 'mapping_data' to be the content of exercise_joints.json.

    Args:
        params_json_path: Path to JSON parameters file

    Returns:
        dict: Validated parameters

    Raises:
        FileNotFoundError, json.JSONDecodeError, ValueError
    """
    try:
        with open(params_json_path, 'r') as f:
            all_params = json.load(f)

        # Validate essential parameters
        required_keys = [
            "animation_frames_data",
            "mapping_data",
            "armature_name",
            "mediapipe_biometrics",
            "blender_model_proportions",
            "fk_head_bone_name",
            "output_blend_file_path",
            "sorted_frame_keys",
            "debug_flags",
            "debug_joint_settings"
        ]

        for key in required_keys:
            if key not in all_params:
                raise ValueError(f"Missing essential parameter: '{key}'")

        # Basic type validation
        if not isinstance(all_params["animation_frames_data"], dict):
            raise ValueError("'animation_frames_data' must be a dictionary.")
        if not isinstance(all_params["mapping_data"], dict):
            raise ValueError("'mapping_data' must be a dictionary.")
        if not isinstance(all_params["armature_name"], str):
            raise ValueError("'armature_name' must be a string.")
        if not isinstance(all_params["mediapipe_biometrics"], dict):
            raise ValueError("'mediapipe_biometrics' must be a dictionary.")
        if not isinstance(all_params["blender_model_proportions"], dict):
            raise ValueError("'blender_model_proportions' must be a dictionary.")
        if not isinstance(all_params["output_blend_file_path"], str):
            raise ValueError("'output_blend_file_path' must be a string.")
        if not isinstance(all_params["fk_head_bone_name"], str):
            raise ValueError("'fk_head_bone_name' must be a string.")
        if not isinstance(all_params["sorted_frame_keys"], list):
            raise ValueError("'sorted_frame_keys' must be a list.")
        if not isinstance(all_params["debug_flags"], dict):
            raise ValueError("'debug_flags' must be a dictionary.")
        if not isinstance(all_params["debug_joint_settings"], dict):
            raise ValueError("'debug_joint_settings' must be a dictionary.")

        # Ensure animation_frames_data is not empty and contains the first expected frame key
        animation_data = all_params["animation_frames_data"]
        sorted_frame_keys = all_params["sorted_frame_keys"]

        bpy.context.scene.frame_start = 0
        bpy.context.scene.frame_end = len(sorted_frame_keys) - 1  # frames are 0-indexed

        if not animation_data:
            raise ValueError("'animation_frames_data' dictionary is empty. No animation frames to process.")

        if not sorted_frame_keys:
            raise ValueError("'sorted_frame_keys' list is empty. Cannot determine animation frame order.")

        # Check if the first frame key exists in animation_data
        first_frame_key = str(sorted_frame_keys[0])  # Ensure it's a string key
        if first_frame_key not in animation_data:
            raise ValueError(
                f"First frame key '{first_frame_key}' from 'sorted_frame_keys' not found in 'animation_frames_data'.")

        # Ensure armature_name is correctly derived if metadata is present
        if "metadata" in all_params["mapping_data"] and "ARMATURE_NAME" in all_params["mapping_data"]["metadata"]:
            if all_params["armature_name"] != all_params["mapping_data"]["metadata"]["ARMATURE_NAME"]:
                script_log(f"Warning: 'armature_name' parameter '{all_params['armature_name']}' "
                           f"does not match 'ARMATURE_NAME' in mapping_data metadata "
                           f"('{all_params['mapping_data']['metadata']['ARMATURE_NAME']}'). "
                           f"Using mapping_data's ARMATURE_NAME.", force_log=True)
                all_params["armature_name"] = all_params["mapping_data"]["metadata"]["ARMATURE_NAME"]
        elif "metadata" not in all_params["mapping_data"] or "ARMATURE_NAME" not in all_params["mapping_data"][
            "metadata"]:
            script_log(
                "Warning: 'metadata' or 'ARMATURE_NAME' missing in mapping_data (exercise_joints.json). Defaulting armature name.",
                force_log=True)

        return all_params

    except FileNotFoundError:
        script_log(f"Error: Parameters file not found at '{params_json_path}'", force_log=True)
        sys.exit(1)
    except json.JSONDecodeError:
        script_log(f"Error: Invalid JSON in '{params_json_path}'", force_log=True)
        sys.exit(1)
    except ValueError as e:
        script_log(f"Validation error: {e}", force_log=True)
        sys.exit(1)


# ==================================================================================================
# === Hand and Foot Control ===
# ==================================================================================================

def get_elbow_ik_pole_bone_name(mp_joint_name, hand_mapping_data):
    """
    Gets the elbow IK pole bone name from the hand mapping data for the given wrist joint.

    Args:
        mp_joint_name (str): The MediaPipe joint name ("RIGHT_WRIST" or "LEFT_WRIST")
        hand_mapping_data (dict): The loaded content from exercise_hands.json

    Returns:
        str: The Blender bone name for the elbow IK pole (e.g., "IK-Pole-Forearm.R")
        or None if not found
    """
    if mp_joint_name not in ["RIGHT_WRIST", "LEFT_WRIST"]:
        return None

    joint_config = hand_mapping_data.get(mp_joint_name)
    if not joint_config:
        return None

    return joint_config.get("ELBOW_IK_POLE_BONE")

def get_shoulder_deform_bone_name(mp_joint_name, hand_mapping_data):
    """
    Gets the shoulder deform bone name from the hand mapping data for the given wrist joint.

    Args:
        mp_joint_name (str): The MediaPipe joint name ("RIGHT_WRIST" or "LEFT_WRIST")
        hand_mapping_data (dict): The loaded content from exercise_hands.json

    Returns:
        str: The Blender bone name for the shoulder DEF- bone (e.g., "DEF-Clavicle.R")
        or None if not found
    """
    if mp_joint_name not in ["RIGHT_WRIST", "LEFT_WRIST"]:
        return None

    joint_config = hand_mapping_data.get(mp_joint_name)
    if not joint_config:
        return None

    return joint_config.get("SHOULDER_DEFORM_BONE")


def get_reflection_matrix(mp_joint_name, hand_mapping_data):
    """
    Gets the handedness reflection matrix from the hand mapping data.

    Args:
        mp_joint_name (str): The MediaPipe joint name ("RIGHT_WRIST" or "LEFT_WRIST")
        hand_mapping_data (dict): The loaded content from exercise_hands.json

    Returns:
        Matrix: The reflection matrix to account for handedness differences
        or None if not found
    """
    if mp_joint_name not in ["RIGHT_WRIST", "LEFT_WRIST"]:
        return None

    joint_config = hand_mapping_data.get(mp_joint_name)
    if not joint_config:
        return None

    reflection_data = joint_config.get("HANDEDNESS_REFLECTION")
    if not reflection_data or len(reflection_data) != 9:
        return None

    # Convert the 9-element list into a 3x3 matrix
    try:
        return Matrix((
            (reflection_data[0], reflection_data[1], reflection_data[2]),
            (reflection_data[3], reflection_data[4], reflection_data[5]),
            (reflection_data[6], reflection_data[7], reflection_data[8])
        ))
    except Exception as e:
        script_log(f"Error creating reflection matrix: {e}", mp_joint_name, force_log=True)
        return None


def set_ik_hand_location_and_rotation(armature_obj, mp_joint_name, hand_mp_world_location, mapping_data,
                                      debug_joint_settings=None):
    if debug_joint_settings is None:
        debug_joint_settings = {}

    try:
        # Load hand configuration
        script_dir = os.path.dirname(os.path.abspath(__file__))
        hands_config_path = os.path.join(script_dir, "exercise_hands.json")

        with open(hands_config_path, 'r') as f:
            hands_config = json.load(f)

        joint_config = hands_config.get(mp_joint_name, {})
        base_offsets = joint_config.get("additional_rotation_offset", {})

        script_log(f"  Loaded hand config for {mp_joint_name}: {joint_config}", mp_joint_name, debug_joint_settings)
        script_log(f"  Rotation offsets: {base_offsets}", mp_joint_name, debug_joint_settings)

        # 1. Set the location first
        set_global_xyz(
            armature_obj,
            mp_joint_name,
            mapping_data,
            bone_type="IK_CONTROLLER",
            target_world_location=hand_mp_world_location,
            debug_joint_settings=debug_joint_settings
        )

        # 2. Get the IK controller bone
        ik_controller_name = mapping_data.get(mp_joint_name, {}).get("IK_CONTROLLER")
        if not ik_controller_name:
            script_log(f"Error: Missing IK controller for {mp_joint_name}", mp_joint_name, debug_joint_settings)
            return

        pose_bone = armature_obj.pose.bones.get(ik_controller_name)
        if not pose_bone:
            script_log(f"Error: Pose bone '{ik_controller_name}' not found", mp_joint_name, debug_joint_settings)
            return

        # 3. Get required bones for rotation calculation
        elbow_bone_name = joint_config.get("ELBOW_IK_POLE_BONE")
        shoulder_bone_name = joint_config.get("SHOULDER_DEFORM_BONE")

        if not elbow_bone_name or not shoulder_bone_name:
            script_log(f"Warning: Missing elbow/shoulder bone names in config for {mp_joint_name}",
                       mp_joint_name, debug_joint_settings)
            return

        elbow_bone = armature_obj.pose.bones.get(elbow_bone_name)
        shoulder_bone = armature_obj.pose.bones.get(shoulder_bone_name)

        if not elbow_bone or not shoulder_bone:
            script_log(f"Warning: Could not find elbow/shoulder bones in armature",
                       mp_joint_name, debug_joint_settings)
            return

        # 4. Calculate world positions of posed bones (corrected approach)
        shoulder_world = armature_obj.matrix_world @ shoulder_bone.head
        elbow_world = armature_obj.matrix_world @ elbow_bone.head
        wrist_world = armature_obj.matrix_world @ pose_bone.head

        # 5. Calculate vectors for rotation (using current posed positions)
        upper_arm = (elbow_world - shoulder_world).normalized()
        forearm = (wrist_world - elbow_world).normalized()

        # Calculate cross product to check if arm is straight
        cross_product = upper_arm.cross(forearm)
        cross_product_length = cross_product.length

        # Small threshold to detect straight arm (adjust as needed)
        STRAIGHT_ARM_THRESHOLD = 0.01

        if cross_product_length < STRAIGHT_ARM_THRESHOLD:
            # Arm is straight - use forearm direction for rotation
            script_log(
                f"  Arm is straight (cross product length: {cross_product_length:.6f}), using forearm direction for rotation",
                mp_joint_name, debug_joint_settings)

            # Get reflection matrix for handedness
            reflection_matrix = get_reflection_matrix(mp_joint_name, hands_config)
            if not reflection_matrix:
                reflection_matrix = Matrix()  # Identity matrix as fallback

            # Create rotation matrix from forearm vector
            # For a straight arm, we'll align the Y-axis with the forearm
            # and choose arbitrary but consistent X and Z axes
            y_axis = reflection_matrix @ forearm

            # Create an arbitrary but consistent right vector
            if abs(y_axis.z) > 0.707:  # If forearm is mostly up/down
                x_axis = reflection_matrix @ Vector((1, 0, 0)).cross(y_axis).normalized()
            else:
                x_axis = reflection_matrix @ Vector((0, 0, 1)).cross(y_axis).normalized()

            z_axis = x_axis.cross(y_axis).normalized()

            rotation_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()
        else:
            # Normal case - arm is bent
            # Calculate arm plane normal (points outward from body)
            arm_plane_normal = cross_product.normalized()

            script_log(f"  Vectors for {mp_joint_name}:", mp_joint_name, debug_joint_settings)
            script_log(f"    Upper Arm: {upper_arm}", mp_joint_name, debug_joint_settings)
            script_log(f"    Forearm: {forearm}", mp_joint_name, debug_joint_settings)
            script_log(f"    Arm Plane Normal: {arm_plane_normal}", mp_joint_name, debug_joint_settings)

            # Get reflection matrix for handedness
            reflection_matrix = get_reflection_matrix(mp_joint_name, hands_config)
            if not reflection_matrix:
                reflection_matrix = Matrix()  # Identity matrix as fallback

            # Create rotation matrix axes
            x_axis = reflection_matrix @ forearm.cross(arm_plane_normal).normalized()
            y_axis = reflection_matrix @ forearm
            z_axis = reflection_matrix @ arm_plane_normal

            # Create final rotation matrix
            rotation_matrix = Matrix((x_axis, y_axis, z_axis)).transposed()

        # Convert to Euler XYZ
        rotation_euler = rotation_matrix.to_euler('XYZ')

        # Apply base offsets (convert degrees to radians)
        rotation_euler.x += math.radians(base_offsets.get("x", 0))
        rotation_euler.y += math.radians(base_offsets.get("y", 0))
        rotation_euler.z += math.radians(base_offsets.get("z", 0))

        # 6. Set rotation mode and apply rotation
        pose_bone.rotation_mode = 'XYZ'
        pose_bone.rotation_euler = rotation_euler

        # 7. Keyframe both location and rotation
        current_frame = bpy.context.scene.frame_current
        pose_bone.keyframe_insert(data_path="location", frame=current_frame)
        pose_bone.keyframe_insert(data_path="rotation_euler", frame=current_frame)

        # Log results
        script_log(f"Hand rotation set | Euler XYZ: ({math.degrees(rotation_euler.x):.1f}°, "
                   f"{math.degrees(rotation_euler.y):.1f}°, {math.degrees(rotation_euler.z):.1f}°)",
                   mp_joint_name, debug_joint_settings)

    except Exception as e:
        script_log(f"Error in hand rotation for {mp_joint_name}: {str(e)}",
                   mp_joint_name, debug_joint_settings, force_log=True)


def process_hip_movement(frame_idx, animation_frames_data, armature_obj, mapping_data, debug_joint_settings):
    """Special handling for hip translation and rotation"""
    frame_data = animation_frames_data[str(frame_idx)]

    # Get left and right hip positions
    left_hip_data = frame_data.get("LEFT_HIP")
    right_hip_data = frame_data.get("RIGHT_HIP")

    if not left_hip_data or not right_hip_data:
        return

    left_hip_pos = Vector((left_hip_data['x'], left_hip_data['y'], left_hip_data['z']))
    right_hip_pos = Vector((right_hip_data['x'], right_hip_data['y'], right_hip_data['z']))

    # Calculate average position for translation
    hip_center = (left_hip_pos + right_hip_pos) / 2

    # Set hip translation
    set_global_xyz(
        armature_obj,
        "LEFT_HIP",  # Use either hip joint name since they share the same controller
        mapping_data,
        bone_type="IK_CONTROLLER",
        target_world_location=hip_center,
        debug_joint_settings=debug_joint_settings
    )

    # Calculate hip rotation
    calculate_hip_rotation(armature_obj, left_hip_pos, right_hip_pos, mapping_data, debug_joint_settings)

    # Keyframe the hip controller
    hip_controller_name = mapping_data.get("LEFT_HIP", {}).get("IK_CONTROLLER")
    if hip_controller_name:
        pose_bone = armature_obj.pose.bones.get(hip_controller_name)
        if pose_bone:
            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
            pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)


def calculate_hip_rotation(armature_obj, left_hip_pos, right_hip_pos, mapping_data, debug_joint_settings):
    """Calculate and apply hip rotation based on left and right hip positions"""
    hip_controller_name = mapping_data.get("LEFT_HIP", {}).get("IK_CONTROLLER")
    if not hip_controller_name:
        return

    pose_bone = armature_obj.pose.bones.get(hip_controller_name)
    if not pose_bone:
        return

    # Calculate the vector between hips (left to right)
    hip_vector = right_hip_pos - left_hip_pos

    # Normalize and calculate rotation
    if hip_vector.length > 0:
        # Calculate the forward direction (approximate - you might need to adjust this)
        # For hips, we can use the vector perpendicular to both hip vector and up vector
        up_vector = Vector((0, 0, 1))  # Z is up in Blender
        forward_vector = up_vector.cross(hip_vector).normalized()

        # Create rotation matrix
        right_vector = hip_vector.normalized()
        up_vector_corrected = right_vector.cross(forward_vector).normalized()

        rotation_matrix = Matrix((right_vector, forward_vector, up_vector_corrected)).transposed()

        # Convert to Euler and apply
        rotation_euler = rotation_matrix.to_euler('XYZ')
        pose_bone.rotation_mode = 'XYZ'
        pose_bone.rotation_euler = rotation_euler

        script_log(f"Hip rotation set | Euler XYZ: ({math.degrees(rotation_euler.x):.1f}°, "
                   f"{math.degrees(rotation_euler.y):.1f}°, {math.degrees(rotation_euler.z):.1f}°)",
                   "LEFT_HIP", debug_joint_settings)


# ==================================================================================================
# === Core Animation Functions ===
# ==================================================================================================

def apply_animation_to_frame(frame_idx, animation_frames_data, armature_obj, mapping_data,
                             blender_model_proportions, mediapipe_biometrics, debug_joint_settings):
    global _GLOBAL_DEBUG_SETTINGS
    bpy.context.scene.frame_set(frame_idx)
    _GLOBAL_DEBUG_SETTINGS = debug_joint_settings.get("metadata", {}).get("debug_flags", {})

    # Process hips first (special handling)
    process_hip_movement(frame_idx, animation_frames_data, armature_obj, mapping_data, debug_joint_settings)

    for mp_joint_name, joint_data in mapping_data.items():
        if joint_data.get("DRIVE_THIS_JOINT", False) and mp_joint_name not in ["LEFT_HIP", "RIGHT_HIP"]:
            frame_data = animation_frames_data[str(frame_idx)]
            mp_location = frame_data.get(mp_joint_name)

            if mp_location:
                mp_vector = Vector((mp_location['x'], mp_location['y'], mp_location['z']))

                if "WRIST" in mp_joint_name:
                    # Hand IK controller logic
                    set_ik_hand_location_and_rotation(
                        armature_obj,
                        mp_joint_name,
                        mp_vector,
                        mapping_data,
                        debug_joint_settings
                    )

                    ik_controller_name = mapping_data.get(mp_joint_name, {}).get("IK_CONTROLLER")
                    if ik_controller_name:
                        pose_bone = armature_obj.pose.bones.get(ik_controller_name)
                        if pose_bone:
                            if bpy.context.object.mode != 'POSE':
                                bpy.ops.object.mode_set(mode='POSE')

                            pose_bone.rotation_mode = 'XYZ'
                            pose_bone.keyframe_insert(data_path="location", frame=frame_idx)
                            pose_bone.keyframe_insert(data_path="rotation_euler", frame=frame_idx)

                            # Corrected diagnostic logging
                            action = armature_obj.animation_data.action if armature_obj.animation_data else None
                            if action:
                                loc_found = any(fc.data_path == f'pose.bones["{ik_controller_name}"].location' for fc in
                                                action.fcurves)
                                rot_found = any(
                                    fc.data_path == f'pose.bones["{ik_controller_name}"].rotation_euler' for fc in
                                    action.fcurves)
                                script_log(f"  Hand IK controller '{ik_controller_name}' keyframes: "
                                           f"Location={'found' if loc_found else 'missing'} "
                                           f"Rotation={'found' if rot_found else 'missing'}",
                                           mp_joint_name, debug_joint_settings)
                            else:
                                script_log(f"No animation action found for armature", mp_joint_name,
                                           debug_joint_settings)
                else:
                    # Standard bone logic
                    set_global_xyz(
                        armature_obj,
                        mp_joint_name,
                        mapping_data,
                        bone_type="IK_CONTROLLER",
                        target_world_location=mp_vector,
                        debug_joint_settings=debug_joint_settings
                    )
                    pose_bone = armature_obj.pose.bones.get(mapping_data.get(mp_joint_name, {}).get("IK_CONTROLLER"))
                    if pose_bone:
                        pose_bone.keyframe_insert(data_path="location", frame=frame_idx)

    # Ensure linear interpolation for rotation
    if armature_obj.animation_data and armature_obj.animation_data.action:
        for fcurve in armature_obj.animation_data.action.fcurves:
            if "rotation_quaternion" in fcurve.data_path:
                for kf in fcurve.keyframe_points:
                    kf.interpolation = 'LINEAR'

    script_log(f"Frame {frame_idx} animation applied", force_log=True)


def save_blender_file(output_blend_file_path, debug_joint_settings=None):
    """
    Saves the current Blender file to the specified path.
    """
    try:
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_file_path)
        script_log(f"Blender file saved to: {output_blend_file_path}", force_log=True)
    except Exception as e:
        script_log(f"Error: Failed to save Blender file to {output_blend_file_path}: {e}", force_log=True)


# ==================================================================================================
# === Main execution block ===
# ==================================================================================================

def main():
    script_log("=== 4D MAGIC INNER STARTED ===", force_log=True)

    try:
        # Use project utilities to get scene information
        show_name = get_current_show_name()
        scene_name = get_current_scene_name(show_name)
        scene_paths = get_scene_paths(show_name, scene_name)

        # Get scene configuration
        scene_config = get_scene_config(show_name, scene_name)

        # Define file paths using scene-config.json
        processing_steps = scene_config.get("processing_steps", {})

        # Get input JSON file from processing steps
        apply_physics_step = processing_steps.get("apply_physics", {})
        blender_animation_step = processing_steps.get("blender_animation", {})

        # Input JSON is the output from apply_physics step
        input_json_relative = blender_animation_step.get("input_file", "step_4_input.json")

        # The inputs folder is where scene-config.json lives
        inputs_dir = os.path.dirname(scene_paths["output_pose_data"])
        input_json_file = os.path.join(inputs_dir, input_json_relative)

        # Get armature name from scene-config.json
        armature_name = scene_config.get("armature_name", "Armature")

        # Get video settings from scene-config.json
        video_settings = scene_config.get("video_settings", {})

        # Update global VIDEO_SETTINGS with config values
        global VIDEO_SETTINGS
        VIDEO_SETTINGS.update(video_settings)

        # Use get_show_path and get_scene_folder_name for correct output path
        from utils import get_show_path, get_scene_folder_name
        show_path = get_show_path(show_name)
        scene_folder_name = get_scene_folder_name(show_name, scene_name)
        outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
        output_blend_file = os.path.join(outputs_dir, f"{scene_name}.blend")

        # Joints config path (this should remain in script directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        joints_config_file = os.path.join(script_dir, "exercise_joints.json")

        script_log(f"Show: {show_name}", force_log=True)
        script_log(f"Scene: {scene_name}", force_log=True)
        script_log(f"Input JSON: {input_json_file}", force_log=True)
        script_log(f"Output .blend: {output_blend_file}", force_log=True)
        script_log(f"Joints Config: {joints_config_file}", force_log=True)
        script_log(f"Armature Name: {armature_name}", force_log=True)
        script_log(f"Video Settings: {video_settings}", force_log=True)

        # Load the output Blender file (which should have been copied by 4D_magic.py)
        if not os.path.exists(output_blend_file):
            script_log(f"Error: Output Blender file not found at {output_blend_file}", force_log=True)
            script_log(f"This file should have been created by 4D_magic.py", force_log=True)
            return

        script_log(f"Loading Blender file: {output_blend_file}", force_log=True)
        bpy.ops.wm.open_mainfile(filepath=output_blend_file)

        # Load animation data
        if not os.path.exists(input_json_file):
            script_log(f"Error: Input JSON file not found at {input_json_file}", force_log=True)
            return

        with open(input_json_file, 'r') as f:
            animation_frames_data = json.load(f)

        # Load joints config
        if not os.path.exists(joints_config_file):
            script_log(f"Error: Joints config file not found at {joints_config_file}", force_log=True)
            return

        with open(joints_config_file, 'r') as f:
            mapping_data = json.load(f)

        # Create parameters structure - use armature name from scene-config.json
        all_params = {
            "animation_frames_data": animation_frames_data,
            "mapping_data": mapping_data,
            "armature_name": armature_name,  # Use from scene-config.json
            "mediapipe_biometrics": {},
            "blender_model_proportions": {},
            "fk_head_bone_name": "Head",
            "output_blend_file_path": output_blend_file,
            "sorted_frame_keys": sorted(animation_frames_data.keys(), key=lambda x: int(x)),
            "debug_flags": {},
            "debug_joint_settings": mapping_data.get("debug_joint_settings", {})
        }

        _GLOBAL_DEBUG_SETTINGS = all_params.get("debug_flags", {})
        _GLOBAL_DEBUG_JOINT_SETTINGS = all_params.get("debug_joint_settings", {})

        animation_frames_data = all_params["animation_frames_data"]
        mapping_data = all_params["mapping_data"]
        armature_name = all_params["armature_name"]
        mediapipe_biometrics = all_params["mediapipe_biometrics"]
        blender_model_proportions = all_params["blender_model_proportions"]
        output_blend_file_path = all_params["output_blend_file_path"]
        sorted_frame_keys = all_params["sorted_frame_keys"]

        # Setup Blender environment
        armature_obj = setup_blender_environment(armature_name)
        if not armature_obj:
            script_log("Failed to set up Blender environment. Exiting.", force_log=True)
            return

        # Apply animation to each frame
        for frame_key in sorted_frame_keys:
            frame_idx = int(frame_key)
            apply_animation_to_frame(frame_idx, animation_frames_data, armature_obj, mapping_data,
                                     blender_model_proportions, mediapipe_biometrics, _GLOBAL_DEBUG_JOINT_SETTINGS)

        # Create video plane
        script_log("Creating video plane...", force_log=True)
        create_video_plane()

        # Save the Blender file with the animation (will overwrite the same file)
        script_log(f"Saving Blender file to: {output_blend_file_path}", force_log=True)
        save_blender_file(output_blend_file_path, _GLOBAL_DEBUG_JOINT_SETTINGS)

        script_log("=== 4D MAGIC INNER COMPLETE ===", force_log=True)

    except Exception as e:
        script_log(f"FATAL ERROR in main execution: {e}", force_log=True)

if __name__ == "__main__":
    main()

