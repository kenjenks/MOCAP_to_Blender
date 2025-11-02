# 4K_kid_anim_inner.py (Version 1.2 - Make hips and shoulders rotate where they should)

import bpy
import bmesh
import sys
import os
import argparse
import json
from mathutils import Vector, Matrix, Quaternion
import math

######################################################################################################

def parse_arguments():
    """Parse command line arguments passed from 4K_kid_anim.py"""
    parser = argparse.ArgumentParser(description='4K Kid Animation Inner Script')
    parser.add_argument('--project-root', required=True, help='Path to project root')
    parser.add_argument('--show', required=True, help='Show name')
    parser.add_argument('--scene', required=True, help='Scene name')

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    return parser.parse_args(argv)


# Parse arguments
args = parse_arguments()

# Add project_root to sys.path so we can import utils
if args.project_root not in sys.path:
    sys.path.append(args.project_root)

# Import project utilities
try:
    from utils import script_log, comment, get_scene_config, get_processing_step_paths, get_scene_paths
except ImportError as e:
    comment(f"FAILED to import utils: {e}", force_log=True)
    sys.exit(1)

# Global variables
mocap_data = {}
bone_definitions = {}
frame_numbers = []
squish_factors = {}

######################################################################################################

def clear_existing_animation():
    """Clear all existing animation data from the scene"""
    script_log("=== CLEARING EXISTING ANIMATION ===", force_log=True)

    # Clear animation from all objects
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data_clear()
            script_log(f"Cleared animation from: {obj.name}", force_log=True)

    # Clear animation from pose bones by clearing their keyframes
    for armature in [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']:
        # Clear animation from the armature object itself
        if armature.animation_data:
            armature.animation_data_clear()

        # Clear pose bone animations by removing all keyframes
        for bone in armature.pose.bones:
            # Clear location keyframes
            if bone.location != Vector((0, 0, 0)):
                bone.location = Vector((0, 0, 0))

            # Clear rotation keyframes
            if hasattr(bone, 'rotation_quaternion') and bone.rotation_quaternion != Quaternion():
                bone.rotation_quaternion = Quaternion()
            if hasattr(bone, 'rotation_euler') and bone.rotation_euler != Vector((0, 0, 0)):
                bone.rotation_euler = Vector((0, 0, 0))

            # Clear scale keyframes
            if bone.scale != Vector((1, 1, 1)):
                bone.scale = Vector((1, 1, 1))

        script_log(f"Cleared animation from armature: {armature.name}", force_log=True)

    # Clear all actions (animation data)
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    script_log("Cleared all animation actions")

    # Reset scene frame range to defaults
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250
    script_log("Reset scene frame range to 1-250")

######################################################################################################

def load_config_and_data():
    """Load configuration and CURRENT mocap data using project utilities"""
    global mocap_data, bone_definitions, frame_numbers, squish_factors

    try:
        # Get scene configuration
        scene_config = get_scene_config(args.show, args.scene)

        # Get processing step paths
        step_paths = get_processing_step_paths(args.show, args.scene, "kid_animation")

        # Input JSON is from the apply_physics step - use CURRENT data
        processing_steps = scene_config.get("processing_steps", {})
        apply_physics_step = processing_steps.get("apply_physics", {})
        input_json_relative = apply_physics_step.get("output_file", "step_4_input.json")

        # Build absolute path to input JSON
        scene_paths = get_scene_paths(args.show, args.scene)
        inputs_dir = os.path.dirname(scene_paths["output_pose_data"])
        INPUT_JSON_FILE = os.path.join(inputs_dir, input_json_relative)

        # Load kid figure specific config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        KID_CONFIG_FILE = os.path.join(script_dir, "4K_kid_config.json")

        with open(KID_CONFIG_FILE, 'r') as file:
            config = json.load(file)
            bone_definitions = config.get("bone_definitions", {})

            # CORRECTED: Load squish factors from kid_figure_settings
            kid_settings = config.get("kid_figure_settings", {})
            squish_factors = {
                "x": kid_settings.get("x_squish_fraction", 1.0),
                "y": kid_settings.get("y_squish_fraction", 1.0),
                "z": kid_settings.get("z_squish_fraction", 1.0)
            }

        # Load CURRENT MediaPipe JSON data
        with open(INPUT_JSON_FILE, 'r') as file:
            mocap_data = json.load(file)

        script_log(f"Loaded CURRENT mocap data from: {INPUT_JSON_FILE}", force_log=True)
        script_log(f"Loaded kid config from: {KID_CONFIG_FILE}", force_log=True)
        script_log(f"Found {len(mocap_data)} frames of animation data", force_log=True)

        # LOG THE ACTUAL SQUISH FACTORS BEING USED
        script_log(
            f"Squish factors from config: X={squish_factors['x']}, Y={squish_factors['y']}, Z={squish_factors['z']}",
            force_log=True)

    except FileNotFoundError as e:
        script_log(f"Error: {e}", force_log=True)
        sys.exit(1)
    except json.JSONDecodeError as e:
        script_log(f"Error: Could not decode JSON from file: {e}", force_log=True)
        sys.exit(1)

    # Get the frame numbers from the JSON data
    frame_numbers = sorted([int(frame) for frame in mocap_data.keys()])
    if not frame_numbers:
        script_log("Error: No frame data found in JSON file.", force_log=True)
        sys.exit(1)

    script_log(f"Animation will span frames {frame_numbers[0]} to {frame_numbers[-1]}", force_log=True)

######################################################################################################

def find_armature_and_control_points():
    """Find the existing armature and ALL control points in the scene"""
    armature_obj = None
    control_points = {}

    # DEBUG: List all objects to see what we have
    script_log("=== DEBUG: ALL OBJECTS IN SCENE ===", force_log=True)
    for obj in bpy.data.objects:
        script_log(f"Object: {obj.name} (Type: {obj.type})", force_log=True)

    # Find the armature - look specifically for ARMATURE type objects
    for obj in bpy.data.objects:
        if obj.type == 'ARMATURE':
            armature_obj = obj
            script_log(f"✓ Found armature: {armature_obj.name}", force_log=True)
            break

    # COMPREHENSIVE search for control points
    script_log("=== COMPREHENSIVE CONTROL POINT SEARCH ===", force_log=True)

    # Look in ALL objects regardless of type
    for obj in bpy.data.objects:
        obj_name = obj.name

        # Skip the armature itself
        if obj == armature_obj:
            continue

        # Look for objects with MediaPipe joint names (MESH spheres)
        mp_joints_in_name = [
            "LEFT_HIP", "RIGHT_HIP", "LEFT_SHOULDER", "RIGHT_SHOULDER",
            "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_INDEX", "RIGHT_INDEX", "LEFT_KNEE", "RIGHT_KNEE",
            "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
            "NOSE", "HEAD_TOP"
        ]

        for joint in mp_joints_in_name:
            if joint in obj_name:
                control_points[joint] = obj
                script_log(f"Found control point: {obj_name} -> {joint} (Type: {obj.type})", force_log=True)
                break

        # Look for virtual midpoints (EMPTY objects)
        if "HIP_MIDPOINT" in obj_name and obj.type == 'EMPTY':
            control_points["HIP_MIDPOINT"] = obj
            script_log(f"Found HIP_MIDPOINT: {obj_name}", force_log=True)
        elif "SHOULDER_MIDPOINT" in obj_name and obj.type == 'EMPTY':
            control_points["SHOULDER_MIDPOINT"] = obj
            script_log(f"Found SHOULDER_MIDPOINT: {obj_name}", force_log=True)

    # Final verification
    script_log(f"=== CONTROL POINT SEARCH COMPLETE ===", force_log=True)
    script_log(f"Total control points found: {len(control_points)}", force_log=True)

    for joint_name, obj in control_points.items():
        script_log(f"  ✓ {joint_name}: {obj.name} ({obj.type})", force_log=True)

    return armature_obj, control_points

######################################################################################################

def calculate_virtual_position(frame_data, virtual_calc):
    """Calculate virtual position from multiple MediaPipe landmarks"""
    if not virtual_calc or len(virtual_calc) == 0:
        return Vector((0, 0, 0))

    if len(virtual_calc) == 1:
        # Single point - just return its position
        mp_name = virtual_calc[0]
        if mp_name in frame_data:
            pos_data = frame_data[mp_name]
            return Vector((pos_data["x"], pos_data["y"], pos_data["z"]))
        return Vector((0, 0, 0))

    # Multiple points - calculate midpoint
    total_vec = Vector((0, 0, 0))
    valid_points = 0

    for mp_name in virtual_calc:
        if mp_name in frame_data:
            pos_data = frame_data[mp_name]
            total_vec += Vector((pos_data["x"], pos_data["y"], pos_data["z"]))
            valid_points += 1

    if valid_points > 0:
        return total_vec / valid_points
    return Vector((0, 0, 0))

######################################################################################################

def update_virtual_midpoints(frame_data, figure_name):
    """Update virtual midpoint positions based on current frame data"""
    # Calculate hip midpoint from LEFT_HIP and RIGHT_HIP
    left_hip_pos = Vector((frame_data["LEFT_HIP"]["x"], frame_data["LEFT_HIP"]["y"], frame_data["LEFT_HIP"]["z"]))
    right_hip_pos = Vector(
        (frame_data["RIGHT_HIP"]["x"], frame_data["RIGHT_HIP"]["y"], frame_data["RIGHT_HIP"]["z"]))
    hip_midpoint = (left_hip_pos + right_hip_pos) / 2

    # Calculate shoulder midpoint from LEFT_SHOULDER and RIGHT_SHOULDER
    left_shoulder_pos = Vector(
        (frame_data["LEFT_SHOULDER"]["x"], frame_data["LEFT_SHOULDER"]["y"], frame_data["LEFT_SHOULDER"]["z"]))
    right_shoulder_pos = Vector(
        (frame_data["RIGHT_SHOULDER"]["x"], frame_data["RIGHT_SHOULDER"]["y"], frame_data["RIGHT_SHOULDER"]["z"]))
    shoulder_midpoint = (left_shoulder_pos + right_shoulder_pos) / 2

    # Update the empty objects
    hip_empty = bpy.data.objects.get(f"{figure_name}_HIP_MIDPOINT")
    shoulder_empty = bpy.data.objects.get(f"{figure_name}_SHOULDER_MIDPOINT")

    if hip_empty:
        hip_empty.location = hip_midpoint
    if shoulder_empty:
        shoulder_empty.location = shoulder_midpoint

    return hip_midpoint, shoulder_midpoint

######################################################################################################

def animate_rig(armature_obj, control_points):
    """
    Animate the existing rig using the control points and virtual midpoints
    with coordinate-based squish factors and stretching to match landmark positions
    """
    script_log(f"=== APPLYING FRESH ANIMATION for {len(frame_numbers)} frames ===", force_log=True)

    if not armature_obj or armature_obj.type != 'ARMATURE':
        script_log(
            f"FATAL ERROR: Object passed to animate_rig is not an Armature. Type found: {armature_obj.type if armature_obj else 'None'}",
            force_log=True)
        script_log(f"Object name: {armature_obj.name if armature_obj else 'None'}", force_log=True)
        return

    # Clear existing animation data
    if armature_obj.animation_data:
        armature_obj.animation_data_clear()

    for control_obj in control_points.values():
        if control_obj.animation_data:
            control_obj.animation_data_clear()

    # Use the squish factors loaded from config
    effective_squish_factors = squish_factors

    script_log(
        f"Using coordinate squish factors from config: X={effective_squish_factors['x']}, Y={effective_squish_factors['y']}, Z={effective_squish_factors['z']}")

    # === Set up context for pose_bones access ===
    original_mode = bpy.context.object.mode if bpy.context.object else 'OBJECT'
    original_active = bpy.context.view_layer.objects.active

    # Select and activate the armature
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj

    # Switch to Pose Mode to access pose.bones
    bpy.ops.object.mode_set(mode='POSE')
    pose_bones = armature_obj.pose.bones

    # List available pose bones for debugging
    script_log("=== AVAILABLE POSE BONES ===")
    for bone_name in sorted(pose_bones.keys()):
        script_log(f"  - {bone_name}")

    # Store original bone lengths for reference - access through armature data
    original_bone_data = {}
    armature_data = armature_obj.data

    # Look for bones with DEF_ prefix (deformation bones)
    bone_names_to_check = [
        "DEF_Hip", "DEF_Shoulder", "DEF_Neck", "DEF_Spine", "DEF_Spine1", "DEF_Spine2", "DEF_Head",
        "DEF_LeftHip", "DEF_RightHip", "DEF_LeftShoulder", "DEF_RightShoulder"
    ]

    for bone_name in bone_names_to_check:
        armature_bone = armature_data.bones.get(bone_name)
        if armature_bone:
            original_bone_data[bone_name] = armature_bone.length
            script_log(f"Original {bone_name} bone length: {armature_bone.length:.4f}", force_log=True)
        else:
            script_log(f"⚠ Bone not found in armature: {bone_name}", force_log=True)

    # Animate control points and bones starting at frame 1
    for frame_number in frame_numbers:
        # Convert frame number to start at 1 instead of 0
        blender_frame = frame_number + 1
        frame_data = mocap_data[str(frame_number)]
        bpy.context.scene.frame_set(blender_frame)

        # Get original joint positions (already in Blender Z-up coordinates)
        left_shoulder_pos = Vector((
            frame_data["LEFT_SHOULDER"]["x"],
            frame_data["LEFT_SHOULDER"]["y"],
            frame_data["LEFT_SHOULDER"]["z"]
        ))
        right_shoulder_pos = Vector((
            frame_data["RIGHT_SHOULDER"]["x"],
            frame_data["RIGHT_SHOULDER"]["y"],
            frame_data["RIGHT_SHOULDER"]["z"]
        ))
        left_hip_pos = Vector((
            frame_data["LEFT_HIP"]["x"],
            frame_data["LEFT_HIP"]["y"],
            frame_data["LEFT_HIP"]["z"]
        ))
        right_hip_pos = Vector((
            frame_data["RIGHT_HIP"]["x"],
            frame_data["RIGHT_HIP"]["y"],
            frame_data["RIGHT_HIP"]["z"]
        ))

        # Apply squish factors to joint positions
        left_shoulder_squished = Vector((
            left_shoulder_pos.x * effective_squish_factors["x"],
            left_shoulder_pos.y * effective_squish_factors["y"],
            left_shoulder_pos.z * effective_squish_factors["z"]
        ))
        right_shoulder_squished = Vector((
            right_shoulder_pos.x * effective_squish_factors["x"],
            right_shoulder_pos.y * effective_squish_factors["y"],
            right_shoulder_pos.z * effective_squish_factors["z"]
        ))
        left_hip_squished = Vector((
            left_hip_pos.x * effective_squish_factors["x"],
            left_hip_pos.y * effective_squish_factors["y"],
            left_hip_pos.z * effective_squish_factors["z"]
        ))
        right_hip_squished = Vector((
            right_hip_pos.x * effective_squish_factors["x"],
            right_hip_pos.y * effective_squish_factors["y"],
            right_hip_pos.z * effective_squish_factors["z"]
        ))

        # Calculate shoulder and hip vectors and midpoints
        shoulder_vector = right_shoulder_squished - left_shoulder_squished
        shoulder_distance = shoulder_vector.length
        shoulder_mid = (left_shoulder_squished + right_shoulder_squished) / 2
        shoulder_direction = shoulder_vector.normalized()

        hip_vector = right_hip_squished - left_hip_squished
        hip_distance = hip_vector.length
        hip_mid = (left_hip_squished + right_hip_squished) / 2
        hip_direction = hip_vector.normalized()

        # Update virtual midpoint empties
        hip_empty = bpy.data.objects.get("Main_HIP_MIDPOINT")
        shoulder_empty = bpy.data.objects.get("Main_SHOULDER_MIDPOINT")

        if hip_empty:
            hip_empty.location = hip_mid
            hip_empty.keyframe_insert(data_path="location", frame=blender_frame)

        if shoulder_empty:
            shoulder_empty.location = shoulder_mid
            shoulder_empty.keyframe_insert(data_path="location", frame=blender_frame)

        # CORRECTED HIP BONE WITH STRETCHING - Use DEF_Hip bone
        if "DEF_Hip" in pose_bones:
            hip_bone = pose_bones["DEF_Hip"]

            # Calculate the desired rotation to align with hip vector
            default_hip_forward = Vector((1, 0, 0))  # Typically points right
            rotation_to_hip = default_hip_forward.rotation_difference(hip_direction)

            # Apply the rotation
            hip_bone.rotation_quaternion = rotation_to_hip

            # STRETCH the hip bone to match the actual distance between hips
            original_hip_length = original_bone_data.get("DEF_Hip", 1.0)
            stretch_factor = hip_distance / original_hip_length if original_hip_length > 0 else 1.0

            # Apply scale to stretch the bone (X scale for length in Blender)
            hip_bone.scale = (stretch_factor, 1.0, 1.0)

            # Keyframe both rotation and scale
            hip_bone.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
            hip_bone.keyframe_insert(data_path="scale", frame=blender_frame)

            # DEBUG: Log hip stretching on first frame
            if frame_number == frame_numbers[0]:
                script_log(f"=== HIP STRETCHING DEBUG ===", force_log=True)
                script_log(f"Hip distance: {hip_distance:.4f}", force_log=True)
                script_log(f"Original hip length: {original_hip_length:.4f}", force_log=True)
                script_log(f"Hip stretch factor: {stretch_factor:.4f}", force_log=True)

        # CORRECTED SHOULDER BONE WITH STRETCHING - Use DEF_Shoulder bone
        if "DEF_Shoulder" in pose_bones:
            shoulder_bone = pose_bones["DEF_Shoulder"]

            # Calculate the desired rotation to align with shoulder vector
            default_shoulder_forward = Vector((1, 0, 0))  # Typically points right
            rotation_to_shoulder = default_shoulder_forward.rotation_difference(shoulder_direction)

            # Apply the rotation
            shoulder_bone.rotation_quaternion = rotation_to_shoulder

            # STRETCH the shoulder bone to match the actual distance between shoulders
            original_shoulder_length = original_bone_data.get("DEF_Shoulder", 1.0)
            stretch_factor = shoulder_distance / original_shoulder_length if original_shoulder_length > 0 else 1.0

            # Apply scale to stretch the bone (X scale for length in Blender)
            shoulder_bone.scale = (stretch_factor, 1.0, 1.0)

            # Keyframe both rotation and scale
            shoulder_bone.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)
            shoulder_bone.keyframe_insert(data_path="scale", frame=blender_frame)

            # DEBUG: Log shoulder stretching on first frame
            if frame_number == frame_numbers[0]:
                script_log(f"=== SHOULDER STRETCHING DEBUG ===", force_log=True)
                script_log(f"Shoulder distance: {shoulder_distance:.4f}", force_log=True)
                script_log(f"Original shoulder length: {original_shoulder_length:.4f}", force_log=True)
                script_log(f"Shoulder stretch factor: {stretch_factor:.4f}", force_log=True)

        # Update control point positions with coordinate-based squish factors
        for joint_name, control_obj in control_points.items():
            # Skip virtual midpoints - they're handled separately above
            if "MIDPOINT" in joint_name:
                continue

            if joint_name in frame_data:
                joint_data = frame_data[joint_name]
                original_position = Vector((
                    joint_data["x"],
                    joint_data["y"],
                    joint_data["z"]
                ))

                # Apply coordinate-based squish factors
                squished_position = Vector((
                    original_position.x * effective_squish_factors["x"],
                    original_position.y * effective_squish_factors["y"],
                    original_position.z * effective_squish_factors["z"]
                ))

                control_obj.location = squished_position
                control_obj.keyframe_insert(data_path="location", frame=blender_frame)

    # === RESTORE ORIGINAL CONTEXT ===
    if original_mode != 'POSE':
        bpy.ops.object.mode_set(mode=original_mode)

    # Restore original active object if different
    if original_active and original_active != armature_obj:
        bpy.ops.object.select_all(action='DESELECT')
        original_active.select_set(True)
        bpy.context.view_layer.objects.active = original_active

    # Set animation range starting from frame 1
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame_numbers[-1] + 1

    script_log(f"Animation range: {bpy.context.scene.frame_start} to {bpy.context.scene.frame_end}", force_log=True)
    script_log("COMPLETE: Bone stretching and rotation applied", force_log=True)

######################################################################################################

def verify_animation(armature_obj, control_points):
    """Verify that the animation is working correctly"""
    script_log("=== VERIFYING FRESH ANIMATION ===", force_log=True)

    # Test a few frames to ensure movement
    test_frames = [frame_numbers[0], frame_numbers[min(1, len(frame_numbers) - 1)]]

    for control_name, control_obj in list(control_points.items())[:3]:  # Test first 3 controls
        positions = {}
        for frame in test_frames:
            bpy.context.scene.frame_set(frame + 1)  # Account for frame offset
            bpy.context.view_layer.update()
            positions[frame] = control_obj.location.copy()

        if len(positions) > 1:
            movement = (positions[test_frames[1]] - positions[test_frames[0]]).length
            script_log(f"Movement {control_name}: {movement:.4f}", force_log=True)
            if movement > 0.001:
                script_log(f"✓ {control_name} is moving correctly", force_log=True)
            else:
                script_log(f"⚠ {control_name} has minimal movement", force_log=True)

######################################################################################################

def save_animated_file():
    """Save the animated Blender file"""
    try:
        from utils import get_show_path, get_scene_folder_name

        show_path = get_show_path(args.show)
        scene_folder_name = get_scene_folder_name(args.show, args.scene)
        outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
        os.makedirs(outputs_dir, exist_ok=True)

        output_blend_file = os.path.join(outputs_dir, f"{args.scene}_anim.blend")
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_file)
        script_log(f"Fresh animated figure saved to: {output_blend_file}", force_log=True)

    except Exception as e:
        script_log(f"Error saving Blender file: {e}", force_log=True)

######################################################################################################

def main_execution():
    """Main execution for animation pipeline"""
    script_log("=== 4K KID ANIMATION INNER STARTED ===\n")

    try:
        # Clear all existing animation first
        clear_existing_animation()

        # Load configuration and CURRENT data
        load_config_and_data()

        # Find existing armature and control points
        armature_obj, control_points = find_armature_and_control_points()
        if not armature_obj:
            script_log("Error: Cannot animate without armature!", force_log=True)
            return

        # Apply fresh animation
        animate_rig(armature_obj, control_points)

        # Verify animation
        verify_animation(armature_obj, control_points)

        # Save the animated file
        save_animated_file()

        script_log("=== FRESH ANIMATION PIPELINE COMPLETED SUCCESSFULLY ===", force_log=True)

    except Exception as e:
        script_log(f"ERROR in animation execution: {e}", force_log=True)
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}", force_log=True)


if __name__ == "__main__":
    main_execution()