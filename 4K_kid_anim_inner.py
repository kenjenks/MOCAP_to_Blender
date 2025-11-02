# 4K_kid_anim_inner.py (Version 4.0 - Updated for two-segment spine rig structure)

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
    print(f"FAILED to import utils: {e}")
    sys.exit(1)

# Global variables
mocap_data = {}
bone_definitions = {}
frame_numbers = []
squish_factors = {"x": 1.0, "y": 1.0, "z": 1.0}
control_point_objs = {}
figure_name = "Main"

# Bone hierarchy structure (copied from rig script)
bone_parents = {}
bone_tail_control_points = {}
bone_head_control_points = {}
bone_types = {}
def_bone_names = {}
bone_constraint_types = {}
bone_tail_landmarks = {}

# Virtual control point calculations (updated for two-segment spine)
VIRTUAL_POINT_CALCULATIONS = {
    "VIRTUAL_HIP_MIDPOINT": ["LEFT_HIP", "RIGHT_HIP"],
    "VIRTUAL_SHOULDER_MIDPOINT": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
    "VIRTUAL_SPINE_MIDPOINT": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
    "VIRTUAL_HEAD_BASE": ["NOSE", "HEAD_TOP"]
}


######################################################################################################

def clear_existing_animation():
    """Clear all existing animation data from the scene"""
    script_log("=== CLEARING EXISTING ANIMATION ===")

    # Clear animation from all objects
    for obj in bpy.data.objects:
        if obj.animation_data:
            obj.animation_data_clear()
            comment(f"Cleared animation from: {obj.name}")

    # Clear animation from pose bones
    for armature in [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']:
        if armature.animation_data:
            armature.animation_data_clear()

        # Clear pose bone animations
        for bone in armature.pose.bones:
            bone.location = Vector((0, 0, 0))
            if hasattr(bone, 'rotation_quaternion'):
                bone.rotation_quaternion = Quaternion()
            if hasattr(bone, 'rotation_euler'):
                bone.rotation_euler = Vector((0, 0, 0))
            bone.scale = Vector((1, 1, 1))

        comment(f"Cleared animation from armature: {armature.name}")

    # Clear all actions
    for action in bpy.data.actions:
        bpy.data.actions.remove(action)
    comment("Cleared all animation actions")

    # Reset scene frame range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 250
    comment("Reset scene frame range to 1-250")


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

            # Load squish factors from kid_figure_settings
            kid_settings = config.get("kid_figure_settings", {})
            squish_factors = {
                "x": kid_settings.get("x_squish_fraction", 1.0),
                "y": kid_settings.get("y_squish_fraction", 1.0),
                "z": kid_settings.get("z_squish_fraction", 1.0)
            }

        # Load CURRENT MediaPipe JSON data
        with open(INPUT_JSON_FILE, 'r') as file:
            mocap_data = json.load(file)

        script_log(f"Loaded CURRENT mocap data from: {INPUT_JSON_FILE}")
        script_log(f"Loaded kid config from: {KID_CONFIG_FILE}")
        script_log(f"Found {len(mocap_data)} frames of animation data")

        comment(
            f"Squish factors from config: X={squish_factors['x']}, Y={squish_factors['y']}, Z={squish_factors['z']}")

    except FileNotFoundError as e:
        script_log(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        script_log(f"Error: Could not decode JSON from file: {e}")
        sys.exit(1)

    # Get the frame numbers from the JSON data
    frame_numbers = sorted([int(frame) for frame in mocap_data.keys()])
    if not frame_numbers:
        script_log("Error: No frame data found in JSON file.")
        sys.exit(1)

    comment(f"Animation will span frames {frame_numbers[0]} to {frame_numbers[-1]}")


######################################################################################################

def extract_bone_hierarchy_and_controls(bone_definitions):
    """Extract clean bone relationships from new config structure - COPIED FROM RIG SCRIPT"""
    global bone_parents, bone_tail_control_points, bone_head_control_points
    global bone_types, def_bone_names, bone_constraint_types, bone_tail_landmarks

    # Initialize all globals
    bone_parents = {}
    bone_tail_control_points = {}
    bone_head_control_points = {}
    bone_types = {}
    def_bone_names = {}
    bone_constraint_types = {}
    bone_tail_landmarks = {}

    for bone_name, bone_data in bone_definitions.items():
        # Store parent
        bone_parents[bone_name] = bone_data.get("parent")

        # Store type
        bone_types[bone_name] = bone_data.get("type", "standard")

        # Store DEF bone name
        def_bone_name = bone_data.get("def_bone")
        if def_bone_name:
            def_bone_names[bone_name] = def_bone_name
        else:
            def_bone_names[bone_name] = f"DEF_{bone_name}"
            script_log(f"WARNING: No def_bone defined for {bone_name}, using {def_bone_names[bone_name]}")

        # Store constraint type
        bone_constraint_types[bone_name] = bone_data.get("constraint", "COPY_TO")

        # Store tail control point
        if "tail_control_point" in bone_data:
            bone_tail_control_points[bone_name] = bone_data["tail_control_point"]

        # Store tail landmark
        if "tail_landmark" in bone_data:
            bone_tail_landmarks[bone_name] = bone_data["tail_landmark"]

        # Store head control point (only for root bones)
        if bone_data.get("parent") is None and "head_control_point" in bone_data:
            bone_head_control_points[bone_name] = bone_data["head_control_point"]

    script_log(
        f"Extracted: {len(bone_parents)} bones, {len(def_bone_names)} DEF bones, {len(bone_constraint_types)} constraints")
    script_log(f"Control points: {len(bone_tail_control_points)} tail control points")


######################################################################################################

def get_landmark_position(frame_data, landmark_name):
    """Get direct landmark position from landmark name"""
    if landmark_name in frame_data:
        pos_data = frame_data[landmark_name]
        return Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

    script_log(f"WARNING: Landmark {landmark_name} not found in frame data")
    return Vector((0, 0, 0))


######################################################################################################

def calculate_midpoint(frame_data, landmark_names):
    """Calculate midpoint between multiple landmarks"""
    if not landmark_names:
        return Vector((0, 0, 0))

    total_vec = Vector((0, 0, 0))
    valid_points = 0

    for mp_name in landmark_names:
        if mp_name in frame_data:
            pos_data = frame_data[mp_name]
            total_vec += Vector((pos_data["x"], pos_data["y"], pos_data["z"]))
            valid_points += 1

    return total_vec / valid_points if valid_points > 0 else Vector((0, 0, 0))


######################################################################################################

def calculate_virtual_position(frame_data, virtual_point_name):
    """Simple direct lookup for virtual point calculations"""
    if virtual_point_name in VIRTUAL_POINT_CALCULATIONS:
        landmarks = VIRTUAL_POINT_CALCULATIONS[virtual_point_name]
        position = calculate_midpoint(frame_data, landmarks)
        return position

    script_log(f"ERROR: Unknown virtual point: {virtual_point_name}")
    return Vector((0, 0, 0))


######################################################################################################

def find_armature_and_control_points():
    """
    Finds the main armature object and all control point objects
    based on the bone_tail_control_points mapping
    """
    global control_point_objs

    # Find the main armature object
    armature_obj = bpy.data.objects.get(f"{figure_name}_Rig")
    if not armature_obj:
        script_log(f"Error: Armature '{figure_name}_Rig' not found in scene.")
        return None

    script_log(f"Found armature: {armature_obj.name}")

    # Clear existing control point objects
    control_point_objs.clear()

    # Find ALL control point objects including virtual points
    all_control_point_names = set()

    # Add tail control points
    for control_point_name in bone_tail_control_points.values():
        all_control_point_names.add(control_point_name)

    # Add head control points
    for control_point_name in bone_head_control_points.values():
        all_control_point_names.add(control_point_name)

    # Add virtual points from calculations
    for virtual_point_name in VIRTUAL_POINT_CALCULATIONS.keys():
        all_control_point_names.add(virtual_point_name)

    # Find all control objects
    for control_point_name in all_control_point_names:
        control_obj = bpy.data.objects.get(control_point_name)
        if control_obj:
            # Map control point to bone(s) that use it
            for bone_name, cp_name in bone_tail_control_points.items():
                if cp_name == control_point_name:
                    control_point_objs[bone_name] = control_obj
                    script_log(f"Found tail control point: {control_point_name} for bone {bone_name}")

            for bone_name, cp_name in bone_head_control_points.items():
                if cp_name == control_point_name and bone_name not in control_point_objs:
                    control_point_objs[bone_name] = control_obj
                    script_log(f"Found head control point: {control_point_name} for bone {bone_name}")

            # Track virtual points separately
            if control_point_name.startswith("VIRTUAL_") and control_point_name not in [obj.name for obj in
                                                                                        control_point_objs.values()]:
                # Create a special mapping for virtual points
                virtual_bone_name = f"VIRTUAL_{control_point_name}"
                control_point_objs[virtual_bone_name] = control_obj
                script_log(f"Found virtual control point: {control_point_name}")

    script_log(f"Found {len(control_point_objs)} total control point mappings")

    return armature_obj


######################################################################################################

def apply_squish_factors(position, squish_factors):
    """Apply squish factors to a position vector"""
    return Vector((
        position.x * squish_factors["x"],
        position.y * squish_factors["y"],
        position.z * squish_factors["z"]
    ))


######################################################################################################

def animate_rig(armature_obj):
    """Animate the rig using the new control point system with two-segment spine support"""
    script_log(f"=== APPLYING ANIMATION for {len(frame_numbers)} frames ===")

    # Animate control points starting at frame 1
    for frame_number in frame_numbers:
        blender_frame = frame_number + 1
        frame_data = mocap_data[str(frame_number)]
        bpy.context.scene.frame_set(blender_frame)

        if frame_number % 10 == 0:  # Log every 10 frames to avoid spam
            script_log(f"Animating frame {frame_number} (Blender frame {blender_frame})")

        # Animate ALL control points including virtual points
        for bone_name, control_obj in control_point_objs.items():
            control_point_name = control_obj.name

            # Determine position based on control point type
            if control_point_name.startswith("VIRTUAL_"):
                # VIRTUAL CONTROL POINT: Calculate from multiple landmarks
                position = calculate_virtual_position(frame_data, control_point_name)

            elif control_point_name.startswith("CTRL_"):
                # DIRECT CONTROL POINT: Use tail landmark from bone definition
                tail_landmark = bone_tail_landmarks.get(bone_name)
                if tail_landmark:
                    position = get_landmark_position(frame_data, tail_landmark)
                else:
                    # Fallback: try to extract landmark name from control point name
                    landmark_name = control_point_name.replace("CTRL_", "")
                    position = get_landmark_position(frame_data, landmark_name)
                    if position == Vector((0, 0, 0)):
                        # Skip if no landmark found (don't log to avoid spam)
                        continue
            else:
                # Skip unknown control point types
                continue

            # Apply squish factors
            squished_position = apply_squish_factors(position, squish_factors)

            # Set position and keyframe
            control_obj.location = squished_position
            control_obj.keyframe_insert(data_path="location", frame=blender_frame)

            # Log first few frames for debugging
            if frame_number <= 2:
                script_log(f"  {control_point_name}: {squished_position}")

        # ANIMATE CTRL_NOSE DIRECTLY (since no bone uses it as a control point)
        nose_obj = bpy.data.objects.get("CTRL_NOSE")
        if nose_obj:
            position = get_landmark_position(frame_data, "NOSE")
            squished_position = apply_squish_factors(position, squish_factors)
            nose_obj.location = squished_position
            nose_obj.keyframe_insert(data_path="location", frame=blender_frame)
            if frame_number <= 2:
                script_log(f"  CTRL_NOSE: {squished_position}")

        # Also ensure all virtual points are animated (double-check)
        for virtual_point_name in VIRTUAL_POINT_CALCULATIONS.keys():
            virtual_obj = bpy.data.objects.get(virtual_point_name)
            if virtual_obj:
                position = calculate_virtual_position(frame_data, virtual_point_name)
                squished_position = apply_squish_factors(position, squish_factors)
                virtual_obj.location = squished_position
                virtual_obj.keyframe_insert(data_path="location", frame=blender_frame)

    # Set animation range
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = frame_numbers[-1] + 1

    script_log(
        f"Animation complete: {len(frame_numbers)} frames, range {bpy.context.scene.frame_start}-{bpy.context.scene.frame_end}")

######################################################################################################

def verify_animation(armature_obj):
    """Verify that the animation is working correctly"""
    script_log("=== VERIFYING ANIMATION ===")

    # Test a few frames to ensure movement
    if len(frame_numbers) < 2:
        script_log("Not enough frames to verify animation")
        return

    test_frames = [frame_numbers[0], frame_numbers[min(10, len(frame_numbers) - 1)]]

    # Test key control points including virtual points
    key_control_points = []

    # Add some direct control points
    for bone_name in ["LeftHip", "LeftShoulder", "LeftHand"]:
        if bone_name in control_point_objs:
            key_control_points.append(control_point_objs[bone_name])

    # Add virtual control points
    for virtual_name in ["VIRTUAL_HIP_MIDPOINT", "VIRTUAL_SHOULDER_MIDPOINT", "VIRTUAL_SPINE_MIDPOINT", "VIRTUAL_HEAD_BASE"]:
        virtual_obj = bpy.data.objects.get(virtual_name)
        if virtual_obj:
            key_control_points.append(virtual_obj)

    for control_obj in key_control_points[:5]:  # Test first 5 key points
        positions = {}
        for frame in test_frames:
            bpy.context.scene.frame_set(frame + 1)
            bpy.context.view_layer.update()
            positions[frame] = control_obj.location.copy()

        if len(positions) > 1:
            movement = (positions[test_frames[1]] - positions[test_frames[0]]).length
            script_log(f"Movement {control_obj.name}: {movement:.4f}")
            if movement > 0.001:
                script_log(f"✓ {control_obj.name} is moving correctly")
            else:
                script_log(f"⚠ {control_obj.name} has minimal movement")

    # Test constraint solving by checking spine bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    spine_bones = ["DEF_LowerSpine", "DEF_UpperSpine"]
    for spine_bone_name in spine_bones:
        if spine_bone_name in armature_obj.pose.bones:
            spine_bone = armature_obj.pose.bones[spine_bone_name]
            script_log(f"Spine bone {spine_bone_name} length: {spine_bone.length:.4f}")

    bpy.ops.object.mode_set(mode='OBJECT')


######################################################################################################

def force_constraint_solve():
    """Force constraint solving to ensure proper spine stretching"""
    script_log("=== FORCING CONSTRAINT SOLVE ===")

    # Find all armatures
    for armature_obj in [obj for obj in bpy.data.objects if obj.type == 'ARMATURE']:
        script_log(f"Solving constraints for: {armature_obj.name}")

        # Go through all frames to force constraint solving
        original_frame = bpy.context.scene.frame_current

        for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end + 1, 5):  # Every 5 frames
            bpy.context.scene.frame_set(frame)
            bpy.context.view_layer.update()

        # Return to original frame
        bpy.context.scene.frame_set(original_frame)
        bpy.context.view_layer.update()


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
        script_log(f"Animated figure saved to: {output_blend_file}")

    except Exception as e:
        script_log(f"Error saving Blender file: {e}")


######################################################################################################

def main_execution():
    """Main execution for animation pipeline"""
    script_log("=== 4K KID ANIMATION INNER STARTED ===\n")

    try:
        # Clear all existing animation first
        clear_existing_animation()

        # Load configuration and data
        load_config_and_data()

        # Extract bone hierarchy and controls (same as rig script)
        extract_bone_hierarchy_and_controls(bone_definitions)

        # Find existing armature and control points
        armature_obj = find_armature_and_control_points()
        if not armature_obj:
            script_log("Error: Cannot animate without armature!")
            return

        # Apply fresh animation
        animate_rig(armature_obj)

        # Force constraint solving to ensure proper spine behavior
        force_constraint_solve()

        # Verify animation
        verify_animation(armature_obj)

        # Save the animated file
        save_animated_file()

        script_log("=== ANIMATION PIPELINE COMPLETED SUCCESSFULLY ===")

    except Exception as e:
        script_log(f"ERROR in animation execution: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main_execution()