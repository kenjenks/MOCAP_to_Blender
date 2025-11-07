# 4K_kid_inner.py (Version 19.0 - Coordinate Frame System for HipRoot and ShoulderRoot)

import bpy
import bmesh
import sys
import os
import argparse
import json
from mathutils import Vector, Matrix
import math


##########################################################################################

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments passed from 4K_kid.py"""
    parser = argparse.ArgumentParser(description='4K Kid Inner Script')
    parser.add_argument('--project-root', required=True, help='Path to project root')
    parser.add_argument('--show', required=True, help='Show name')
    parser.add_argument('--scene', required=True, help='Scene name')

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    return parser.parse_args(argv)


##########################################################################################

# Parse arguments
args = parse_arguments()

# Add project_root to sys.path so we can import utils
if args.project_root not in sys.path:
    sys.path.append(args.project_root)

# Import project utilities
try:
    from utils import script_log, comment, get_scene_config, get_processing_step_paths, get_scene_paths
except ImportError as e:
    script_log(f"FAILED to import utils: {e}")
    sys.exit(1)

# Global variables
mocap_data = {}
bone_definitions = {}
frame_numbers = []
squish_factors = {"x": 1.0, "y": 1.0, "z": 1.0}

# Control point tracking
control_point_objs = {}

# Bone hierarchy structure
bone_parents = {}
bone_tail_control_points = {}
bone_head_control_points = {}
bone_types = {}
def_bone_names = {}
bone_constraint_types = {}
bone_tail_landmarks = {}

# Virtual control point calculations
VIRTUAL_POINT_CALCULATIONS = {
    "VIRTUAL_HIP_MIDPOINT": ["LEFT_HIP", "RIGHT_HIP"],
    "VIRTUAL_SHOULDER_MIDPOINT": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
    "VIRTUAL_SPINE_MIDPOINT": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
    "VIRTUAL_HEAD_BASE": ["NOSE", "HEAD_TOP"]
}


##########################################################################################

def load_config_and_data():
    """Load configuration and mocap data using project utilities"""
    global mocap_data, bone_definitions, frame_numbers, squish_factors

    try:
        # Get scene configuration
        scene_config = get_scene_config(args.show, args.scene)

        # Get processing step paths
        step_paths = get_processing_step_paths(args.show, args.scene, "export_to_blender")

        # Input JSON is from the apply_physics step
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
            kid_settings = config.get("kid_figure_settings", {})
            squish_factors["x"] = kid_settings.get("x_squish_fraction", 1.0)
            squish_factors["y"] = kid_settings.get("y_squish_fraction", 1.0)
            squish_factors["z"] = kid_settings.get("z_squish_fraction", 1.0)

        with open(INPUT_JSON_FILE, 'r') as file:
            mocap_data = json.load(file)

        script_log(f"Loaded mocap data from: {INPUT_JSON_FILE}")
        script_log(f"Loaded kid config from: {KID_CONFIG_FILE}")

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


##########################################################################################

def extract_bone_hierarchy_and_controls(bone_definitions):
    """Extract clean bone relationships from new config structure"""
    global bone_parents, bone_tail_control_points, bone_head_control_points
    global def_bone_names, bone_constraint_types, bone_tail_landmarks

    bone_parents = {}
    bone_tail_control_points = {}
    bone_head_control_points = {}
    def_bone_names = {}
    bone_constraint_types = {}
    bone_tail_landmarks = {}

    for bone_name, bone_data in bone_definitions.items():
        # Store parent
        bone_parents[bone_name] = bone_data.get("parent")

        # Store DEF bone name
        def_bone_name = bone_data.get("def_bone")
        if def_bone_name:
            def_bone_names[bone_name] = def_bone_name
        else:
            def_bone_names[bone_name] = f"DEF_{bone_name}"

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

    script_log(f"Extracted: {len(bone_parents)} bones, {len(def_bone_names)} DEF bones")


##########################################################################################

def add_collection(name, parent_collection=None):
    """Add a new collection if it doesn't exist and link it to the scene."""
    if name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[name]

    if parent_collection:
        if new_collection in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(new_collection)
        if new_collection not in parent_collection.children:
            parent_collection.children.link(new_collection)

    return new_collection


##########################################################################################

def cleanup_existing_objects(figure_name):
    """Clean up existing objects to prevent duplicates"""
    script_log("Cleaning up existing objects to prevent duplicates...")

    objects_to_remove = []

    # Get all control point names from bone_tail_control_points
    control_point_names_list = list(bone_tail_control_points.values())

    for obj in bpy.data.objects:
        # Remove control points using bone_tail_control_points names
        if obj.name in control_point_names_list:
            objects_to_remove.append(obj)
        elif "_Sphere_" in obj.name:
            objects_to_remove.append(obj)
        elif obj.name == "Kid_Complete_Skin":
            objects_to_remove.append(obj)
        elif obj.name.startswith(f"{figure_name}_") and any(mp in obj.name for mp in
                                                            ["HIP", "SHOULDER", "ELBOW", "WRIST", "KNEE", "HEEL",
                                                             "NOSE", "HEAD_TOP", "INDEX", "FOOT_INDEX"]):
            objects_to_remove.append(obj)
        # Remove frame control points too
        elif obj.name in ["VIRTUAL_HIP_FRAME", "VIRTUAL_SHOULDER_FRAME"]:
            objects_to_remove.append(obj)

    # Remove identified objects
    for obj in objects_to_remove:
        script_log(f"Removing duplicate object: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)

    # Clean up orphaned data
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for armature in bpy.data.armatures:
        if armature.users == 0:
            bpy.data.armatures.remove(armature)

    script_log(f"Cleaned up {len(objects_to_remove)} duplicate objects")


##########################################################################################

def get_landmark_position(frame_data, landmark_name):
    """Get direct landmark position from landmark name"""
    if landmark_name in frame_data:
        pos_data = frame_data[landmark_name]
        return Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

    script_log(f"WARNING: Landmark {landmark_name} not found in frame data")
    return Vector((0, 0, 0))


##########################################################################################

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


##########################################################################################

def calculate_virtual_position(frame_data, virtual_point_name):
    """Simple direct lookup for virtual point calculations"""
    if virtual_point_name in VIRTUAL_POINT_CALCULATIONS:
        landmarks = VIRTUAL_POINT_CALCULATIONS[virtual_point_name]
        position = calculate_midpoint(frame_data, landmarks)
        return position

    script_log(f"ERROR: Unknown virtual point: {virtual_point_name}")
    return Vector((0, 0, 0))


##########################################################################################

def create_transparent_material(hex_color, alpha=0.3):
    """Create a transparent material from a hex color string."""
    mat_name = f"Material_{hex_color.replace('#', '')}_Alpha_{int(alpha * 100)}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'

    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        principled_bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        principled_bsdf.inputs['Alpha'].default_value = alpha

    return mat


##########################################################################################

def create_hip_frame_control_point(first_frame):
    """Create VIRTUAL_HIP_FRAME with proper coordinate frame calculation"""
    script_log("Creating VIRTUAL_HIP_FRAME with coordinate frame...")

    # Create empty to represent the hip coordinate frame
    bpy.ops.object.empty_add(type='SINGLE_ARROW')  # Visualize orientation
    hip_frame = bpy.context.active_object
    hip_frame.name = "VIRTUAL_HIP_FRAME"

    # Position at midpoint between hips
    left_hip_pos = get_landmark_position(first_frame, "LEFT_HIP")
    right_hip_pos = get_landmark_position(first_frame, "RIGHT_HIP")
    hip_frame.location = (left_hip_pos + right_hip_pos) / 2

    # Calculate coordinate frame axes
    hip_vector = right_hip_pos - left_hip_pos
    x_axis = hip_vector.normalized()  # X = right direction (left hip → right hip)

    # Calculate Z-axis (forward) - perpendicular to hip line and world up
    world_up = Vector((0, 1, 0))
    z_axis = x_axis.cross(world_up).normalized()

    # If z_axis is zero (hips perfectly vertical), use fallback
    if z_axis.length < 0.001:
        z_axis = Vector((0, 0, 1))
        script_log("Used fallback Z-axis for hip frame")

    # Calculate Y-axis (up) - perpendicular to both
    y_axis = z_axis.cross(x_axis).normalized()

    # Create rotation matrix and apply to empty
    rotation_matrix = Matrix([x_axis, y_axis, z_axis]).transposed()
    hip_frame.rotation_euler = rotation_matrix.to_euler()

    # Add to control points collection
    control_collection = bpy.data.collections.get("Main_ControlPoints")
    if control_collection and hip_frame.name not in control_collection.objects:
        control_collection.objects.link(hip_frame)

    script_log(f"Created VIRTUAL_HIP_FRAME at {hip_frame.location} with rotation {hip_frame.rotation_euler}")
    return hip_frame


def create_shoulder_frame_control_point(first_frame):
    """Create VIRTUAL_SHOULDER_FRAME with proper coordinate frame calculation"""
    script_log("Creating VIRTUAL_SHOULDER_FRAME with coordinate frame...")

    bpy.ops.object.empty_add(type='SINGLE_ARROW')
    shoulder_frame = bpy.context.active_object
    shoulder_frame.name = "VIRTUAL_SHOULDER_FRAME"

    # Position at midpoint between shoulders
    left_shoulder_pos = get_landmark_position(first_frame, "LEFT_SHOULDER")
    right_shoulder_pos = get_landmark_position(first_frame, "RIGHT_SHOULDER")
    shoulder_frame.location = (left_shoulder_pos + right_shoulder_pos) / 2

    # Calculate coordinate frame axes
    shoulder_vector = right_shoulder_pos - left_shoulder_pos
    x_axis = shoulder_vector.normalized()  # X = right direction

    # Calculate Z-axis (forward) - perpendicular to shoulder line and world up
    world_up = Vector((0, 1, 0))
    z_axis = x_axis.cross(world_up).normalized()

    if z_axis.length < 0.001:
        z_axis = Vector((0, 0, 1))
        script_log("Used fallback Z-axis for shoulder frame")

    # Calculate Y-axis (up) - perpendicular to both
    y_axis = z_axis.cross(x_axis).normalized()

    # Create rotation matrix and apply to empty
    rotation_matrix = Matrix([x_axis, y_axis, z_axis]).transposed()
    shoulder_frame.rotation_euler = rotation_matrix.to_euler()

    # Add to control points collection
    control_collection = bpy.data.collections.get("Main_ControlPoints")
    if control_collection and shoulder_frame.name not in control_collection.objects:
        control_collection.objects.link(shoulder_frame)

    script_log(
        f"Created VIRTUAL_SHOULDER_FRAME at {shoulder_frame.location} with rotation {shoulder_frame.rotation_euler}")
    return shoulder_frame


##########################################################################################

def create_control_points(figure_name, armature_obj=None):
    """Create all control points including the new VIRTUAL_SPINE_MIDPOINT"""
    global control_point_objs

    script_log("=== CREATING CONTROL POINTS ===")

    control_point_objs = {}
    first_frame = mocap_data.get(str(frame_numbers[0]), {})

    # Create control points collection
    control_collection = add_collection(f"{figure_name}_ControlPoints")

    # STEP 1: CREATE VIRTUAL CONTROL POINTS FIRST
    script_log("Creating virtual control points...")
    virtual_points_created = 0

    # Create ALL virtual points from VIRTUAL_POINT_CALCULATIONS
    for virtual_point_name in VIRTUAL_POINT_CALCULATIONS.keys():
        if virtual_point_name in bpy.data.objects:
            script_log(f"Using existing virtual control point: {virtual_point_name}")
            continue

        # Create empty for virtual control point
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        empty_obj = bpy.context.active_object
        empty_obj.name = virtual_point_name

        # Position from virtual calculation
        position = calculate_virtual_position(first_frame, virtual_point_name)
        empty_obj.location = position

        # Add to control points collection
        if empty_obj.name not in control_collection.objects:
            control_collection.objects.link(empty_obj)

        script_log(f"Created virtual control point: {virtual_point_name} at {position}")
        virtual_points_created += 1

    # STEP 2: Create direct landmark control points (spheres)
    script_log("Creating direct landmark control points...")
    direct_points_created = 0

    # Iterate through bone_tail_control_points directly
    for bone_name, control_point_name in bone_tail_control_points.items():
        # Skip virtual points (already created above)
        if control_point_name.startswith("VIRTUAL_"):
            if control_point_name in bpy.data.objects:
                control_obj = bpy.data.objects[control_point_name]
                control_point_objs[bone_name] = control_obj
            continue

        # Skip if already exists and tracked
        if bone_name in control_point_objs:
            script_log(f"Using existing control point: {control_point_name}")
            continue

        # Skip if object already exists in scene
        if control_point_name in bpy.data.objects:
            control_obj = bpy.data.objects[control_point_name]
            control_point_objs[bone_name] = control_obj
            script_log(f"Using existing control point: {control_point_name}")
            direct_points_created += 1
            continue

        # Create control point sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=(0, 0, 0))
        control_obj = bpy.context.active_object
        control_obj.name = control_point_name

        # Position direct control points using explicit tail landmarks
        tail_landmark = bone_tail_landmarks.get(bone_name)

        if tail_landmark and tail_landmark in first_frame:
            pos_data = first_frame[tail_landmark]
            control_obj.location = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))
            script_log(f"Positioned {control_point_name} at {control_obj.location} using landmark {tail_landmark}")

        # Apply transparent material
        transparent_mat = create_transparent_material("#FF0000", alpha=0.3)
        control_obj.data.materials.append(transparent_mat)

        # Add to system after creation and positioning
        control_point_objs[bone_name] = control_obj

        # Add to control collection
        if control_obj.name not in control_collection.objects:
            control_collection.objects.link(control_obj)

        direct_points_created += 1

    script_log(f"Created {direct_points_created} direct + {virtual_points_created} virtual control points total")
    script_log(f"Total control points tracked: {len(control_point_objs)}")

    return control_point_objs


##########################################################################################

def setup_root_bone_transform_constraints(armature_obj):
    """Use COPY_TRANSFORMS for both position AND rotation on root bones"""
    script_log("=== SETTING UP ROOT BONE TRANSFORM CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # HipRoot follows VIRTUAL_HIP_FRAME (position + rotation)
    if "DEF_HipRoot" in armature_obj.pose.bones:
        hip_root = armature_obj.pose.bones["DEF_HipRoot"]

        # Clear old constraints
        for constraint in list(hip_root.constraints):
            hip_root.constraints.remove(constraint)

        hip_frame = bpy.data.objects.get("VIRTUAL_HIP_FRAME")
        if hip_frame:
            copy_transform = hip_root.constraints.new('COPY_TRANSFORMS')
            copy_transform.target = hip_frame
            copy_transform.influence = 1.0
            constraints_added += 1
            script_log("DEF_HipRoot COPY_TRANSFORMS -> VIRTUAL_HIP_FRAME")

    # ShoulderRoot follows VIRTUAL_SHOULDER_FRAME but limits rotation around left-right axis
    if "DEF_ShoulderRoot" in armature_obj.pose.bones:
        shoulder_root = armature_obj.pose.bones["DEF_ShoulderRoot"]

        # Clear old constraints
        for constraint in list(shoulder_root.constraints):
            shoulder_root.constraints.remove(constraint)

        shoulder_frame = bpy.data.objects.get("VIRTUAL_SHOULDER_FRAME")
        if shoulder_frame:
            # COPY_TRANSFORMS to get both position and rotation from the frame
            copy_transform = shoulder_root.constraints.new('COPY_TRANSFORMS')
            copy_transform.target = shoulder_frame
            copy_transform.influence = 1.0

            # LIMIT ROTATION to block twisting around the X-axis (left-right shoulder axis)
            limit_rotation = shoulder_root.constraints.new('LIMIT_ROTATION')
            limit_rotation.use_limit_x = True
            limit_rotation.min_x = 0.0  # Lock X-axis rotation completely
            limit_rotation.max_x = 0.0
            limit_rotation.use_limit_y = False  # Allow Y-axis rotation (forward/backward tilt)
            limit_rotation.use_limit_z = False  # Allow Z-axis rotation (side-to-side tilt)
            limit_rotation.owner_space = 'LOCAL'  # Limit in the bone's local space

            constraints_added += 2
            script_log("DEF_ShoulderRoot: COPY_TRANSFORMS -> VIRTUAL_SHOULDER_FRAME with X-axis rotation locked")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"Root bone transform constraints: {constraints_added} constraints added")
    return constraints_added

##########################################################################################

def setup_virtual_frame_constraints():
    """Set up constraints for the frame control points to follow the virtual midpoints AND update rotation from landmarks"""
    script_log("=== SETTING UP VIRTUAL FRAME CONSTRAINTS ===")

    constraints_added = 0

    # VIRTUAL_HIP_FRAME should follow VIRTUAL_HIP_MIDPOINT location AND calculate rotation from hip landmarks
    hip_frame = bpy.data.objects.get("VIRTUAL_HIP_FRAME")
    hip_midpoint = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")
    left_hip = bpy.data.objects.get("CTRL_LEFT_HIP")
    right_hip = bpy.data.objects.get("CTRL_RIGHT_HIP")

    if hip_frame and hip_midpoint and left_hip and right_hip:
        # Clear existing constraints
        for constraint in list(hip_frame.constraints):
            hip_frame.constraints.remove(constraint)

        # Copy location from midpoint
        copy_loc = hip_frame.constraints.new('COPY_LOCATION')
        copy_loc.target = hip_midpoint
        copy_loc.influence = 1.0

        # Use DAMPED_TRACK to make the frame point from left hip to right hip
        damped_track = hip_frame.constraints.new('DAMPED_TRACK')
        damped_track.target = right_hip
        damped_track.track_axis = 'TRACK_X'  # X-axis should point from left to right hip

        constraints_added += 2
        script_log("VIRTUAL_HIP_FRAME: COPY_LOCATION -> VIRTUAL_HIP_MIDPOINT, DAMPED_TRACK -> CTRL_RIGHT_HIP")

    # VIRTUAL_SHOULDER_FRAME should follow VIRTUAL_SHOULDER_MIDPOINT location AND calculate rotation from shoulder landmarks
    shoulder_frame = bpy.data.objects.get("VIRTUAL_SHOULDER_FRAME")
    shoulder_midpoint = bpy.data.objects.get("VIRTUAL_SHOULDER_MIDPOINT")
    left_shoulder = bpy.data.objects.get("CTRL_LEFT_SHOULDER")
    right_shoulder = bpy.data.objects.get("CTRL_RIGHT_SHOULDER")

    if shoulder_frame and shoulder_midpoint and left_shoulder and right_shoulder:
        # Clear existing constraints
        for constraint in list(shoulder_frame.constraints):
            shoulder_frame.constraints.remove(constraint)

        # Copy location from midpoint
        copy_loc = shoulder_frame.constraints.new('COPY_LOCATION')
        copy_loc.target = shoulder_midpoint
        copy_loc.influence = 1.0

        # Use DAMPED_TRACK to make the frame point from left shoulder to right shoulder
        damped_track = shoulder_frame.constraints.new('DAMPED_TRACK')
        damped_track.target = right_shoulder
        damped_track.track_axis = 'TRACK_X'  # X-axis should point from left to right shoulder

        constraints_added += 2
        script_log(
            "VIRTUAL_SHOULDER_FRAME: COPY_LOCATION -> VIRTUAL_SHOULDER_MIDPOINT, DAMPED_TRACK -> CTRL_RIGHT_SHOULDER")

    script_log(f"Virtual frame constraints: {constraints_added} constraints added")
    return constraints_added

##########################################################################################

def setup_virtual_point_constraints():
    """Set up constraints for all virtual points including the new spine midpoint"""
    script_log("=== SETTING UP VIRTUAL POINT CONSTRAINTS ===")

    constraints_added = 0

    # Setup VIRTUAL_HIP_MIDPOINT constraint
    hip_midpoint = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")
    left_hip = bpy.data.objects.get("CTRL_LEFT_HIP")
    right_hip = bpy.data.objects.get("CTRL_RIGHT_HIP")

    if hip_midpoint and left_hip and right_hip:
        for constraint in list(hip_midpoint.constraints):
            hip_midpoint.constraints.remove(constraint)

        copy_left = hip_midpoint.constraints.new('COPY_LOCATION')
        copy_left.target = left_hip
        copy_left.use_offset = False
        copy_left.influence = 0.5

        copy_right = hip_midpoint.constraints.new('COPY_LOCATION')
        copy_right.target = right_hip
        copy_right.use_offset = False
        copy_right.influence = 0.5

        constraints_added += 2
        script_log("Set up VIRTUAL_HIP_MIDPOINT constraints")

    # Setup VIRTUAL_SHOULDER_MIDPOINT constraint
    shoulder_midpoint = bpy.data.objects.get("VIRTUAL_SHOULDER_MIDPOINT")
    left_shoulder = bpy.data.objects.get("CTRL_LEFT_SHOULDER")
    right_shoulder = bpy.data.objects.get("CTRL_RIGHT_SHOULDER")

    if shoulder_midpoint and left_shoulder and right_shoulder:
        for constraint in list(shoulder_midpoint.constraints):
            shoulder_midpoint.constraints.remove(constraint)

        copy_left = shoulder_midpoint.constraints.new('COPY_LOCATION')
        copy_left.target = left_shoulder
        copy_left.use_offset = False
        copy_left.influence = 0.5

        copy_right = shoulder_midpoint.constraints.new('COPY_LOCATION')
        copy_right.target = right_shoulder
        copy_right.use_offset = False
        copy_right.influence = 0.5

        constraints_added += 2
        script_log("Set up VIRTUAL_SHOULDER_MIDPOINT constraints")

    # Setup VIRTUAL_SPINE_MIDPOINT constraint - midpoint between hips and shoulders
    spine_midpoint = bpy.data.objects.get("VIRTUAL_SPINE_MIDPOINT")
    if spine_midpoint:
        for constraint in list(spine_midpoint.constraints):
            spine_midpoint.constraints.remove(constraint)

        # 50% influence from hip midpoint
        if hip_midpoint:
            copy_hip = spine_midpoint.constraints.new('COPY_LOCATION')
            copy_hip.target = hip_midpoint
            copy_hip.use_offset = False
            copy_hip.influence = 0.5

        # 50% influence from shoulder midpoint
        if shoulder_midpoint:
            copy_shoulder = spine_midpoint.constraints.new('COPY_LOCATION')
            copy_shoulder.target = shoulder_midpoint
            copy_shoulder.use_offset = False
            copy_shoulder.influence = 0.5

        constraints_added += 2
        script_log("Set up VIRTUAL_SPINE_MIDPOINT constraints")

    # Setup VIRTUAL_HEAD_BASE constraint - FIXED: Follow both HEAD_TOP and NOSE
    head_base = bpy.data.objects.get("VIRTUAL_HEAD_BASE")
    head_top = bpy.data.objects.get("CTRL_HEAD_TOP")
    nose = bpy.data.objects.get("CTRL_NOSE")

    if head_base:
        for constraint in list(head_base.constraints):
            head_base.constraints.remove(constraint)

        # Follow HEAD_TOP with some influence
        if head_top:
            copy_head = head_base.constraints.new('COPY_LOCATION')
            copy_head.target = head_top
            copy_head.use_offset = False
            copy_head.influence = 0.7  # 70% influence from head top
            constraints_added += 1
            script_log("VIRTUAL_HEAD_BASE: 70% influence from CTRL_HEAD_TOP")

        # Also follow NOSE to position head base properly
        if nose:
            copy_nose = head_base.constraints.new('COPY_LOCATION')
            copy_nose.target = nose
            copy_nose.use_offset = False
            copy_nose.influence = 0.3  # 30% influence from nose
            constraints_added += 1
            script_log("VIRTUAL_HEAD_BASE: 30% influence from CTRL_NOSE")

        if not head_top and not nose:
            script_log("WARNING: VIRTUAL_HEAD_BASE has no targets (missing CTRL_HEAD_TOP and CTRL_NOSE)")

    bpy.context.view_layer.update()
    script_log(f"Virtual point constraints: {constraints_added} constraints added")
    return constraints_added

##########################################################################################

def create_kid_rig(figure_name):
    """Create kid rig with hierarchy-driven spine"""

    ###########################################################################
    # COORDINATE FRAME SYSTEM ARCHITECTURE
    ###########################################################################
    # Do not delete these comments.
    #
    # Coordinate Frame Calculation for HipRoot and ShoulderRoot
    # ---------------------------------------------------------------------
    # HipRoot and ShoulderRoot use COPY_TRANSFORMS constraints to follow
    # VIRTUAL_HIP_FRAME and VIRTUAL_SHOULDER_FRAME, which calculate proper
    # coordinate frames from landmark positions.
    #
    # This provides both position AND rotation to the root bones, eliminating
    # the need for separate hip/shoulder constraints.
    #
    # Key Benefits:
    # - Proper pelvis and shoulder rotation from landmark data
    # - Clean inheritance for all child bones
    # - Eliminates failed STRETCH_TO constraints on hips/shoulders
    #
    ###########################################################################

    script_log(f"=== Creating kid rig: {figure_name} ===")

    def calculate_bone_order(bone_definitions):
        """Calculate bone creation order from parent relationships"""
        script_log("Calculating bone order from hierarchy...")

        children = {}
        root_bones = []

        for bone_name, bone_data in bone_definitions.items():
            parent_name = bone_data.get("parent")

            if parent_name:
                if parent_name not in children:
                    children[parent_name] = []
                children[parent_name].append(bone_name)
                script_log(f"  {bone_name} -> {parent_name}")
            else:
                root_bones.append(bone_name)
                script_log(f"  {bone_name} -> ROOT")

        # Depth-first traversal to build order
        bone_order = []

        def traverse_bone(bone_name):
            bone_order.append(bone_name)
            script_log(f"  Traversing: {bone_name}")
            if bone_name in children:
                for child_name in children[bone_name]:
                    traverse_bone(child_name)

        # Start with root bones in specific order for stability
        preferred_order = ["LowerSpine"]
        for preferred_bone in preferred_order:
            if preferred_bone in root_bones:
                traverse_bone(preferred_bone)
                root_bones.remove(preferred_bone)

        # Add any remaining root bones
        for root_bone in root_bones:
            traverse_bone(root_bone)

        script_log(f"Final bone order: {bone_order}")
        return bone_order

    # Calculate bone order
    bone_order = calculate_bone_order(bone_definitions)

    # Create armature data and object
    armature_obj_name = f"{figure_name}_Rig"

    # Clean up existing armature completely
    if armature_obj_name in bpy.data.objects:
        script_log(f"Removing existing armature: {armature_obj_name}")
        bpy.data.objects.remove(bpy.data.objects[armature_obj_name], do_unlink=True)

    # Also clean up any armature data
    armature_data_name = f"{figure_name}_Armature"
    if armature_data_name in bpy.data.armatures:
        script_log(f"Removing existing armature data: {armature_data_name}")
        bpy.data.armatures.remove(armature_data_name)

    # Create new armature data and object
    armature_data = bpy.data.armatures.new(armature_data_name)
    armature_obj = bpy.data.objects.new(armature_obj_name, armature_data)
    script_log(f"Created new armature: {armature_obj_name}")

    # Link to main collection
    main_collection = bpy.data.collections.get(f"{figure_name}_MainFigure")
    if not main_collection:
        main_collection = add_collection(f"{figure_name}_MainFigure")
    main_collection.objects.link(armature_obj)

    # Make armature active and enter Edit Mode to add bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Dictionary to store bone references
    def_bones_dict = {}

    # Get first frame data for initial bone positions
    first_frame = mocap_data.get(str(frame_numbers[0]), {})

    ###########################################################################
    # CREATE PROGRAMMATIC ROOT BONES FIRST
    ###########################################################################

    script_log("Creating programmatic root bones...")

    # Create HipRoot bone (programmatic only - not in config)
    hip_root_bone = armature_data.edit_bones.new("DEF_HipRoot")
    hip_root_pos = calculate_virtual_position(first_frame, "VIRTUAL_HIP_MIDPOINT")
    hip_root_bone.head = hip_root_pos
    hip_root_bone.tail = hip_root_pos + Vector((0, 0.1, 0))  # Small bone for visibility
    def_bones_dict["HipRoot"] = hip_root_bone
    script_log(f"Created DEF_HipRoot at {hip_root_pos}")

    # Create ShoulderRoot bone (programmatic only - not in config)
    shoulder_root_bone = armature_data.edit_bones.new("DEF_ShoulderRoot")
    shoulder_root_pos = calculate_virtual_position(first_frame, "VIRTUAL_SHOULDER_MIDPOINT")
    shoulder_root_bone.head = shoulder_root_pos
    shoulder_root_bone.tail = shoulder_root_pos + Vector((0, 0.1, 0))
    def_bones_dict["ShoulderRoot"] = shoulder_root_bone
    script_log(f"Created DEF_ShoulderRoot at {shoulder_root_pos}")

    # Hard-coded parenting overrides for root bone hierarchy
    ROOT_BONE_PARENTING = {
        "LeftHip": "HipRoot",
        "RightHip": "HipRoot",
        "LowerSpine": "HipRoot",
        "LeftShoulder": "ShoulderRoot",
        "RightShoulder": "ShoulderRoot",
        "UpperSpine": "ShoulderRoot",
        "Neck": "ShoulderRoot"
    }

    # ===========================================================================
    # CREATE ALL BONES INCLUDING SPINE IN MAIN LOOP
    # ===========================================================================
    script_log("Creating DEF bones for skin deformation and animation...")

    # Create bones in proper order to ensure parents exist first
    creation_order = [
        "LeftHip", "RightHip", "LowerSpine", "LeftShoulder", "RightShoulder", "UpperSpine",
        "Neck", "Head", "LeftUpperArm", "RightUpperArm", "LeftForearm", "RightForearm",
        "LeftHand", "RightHand", "LeftThigh", "RightThigh", "LeftShin", "RightShin",
        "LeftFoot", "RightFoot"
    ]

    for bone_name in creation_order:
        if bone_name not in bone_definitions:
            continue

        bone_data = bone_definitions[bone_name]

        # Get DEF bone name from centralized mapping
        def_bone_name = def_bone_names.get(bone_name)
        if not def_bone_name:
            script_log(f"WARNING: No DEF bone name for {bone_name}, skipping")
            continue

        # Get tail position from control point or landmark
        tail_control_point = bone_data["tail_control_point"]
        if tail_control_point.startswith("VIRTUAL_"):
            tail_pos = calculate_virtual_position(first_frame, tail_control_point)
        else:
            # Use explicit tail landmark if available
            tail_landmark = bone_tail_landmarks.get(bone_name)
            if tail_landmark:
                tail_pos = get_landmark_position(first_frame, tail_landmark)
            else:
                tail_pos = get_landmark_position(first_frame, tail_control_point[5:])

        # Determine head position based on bone type and parenting
        head_pos = Vector((0, 0, 0))

        # Use our hard-coded parenting if defined, otherwise use config
        if bone_name in ROOT_BONE_PARENTING:
            parent_name = ROOT_BONE_PARENTING[bone_name]
        else:
            parent_name = bone_data.get("parent")

        # HEAD POSITIONING LOGIC - USING ROOT BONES FOR HIP/SHOULDER REGIONS
        ###########################################################################

        # HIP REGION BONES: Head at HipRoot position
        if bone_name in ["LeftHip", "RightHip", "LowerSpine"]:
            if "HipRoot" in def_bones_dict:
                head_pos = def_bones_dict["HipRoot"].head
                script_log(f"HIP REGION: Positioning {bone_name} head at HipRoot: {head_pos}")

        # SHOULDER REGION BONES: Head at ShoulderRoot position
        elif bone_name in ["LeftShoulder", "RightShoulder", "UpperSpine", "Neck"]:
            if "ShoulderRoot" in def_bones_dict:
                head_pos = def_bones_dict["ShoulderRoot"].head
                script_log(f"SHOULDER REGION: Positioning {bone_name} head at ShoulderRoot: {head_pos}")

        # HEAD BONE: Use VIRTUAL_HEAD_BASE as before
        elif bone_name == "Head":
            head_control_point = "VIRTUAL_HEAD_BASE"
            head_pos = calculate_virtual_position(first_frame, head_control_point)
            script_log(f"HEAD: Positioning {bone_name} head at {head_control_point}: {head_pos}")

        # NORMAL CASE: Use parent's tail for connection
        elif parent_name and parent_name in def_bones_dict:
            parent_bone = def_bones_dict[parent_name]
            head_pos = parent_bone.tail
            script_log(f"Positioning {bone_name} head at parent {parent_name} tail: {head_pos}")

        else:
            # Root bone - use head control point if specified
            if "head_control_point" in bone_data:
                head_cp = bone_data["head_control_point"]
                if head_cp.startswith("VIRTUAL_"):
                    head_pos = calculate_virtual_position(first_frame, head_cp)
                    script_log(f"ROOT: Positioning {bone_name} head at {head_cp}: {head_pos}")

        # Create DEF bone
        def_bone = armature_data.edit_bones.new(def_bone_name)
        def_bone.head = head_pos
        def_bone.tail = tail_pos

        # Store bone reference
        def_bones_dict[bone_name] = def_bone

        script_log(f"Created {def_bone_name}: head={head_pos}, tail={tail_pos}")

    # ===========================================================================
    # SET UP PARENT RELATIONSHIPS - USING ROOT BONE HIERARCHY
    # ===========================================================================
    script_log("Setting up bone parent relationships with root bone hierarchy...")

    # Parent all bones
    for bone_name in creation_order:
        if bone_name not in bone_definitions:
            continue

        # Use our hard-coded parenting if defined, otherwise use config
        if bone_name in ROOT_BONE_PARENTING:
            parent_name = ROOT_BONE_PARENTING[bone_name]
        else:
            bone_data = bone_definitions[bone_name]
            parent_name = bone_data.get("parent")

        if not parent_name:
            continue

        # SPECIAL HANDLING FOR HIP REGION: Parent to HipRoot
        if bone_name in ["LeftHip", "RightHip", "LowerSpine"]:
            if "HipRoot" in def_bones_dict and bone_name in def_bones_dict:
                parent_bone = def_bones_dict["HipRoot"]
                child_bone = def_bones_dict[bone_name]

                child_bone.parent = parent_bone
                child_bone.use_connect = False

                script_log(f"ROOT HIERARCHY: Parented {bone_name} to HipRoot")

        # SPECIAL HANDLING FOR SHOULDER REGION: Parent to ShoulderRoot
        elif bone_name in ["LeftShoulder", "RightShoulder", "UpperSpine", "Neck"]:
            if "ShoulderRoot" in def_bones_dict and bone_name in def_bones_dict:
                parent_bone = def_bones_dict["ShoulderRoot"]
                child_bone = def_bones_dict[bone_name]

                child_bone.parent = parent_bone
                child_bone.use_connect = False

                script_log(f"ROOT HIERARCHY: Parented {bone_name} to ShoulderRoot")

        # NORMAL PARENTING for other bones
        elif parent_name in def_bones_dict and bone_name in def_bones_dict:
            parent_bone = def_bones_dict[parent_name]
            child_bone = def_bones_dict[bone_name]

            child_bone.parent = parent_bone
            child_bone.use_connect = False

            # Position at parent's tail
            child_bone.head = parent_bone.tail
            script_log(f"Parented {bone_name} to {parent_name}")

    script_log("Bone parenting with root hierarchy completed")

    # Exit Edit Mode first
    bpy.ops.object.mode_set(mode='OBJECT')

    # Force scene update to ensure bones are registered
    bpy.context.view_layer.update()

    return armature_obj


##########################################################################################

def setup_two_segment_spine_constraints(armature_obj, figure_name):
    """Setup simplified constraints for hierarchy-driven spine"""

    ###########################################################################
    # SPINE SYSTEM WITH COORDINATE FRAME SUPPORT
    ###########################################################################
    #
    # UPDATED: Spine now inherits proper rotation from coordinate frame roots
    # ----------------------------------------------------------------------
    # LowerSpine inherits rotation from HipRoot (which gets it from VIRTUAL_HIP_FRAME)
    # UpperSpine inherits rotation from ShoulderRoot (which gets it from VIRTUAL_SHOULDER_FRAME)
    #
    # This creates natural spinal curvature with proper pelvis and shoulder rotation
    #
    ###########################################################################

    script_log("=== SETTING UP HIERARCHY-DRIVEN SPINE CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # Get the virtual points
    spine_mid_target = bpy.data.objects.get("VIRTUAL_SPINE_MIDPOINT")

    # ===========================================================================
    # LOWER SPINE CONSTRAINTS: Tail stretches to virtual spine midpoint
    # ===========================================================================
    if "DEF_LowerSpine" in armature_obj.pose.bones and spine_mid_target:
        lower_spine = armature_obj.pose.bones["DEF_LowerSpine"]

        # Clear existing constraints
        for constraint in list(lower_spine.constraints):
            lower_spine.constraints.remove(constraint)

        # HEAD POSITION & ROTATION: Handled by parenting to HipRoot (with coordinate frame)
        # TAIL: STRETCH_TO to virtual spine midpoint
        stretch_to = lower_spine.constraints.new('STRETCH_TO')
        stretch_to.target = spine_mid_target
        stretch_to.influence = 1.0
        constraints_added += 1
        script_log("DEF_LowerSpine STRETCH_TO -> VIRTUAL_SPINE_MIDPOINT (tail stretches to center)")

    # ===========================================================================
    # UPPER SPINE CONSTRAINTS: Tail stretches to virtual spine midpoint
    # ===========================================================================
    if "DEF_UpperSpine" in armature_obj.pose.bones and spine_mid_target:
        upper_spine = armature_obj.pose.bones["DEF_UpperSpine"]

        # Clear existing constraints
        for constraint in list(upper_spine.constraints):
            upper_spine.constraints.remove(constraint)

        # HEAD POSITION & ROTATION: Handled by parenting to ShoulderRoot (with coordinate frame)
        # TAIL: STRETCH_TO to virtual spine midpoint
        stretch_to = upper_spine.constraints.new('STRETCH_TO')
        stretch_to.target = spine_mid_target
        stretch_to.influence = 1.0
        constraints_added += 1
        script_log("DEF_UpperSpine STRETCH_TO -> VIRTUAL_SPINE_MIDPOINT (tail stretches to center)")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"Hierarchy-driven spine constraints: {constraints_added} constraints added")
    return constraints_added


##########################################################################################

def setup_direct_constraints(armature_obj, figure_name="Main"):
    """Set up constraints for all non-spine bones, SKIPPING hips and shoulders"""
    script_log("=== SETTING UP DIRECT CONSTRAINTS (SKIPPING HIPS/SHOULDERS) ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # Apply constraints for all bones in bone_definitions EXCEPT hips and shoulders
    for bone_name, bone_data in bone_definitions.items():
        # SKIP hips and shoulders - they get position and rotation from root bones now
        if bone_name in ["LeftHip", "RightHip", "LeftShoulder", "RightShoulder"]:
            script_log(f"SKIPPING constraints for {bone_name} - using coordinate frame system")
            continue

        def_bone_name = def_bone_names.get(bone_name)
        constraint_type = bone_constraint_types.get(bone_name)

        if not def_bone_name or def_bone_name not in armature_obj.pose.bones:
            continue

        bone = armature_obj.pose.bones[def_bone_name]
        tail_control_point = bone_tail_control_points.get(bone_name)

        if tail_control_point:
            # Find target object
            target_obj = None
            for obj in bpy.data.objects:
                if obj.name == tail_control_point:
                    target_obj = obj
                    break

            if target_obj:
                # Check if STRETCH_TO constraint already exists
                has_stretch_to = any(constraint.type == 'STRETCH_TO' for constraint in bone.constraints)

                if not has_stretch_to:
                    # Use STRETCH_TO constraint for the TAIL
                    stretch_to = bone.constraints.new('STRETCH_TO')
                    stretch_to.target = target_obj
                    stretch_to.influence = 1.0

                    constraints_added += 1
                    script_log(f"{def_bone_name} STRETCH_TO → {tail_control_point}")

    # ADD NECK ROTATION CONSTRAINT TO FOLLOW SHOULDERS
    if "DEF_Neck" in armature_obj.pose.bones:
        neck_bone = armature_obj.pose.bones["DEF_Neck"]
        shoulder_midpoint_obj = bpy.data.objects.get("VIRTUAL_SHOULDER_MIDPOINT")

        if shoulder_midpoint_obj:
            # Check if constraint already exists
            has_copy_rot = any(constraint.type == 'COPY_ROTATION' for constraint in neck_bone.constraints)

            if not has_copy_rot:
                # Add COPY_ROTATION constraint to make neck follow shoulder orientation
                copy_rot = neck_bone.constraints.new('COPY_ROTATION')
                copy_rot.target = shoulder_midpoint_obj
                copy_rot.use_offset = True  # Use offset to maintain local rotation
                copy_rot.influence = 0.7  # Partial influence to allow some independent neck movement
                copy_rot.mix_mode = 'ADD'  # Add to existing rotation

                constraints_added += 1
                script_log(f"DEF_Neck COPY_ROTATION → VIRTUAL_SHOULDER_MIDPOINT (influence: 0.7)")

    # ADD HEAD ROTATION CONSTRAINT TO FOLLOW NOSE LANDMARK
    if "DEF_Head" in armature_obj.pose.bones:
        head_bone = armature_obj.pose.bones["DEF_Head"]

        # Check if constraint already exists
        has_damped_track = any(constraint.type == 'DAMPED_TRACK' for constraint in head_bone.constraints)

        if not has_damped_track:
            # Create or get NOSE control point object
            nose_obj = bpy.data.objects.get("CTRL_NOSE")
            if not nose_obj:
                # Switch to OBJECT mode to create the empty
                bpy.ops.object.mode_set(mode='OBJECT')

                # Create NOSE control point if it doesn't exist
                script_log("Creating NOSE control point for head rotation...")
                first_frame = mocap_data.get(str(frame_numbers[0]), {})
                if "NOSE" in first_frame:
                    pos_data = first_frame["NOSE"]
                    nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

                    # Create empty for nose control point
                    bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
                    nose_obj = bpy.context.active_object
                    nose_obj.name = "CTRL_NOSE"

                    # Add to control points collection
                    control_collection = bpy.data.collections.get(f"{figure_name}_ControlPoints")
                    if control_collection and nose_obj.name not in control_collection.objects:
                        control_collection.objects.link(nose_obj)

                    script_log(f"Created CTRL_NOSE at {nose_pos}")

                # Switch back to POSE mode to continue adding constraints
                bpy.context.view_layer.objects.active = armature_obj
                bpy.ops.object.mode_set(mode='POSE')

            if nose_obj:
                # Add DAMPED_TRACK constraint to make head aim toward nose control point
                damped_track = head_bone.constraints.new('DAMPED_TRACK')
                damped_track.target = nose_obj
                damped_track.track_axis = 'TRACK_Y'  # Adjust based on head bone orientation
                damped_track.influence = 1.0

                constraints_added += 1
                script_log(f"DEF_Head DAMPED_TRACK → CTRL_NOSE (head follows nose orientation)")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"Direct constraints: {constraints_added} constraints added")
    return constraints_added


##########################################################################################

def align_bones_with_control_points(armature_obj, figure_name):
    """Align bone positions with control points"""
    script_log("=== ALIGNING BONES WITH CONTROL POINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bones_aligned = 0

    for bone_name, control_obj in control_point_objs.items():
        def_bone_name = def_bone_names.get(bone_name)

        if not def_bone_name or def_bone_name not in armature_obj.data.edit_bones:
            continue

        bone = armature_obj.data.edit_bones[def_bone_name]
        target_pos = control_obj.location

        # All control points align to bone TAIL
        bone_direction = (bone.tail - bone.head).normalized()
        bone_length = bone.length

        bone.tail = target_pos
        bone.head = target_pos - (bone_direction * bone_length)

        script_log(f"Aligned {def_bone_name} TAIL to {control_obj.name} at {target_pos}")
        bones_aligned += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    script_log(f"Bone alignment complete: {bones_aligned} bones aligned")
    return bones_aligned


##########################################################################################

def create_kid_flesh(armature_obj, figure_name):
    """
    Create a vertex cloud flesh using the manual workflow approach:
    Step 1. Create vertices in rings
    Step 2. Create all edges and faces in one go
    """
    script_log(f"=== CREATING VERTEX CLOUD FLESH ===")

    try:
        # Create vertex cloud mesh
        mesh = bpy.data.meshes.new("Kid_Complete_Skin")
        cloud_obj = bpy.data.objects.new("Kid_Complete_Skin", mesh)

        # Use the same collection that the armature is in
        main_collection = bpy.data.collections.get(f"{figure_name}_MainFigure")
        if main_collection:
            main_collection.objects.link(cloud_obj)
            script_log(f"✓ Linking flesh mesh to existing MainFigure collection")
        else:
            script_log(f"ERROR: Can't find MainFigure collection for flesh mesh")
            return None

        # --- MANDATORY FIX START ---
        # 1. Set the armature as the object parent of the mesh (Fixes stretching on translation)
        cloud_obj.parent = armature_obj
        script_log(f"✓ Parenting flesh mesh to armature object: {armature_obj.name}")

        # 2. Add the Armature Modifier (Ensures the mesh is deformed by the bones)
        armature_mod = cloud_obj.modifiers.new(name='Armature', type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True  # Needs to be here for the vertex groups created below
        script_log(f"✓ Added Armature Modifier targeting: {armature_obj.name}")
        # --- MANDATORY FIX END ---

        # Store all vertices
        all_vertices = []
        vertex_groups = {}  # Track which bone each vertex belongs to

        # Store ring information for edge and face creation
        ring_data_by_bone = {}  # {bone_name: [ring1_data, ring2_data, ...]}

        # List to store hip horizontal ring vertices for use in PART C2 and PART E
        hip_horizontal_ring_vertices = []

        def is_this_a_hip_horizontal_vertex(def_bone_name, vertex_index_in_ring):
            """Check if this vertex should be part of the horizontal hip ring"""
            if not def_bone_name.startswith("DEF_"):
                return False

            bone_name = def_bone_name[4:]  # Remove "DEF_" prefix

            if bone_name not in ["LeftHip", "RightHip"]:
                return False

            # Get the parent_best_offset for this hip bone
            bone_data = bone_definitions.get(bone_name, {})
            parent_offset = bone_data.get("parent_best_offset", 0)

            # Check if this vertex is at key positions around the ring
            ring_size = 12  # Assuming consistent ring size
            key_positions = [
                parent_offset,  # Primary connection point
                (parent_offset + 6) % ring_size  # Opposite side
            ]

            return vertex_index_in_ring in key_positions

        script_log("Generating vertex rings from bone shapes...")

        # Access bones through the armature data
        armature_data = armature_obj.data

        # STEP 1: Create vertices in rings for each bone
        for bone_name, bone_data in bone_definitions.items():
            if "shapes" not in bone_data or not bone_data["shapes"]:
                script_log(f"Skipping bone without shapes: {bone_name}")
                continue

            script_log(f"Creating vertex rings for bone: {bone_name}")

            # Get bone data from armature
            def_bone_name = f"DEF_{bone_name}"

            bone = None
            if def_bone_name in armature_data.bones:
                bone = armature_data.bones[def_bone_name]
            else:
                script_log(f"  Bone {def_bone_name} not found in armature data, skipping")
                continue

            # Get bone data
            bone_matrix = armature_obj.matrix_world @ bone.matrix_local
            bone_length = (bone.tail_local - bone.head_local).length

            # Store rings for this bone
            bone_rings = []

            # Create vertices for each shape ring
            for shape_idx, shape in enumerate(bone_data["shapes"]):
                position_along_bone = shape["position"]
                radius = shape["radius"]
                flatten_scale = shape["flatten_scale"]

                # Calculate position along bone in local space
                local_bone_pos = Vector((0, position_along_bone * bone_length, 0))

                # Create a ring of vertices around this position
                ring_vertex_count = 12
                ring_vertices = []
                ring_vertex_indices = []

                for i in range(ring_vertex_count):
                    angle = (2 * math.pi * i) / ring_vertex_count

                    # Calculate vertex offset in bone's local space
                    local_x = math.cos(angle) * radius * flatten_scale[0]
                    local_z = math.sin(angle) * radius * flatten_scale[1]

                    # Create local vertex offset
                    local_offset = Vector((local_x, 0, local_z))

                    # Transform to world space
                    world_vertex = bone_matrix @ (local_bone_pos + local_offset)

                    ring_vertices.append(world_vertex)
                    vertex_index = len(all_vertices)
                    all_vertices.append(world_vertex)
                    ring_vertex_indices.append(vertex_index)
                    vertex_groups[vertex_index] = def_bone_name

                    # Check if this vertex belongs to the horizontal hip ring
                    if is_this_a_hip_horizontal_vertex(def_bone_name, i):
                        hip_horizontal_ring_vertices.append({
                            'vertex_index': vertex_index,
                            'position': world_vertex,
                            'bone': def_bone_name,
                            'ring_position': i
                        })

                bone_rings.append({
                    'position': position_along_bone,
                    'vertex_indices': ring_vertex_indices,
                    'world_pos': bone_matrix @ local_bone_pos
                })

            ring_data_by_bone[bone_name] = bone_rings

        # Create the initial mesh with just vertices
        mesh.from_pydata(all_vertices, [], [])
        mesh.update()

        script_log(f"Created {len(all_vertices)} total vertices")

        # STEP 2: Create ALL edges and faces in one go
        script_log("STEP 2: Creating all edges and faces...")

        # Use bmesh for direct edge and face creation
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        total_edges_created = 0
        total_faces_created = 0

        # --------------------------------------------------------------------------------
        # PART A: Create edges and faces between rings on same bone
        # --------------------------------------------------------------------------------
        script_log("PART A: Creating edges and faces between rings on same bone...")

        for bone_name, rings in ring_data_by_bone.items():
            if len(rings) < 2:
                continue

            sorted_rings = sorted(rings, key=lambda r: r['position'])

            for i in range(len(sorted_rings) - 1):
                current_ring = sorted_rings[i]
                next_ring = sorted_rings[i + 1]

                current_indices = current_ring['vertex_indices']
                next_indices = next_ring['vertex_indices']

                ring_size = min(len(current_indices), len(next_indices))

                for j in range(ring_size):
                    # VERTEX ORDER: v1 -> v4 -> v3 -> v2
                    v1_idx = current_indices[j]  # Current ring, vertex j
                    v4_idx = next_indices[j]  # Next ring, vertex j
                    v3_idx = next_indices[(j + 1) % ring_size]  # Next ring, vertex j+1
                    v2_idx = current_indices[(j + 1) % ring_size]  # Current ring, vertex j+1

                    v1 = bm.verts[v1_idx]
                    v2 = bm.verts[v2_idx]
                    v3 = bm.verts[v3_idx]
                    v4 = bm.verts[v4_idx]

                    # Create edges if they don't exist
                    edges_to_create = [(v1, v4), (v4, v3), (v3, v2), (v2, v1)]
                    edges_created = 0

                    for edge_verts in edges_to_create:
                        edge_exists = False
                        for edge in edge_verts[0].link_edges:
                            if edge_verts[1] in edge.verts:
                                edge_exists = True
                                break
                        if not edge_exists:
                            try:
                                bm.edges.new(edge_verts)
                                edges_created += 1
                                total_edges_created += 1
                            except Exception as e:
                                pass  # Edge might already exist from adjacent face

                    # Create the face using contextual_create (F-key equivalent)
                    try:
                        # Deselect all first
                        for edge in bm.edges:
                            edge.select = False

                        # Find the edges we want to use for face creation
                        edges_for_face = []
                        for edge_pair in [(v1, v4), (v4, v3), (v3, v2), (v2, v1)]:
                            for edge in edge_pair[0].link_edges:
                                if edge_pair[1] in edge.verts:
                                    edge.select = True
                                    edges_for_face.append(edge)
                                    break

                        # Use contextual_create to create the face
                        if len(edges_for_face) >= 2:
                            result = bmesh.ops.contextual_create(bm, geom=edges_for_face)
                            if result and 'faces' in result:
                                new_faces = len(result['faces'])
                                total_faces_created += new_faces

                    except Exception as e:
                        pass

        # --------------------------------------------------------------------------------
        # PART B: Create edges and faces between bones at joints
        # --------------------------------------------------------------------------------

        script_log("PART B: Creating edges and faces between bones at joints...")

        # Define bone connections
        bone_connections = []
        for bone_name, bone_data in bone_definitions.items():
            parent_name = bone_data.get("parent")
            if bone_name in ring_data_by_bone and parent_name in ring_data_by_bone:
                parent_head_tail = bone_data.get("parent_head_tail", "tail")
                bone_connections.append((parent_name, bone_name, parent_head_tail))

        # SPECIAL HANDLING: Add connections between shoulder bones and hip bones (head-to-head)
        shoulder_bones = ["LeftShoulder", "RightShoulder"]
        hip_bones = ["LeftHip", "RightHip"]

        # Connect shoulder bones to each other (head-to-head at VIRTUAL_SHOULDER_MIDPOINT)
        if all(bone in ring_data_by_bone for bone in shoulder_bones):
            bone_connections.append(("LeftShoulder", "RightShoulder", "head"))
            script_log("Added shoulder head-to-head connection: LeftShoulder <-> RightShoulder")

        # Connect hip bones to each other (head-to-head at VIRTUAL_HIP_MIDPOINT)
        if all(bone in ring_data_by_bone for bone in hip_bones):
            bone_connections.append(("LeftHip", "RightHip", "head"))
            script_log("Added hip head-to-head connection: LeftHip <-> RightHip")

        for parent_name, child_name, connection_type in bone_connections:
            if parent_name not in ring_data_by_bone or child_name not in ring_data_by_bone:
                continue

            parent_rings = ring_data_by_bone[parent_name]
            child_rings = ring_data_by_bone[child_name]

            if not parent_rings or not child_rings:
                continue

            # Determine which rings to connect
            if connection_type == "tail":
                parent_connect_ring = max(parent_rings, key=lambda r: r['position'])
                child_connect_ring = min(child_rings, key=lambda r: r['position'])
            elif connection_type == "head":
                # SPECIAL CASE: For head-to-head connections (shoulders and hips)
                parent_connect_ring = min(parent_rings, key=lambda r: r['position'])
                child_connect_ring = min(child_rings, key=lambda r: r['position'])
            else:
                parent_connect_ring = max(parent_rings, key=lambda r: r['position'])
                child_connect_ring = min(child_rings, key=lambda r: r['position'])

            parent_indices = parent_connect_ring['vertex_indices']
            child_indices = child_connect_ring['vertex_indices']

            ring_size = min(len(parent_indices), len(child_indices))

            # Use config-based offset with negative handling
            child_bone_data = bone_definitions.get(child_name, {})
            parent_offset = child_bone_data.get("parent_best_offset", 0)

            script_log(f"Connecting {parent_name}->{child_name} with offset {parent_offset}")

            # Create vertical edges with config-based offset
            vertical_edges = []
            for j in range(ring_size):
                parent_idx = parent_indices[j]

                # Apply the offset (negative means reverse orientation + offset)
                if parent_offset >= 0:
                    # Normal offset: child ring rotates clockwise by offset
                    child_idx = child_indices[(j + parent_offset) % ring_size]
                else:
                    # Negative offset: reverse child ring orientation, then apply positive offset
                    abs_offset = abs(parent_offset)
                    # Reverse the ring: index = (ring_size - 1 - j)
                    # Then apply offset
                    child_idx = child_indices[(ring_size - 1 - j + abs_offset) % ring_size]

                v1 = bm.verts[parent_idx]
                v4 = bm.verts[child_idx]

                # Create vertical edge if it doesn't exist
                edge_exists = False
                vertical_edge = None
                for edge in v1.link_edges:
                    if v4 in edge.verts:
                        edge_exists = True
                        vertical_edge = edge
                        break

                if not edge_exists:
                    vertical_edge = bm.edges.new((v1, v4))

                vertical_edges.append(vertical_edge)

            # Create faces between consecutive vertical edges
            for j in range(ring_size):
                edge1 = vertical_edges[j]
                edge2 = vertical_edges[(j + 1) % ring_size]

                if edge1 is None or edge2 is None:
                    continue

                try:
                    # Get the four vertices for this quad with offset
                    v1 = bm.verts[parent_indices[j]]

                    if parent_offset >= 0:
                        # Normal orientation
                        v4 = bm.verts[child_indices[(j + parent_offset) % ring_size]]
                        v3 = bm.verts[child_indices[((j + 1) + parent_offset) % ring_size]]
                    else:
                        # Reversed orientation
                        abs_offset = abs(parent_offset)
                        v4 = bm.verts[child_indices[(ring_size - 1 - j + abs_offset) % ring_size]]
                        v3 = bm.verts[child_indices[(ring_size - 1 - (j + 1) + abs_offset) % ring_size]]

                    v2 = bm.verts[parent_indices[(j + 1) % ring_size]]

                    # Check if this face already exists
                    face_exists = False
                    for face in v1.link_faces:
                        if (v2 in face.verts and v3 in face.verts and v4 in face.verts):
                            face_exists = True
                            break

                    if not face_exists:
                        # Create the face with winding order: v1 -> v4 -> v3 -> v2
                        new_face = bm.faces.new([v1, v4, v3, v2])
                        new_face.normal_update()

                except Exception as e:
                    # Face might already exist, continue
                    pass

        # --------------------------------------------------------------------------------
        # PART C1: Create simple flat end caps for extremities
        # --------------------------------------------------------------------------------

        script_log("PART C1: Creating flat end caps for extremities...")

        # Define which bones need end caps (terminal bones)
        end_cap_bones = [
            "Head",  # Head top (tail end)
            "LeftHand",  # Left hand end (tail end)
            "RightHand",  # Right hand end (tail end)
            "LeftFoot",  # Left foot end (tail end)
            "RightFoot",  # Right foot end (tail end)
            "UpperSpine"  # UpperSpine head end (connects to shoulders)
        ]

        end_caps_created = 0

        for bone_name in end_cap_bones:
            if bone_name in ring_data_by_bone:
                bone_rings = ring_data_by_bone[bone_name]
                if bone_rings:
                    # SPECIAL CASE: UpperSpine needs end cap at HEAD (position 0.0) not tail
                    if bone_name == "UpperSpine":
                        extremity_ring = min(bone_rings, key=lambda r: r['position'])  # Head end
                        script_log(f"Creating UpperSpine end cap at HEAD (position {extremity_ring['position']})")
                    else:
                        # All other bones: end cap at tail (highest position)
                        extremity_ring = max(bone_rings, key=lambda r: r['position'])
                        script_log(f"Creating {bone_name} end cap at TAIL (position {extremity_ring['position']})")

                    extremity_indices = extremity_ring['vertex_indices']

                    if len(extremity_indices) >= 3:
                        # Get all vertices for the end cap
                        end_cap_verts = [bm.verts[idx] for idx in extremity_indices]

                        try:
                            # Create a simple flat face using all vertices
                            end_cap_face = bm.faces.new(end_cap_verts)
                            end_caps_created += 1
                            script_log(f"✓ Created flat end cap for {bone_name} with {len(end_cap_verts)} vertices")

                        except Exception as e:
                            # Fallback: triangle fan from first vertex
                            try:
                                center_vert = end_cap_verts[0]  # Use first vertex as anchor
                                for i in range(1, len(end_cap_verts) - 1):
                                    v1 = end_cap_verts[i]
                                    v2 = end_cap_verts[i + 1]
                                    try:
                                        tri_face = bm.faces.new([center_vert, v1, v2])
                                        end_caps_created += 1
                                    except Exception as tri_e:
                                        pass
                                script_log(f"✓ Created triangle fan end cap for {bone_name}")
                            except Exception as fan_e:
                                script_log(f"❌ Failed to create end cap for {bone_name}")

        script_log(f"Created {end_caps_created} flat end caps total")

        # --------------------------------------------------------------------------------
        # PART C2: Add more vertices to hip end caps for better side coverage
        # --------------------------------------------------------------------------------

        script_log("PART C2: Adding more vertices to hip end caps...")

        # Store the count before adding new vertices
        initial_hip_vertex_count = len(hip_horizontal_ring_vertices)

        for hip_bone_name in ["LeftHip", "RightHip"]:
            if hip_bone_name in ring_data_by_bone:
                hip_rings = ring_data_by_bone[hip_bone_name]
                if hip_rings:
                    # Use the ring with highest position (tail end of hip)
                    hip_end_ring = max(hip_rings, key=lambda r: r['position'])
                    hip_end_indices = hip_end_ring['vertex_indices']

                    def_bone_name = f"DEF_{hip_bone_name}"

                    # Use is_this_a_hip_horizontal_vertex to find the 2 horizontal vertices
                    horizontal_vertex_indices = []
                    for vertex_index_in_ring, vertex_idx in enumerate(hip_end_indices):
                        if is_this_a_hip_horizontal_vertex(def_bone_name, vertex_index_in_ring):
                            horizontal_vertex_indices.append(vertex_idx)
                            script_log(
                                f"✓ Found horizontal vertex for {hip_bone_name}: ring_pos={vertex_index_in_ring}, vertex_idx={vertex_idx}")

                    # We should have 2 horizontal vertices (opposite sides)
                    if len(horizontal_vertex_indices) == 2:
                        v1_idx, v2_idx = horizontal_vertex_indices

                        # FIX: Ensure lookup table is current before accessing vertices
                        bm.verts.ensure_lookup_table()
                        bm.edges.ensure_lookup_table()

                        v1 = bm.verts[v1_idx]
                        v2 = bm.verts[v2_idx]

                        script_log(
                            f"Creating edge between horizontal vertices {v1_idx} and {v2_idx} for {hip_bone_name}")

                        # Create a temporary edge between the two horizontal vertices
                        try:
                            temp_edge = bm.edges.new([v1, v2])

                            # FIX: Update lookup table after creating edge
                            bm.verts.ensure_lookup_table()
                            bm.edges.ensure_lookup_table()

                            # Subdivide the edge to add 2 intermediate vertices
                            result = bmesh.ops.subdivide_edges(
                                bm,
                                edges=[temp_edge],
                                cuts=2,  # Creates 2 new vertices (3 segments total)
                                use_grid_fill=True
                            )

                            # FIX: Update lookup table after subdivision
                            bm.verts.ensure_lookup_table()
                            bm.edges.ensure_lookup_table()

                            # Get the new vertices from subdivision
                            if 'geom' in result:
                                new_verts = [item for item in result['geom'] if isinstance(item, bmesh.types.BMVert)]

                                # Add the new subdivided vertices to our existing hip ring
                                for i, new_vert in enumerate(new_verts):
                                    hip_horizontal_ring_vertices.append({
                                        'vertex_index': new_vert.index,
                                        'position': new_vert.co.copy(),
                                        'bone': def_bone_name,
                                        'ring_position': f"subdiv_{i}"  # Mark as subdivided
                                    })

                                    vertex_index = new_vert.index
                                    vertex_groups[vertex_index] = def_bone_name
                                    script_log(f"✓ Added {len(new_verts)} vertices to {hip_bone_name} end cap")

                        except Exception as e:
                            script_log(f"Failed to subdivide hip end cap for {hip_bone_name}: {e}")
                    else:
                        script_log(
                            f"⚠ Wrong number of horizontal vertices for {hip_bone_name}: {len(horizontal_vertex_indices)} (expected 2)")

        # FIX: Final update of lookup tables before continuing
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        script_log(f"Added {len(hip_horizontal_ring_vertices) - initial_hip_vertex_count} new vertices to hip end caps")

        # --------------------------------------------------------------------------------
        # PART D: Create bridge between head and neck
        # --------------------------------------------------------------------------------

        script_log("PART D: Creating bridge between head and neck...")

        if "Neck" in ring_data_by_bone and "Head" in ring_data_by_bone:
            neck_rings = ring_data_by_bone["Neck"]
            head_rings = ring_data_by_bone["Head"]

            if neck_rings and head_rings:
                # Get the top ring of the neck (connects to head)
                neck_connect_ring = max(neck_rings, key=lambda r: r['position'])
                # Get the bottom ring of the head (connects to neck)
                head_connect_ring = min(head_rings, key=lambda r: r['position'])

                neck_indices = neck_connect_ring['vertex_indices']
                head_indices = head_connect_ring['vertex_indices']

                ring_size = min(len(neck_indices), len(head_indices))

                # Create vertical edges between neck and head rings
                vertical_edges = []
                for j in range(ring_size):
                    neck_vert = bm.verts[neck_indices[j]]
                    head_vert = bm.verts[head_indices[j]]

                    # Create vertical edge if it doesn't exist
                    edge_exists = False
                    vertical_edge = None
                    for edge in neck_vert.link_edges:
                        if head_vert in edge.verts:
                            edge_exists = True
                            vertical_edge = edge
                            break

                    if not edge_exists:
                        vertical_edge = bm.edges.new((neck_vert, head_vert))
                        total_edges_created += 1

                    vertical_edges.append(vertical_edge)

                # Create faces between consecutive vertical edges using contextual_create
                faces_created_neck_head = 0
                for j in range(ring_size):
                    edge1 = vertical_edges[j]
                    edge2 = vertical_edges[(j + 1) % ring_size]

                    try:
                        # Deselect all first
                        for edge in bm.edges:
                            edge.select = False

                        # Select the two vertical edges we want to bridge
                        edge1.select = True
                        edge2.select = True

                        # Use contextual_create to create face between the edges
                        result = bmesh.ops.contextual_create(bm, geom=[edge1, edge2])

                        if result and 'faces' in result:
                            new_faces = len(result['faces'])
                            faces_created_neck_head += new_faces
                            total_faces_created += new_faces

                    except Exception as e:
                        pass

        # --------------------------------------------------------------------------------
        # PART E: Systematic bridge with LCM-based subdivision - USING PRE-COLLECTED HIP VERTICES
        # --------------------------------------------------------------------------------

        script_log("PART E: Creating systematic bridge with vertex subdivision...")

        # Use LowerSpine for the spine-hip bridge
        spine_bone_name = "LowerSpine"

        if spine_bone_name in ring_data_by_bone and hip_horizontal_ring_vertices:
            # Get the bottom ring of the spine (lowest position)
            spine_rings = ring_data_by_bone[spine_bone_name]
            spine_bottom_ring = min(spine_rings, key=lambda r: r['position'])
            spine_indices = spine_bottom_ring['vertex_indices']  # n vertices

            script_log(f"Using {len(hip_horizontal_ring_vertices)} pre-collected hip vertices for horizontal ring")

            # Create the horizontal ring from our pre-collected vertices from PART A and PART C2
            # First, sort them by angle around the center to form a ring
            if len(hip_horizontal_ring_vertices) >= 4:  # Need enough vertices for a ring
                # Calculate center of all hip vertices
                center = Vector((0, 0, 0))
                for vertex_data in hip_horizontal_ring_vertices:
                    center += vertex_data['position']
                center /= len(hip_horizontal_ring_vertices)

                script_log(f"Calculated center for hip ring: {center}")

                # Sort by angle in horizontal plane (X-Y plane)
                def get_horizontal_angle(pos):
                    relative = pos - center
                    # Use atan2 but ensure consistent clockwise winding
                    return math.atan2(relative.x, relative.y)  # SWAP X and Y for different winding

                # Sort vertices by their angle around the center
                hip_horizontal_ring_vertices.sort(key=lambda x: get_horizontal_angle(x['position']))

                # Extract just the vertex indices in proper ring order
                horizontal_ring = [vertex_data['vertex_index'] for vertex_data in hip_horizontal_ring_vertices]

                script_log(f"Created horizontal ring with {len(horizontal_ring)} vertices in proper order")

            else:
                script_log(f"WARNING: Not enough hip vertices ({len(hip_horizontal_ring_vertices)}) for proper ring")
                # Fallback: use all collected vertices without sorting
                horizontal_ring = [vertex_data['vertex_index'] for vertex_data in hip_horizontal_ring_vertices]

            # DEFINE n AND m HERE
            n = len(spine_indices)  # spine vertices
            m = len(horizontal_ring)  # hip vertices

            script_log(f"Bridge configuration: spine ring has {n} vertices, hip ring has {m} vertices")

            # STEP E1: Calculate LCM and subdivide both rings
            def calculate_lcm(a, b):
                from math import gcd
                return abs(a * b) // gcd(a, b) if a and b else 0

            lcm = calculate_lcm(n, m)
            script_log(f"Subdividing both rings to {lcm} vertices each for perfect bridging")

            # Ensure lookup table is up to date
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            # Subdivide spine ring (n → lcm) - STORE VERTEX REFERENCES
            spine_subdivided_verts = []
            for i in range(n):
                current_vert = bm.verts[spine_indices[i]]
                next_vert = bm.verts[spine_indices[(i + 1) % n]]

                # Add current vertex
                spine_subdivided_verts.append(current_vert)

                # Add intermediate vertices for this edge
                subdivisions_per_edge = (lcm // n) - 1
                for j in range(1, subdivisions_per_edge + 1):
                    t = j / (subdivisions_per_edge + 1)
                    intermediate_pos = current_vert.co.lerp(next_vert.co, t)

                    new_vert = bm.verts.new(intermediate_pos)
                    spine_subdivided_verts.append(new_vert)

                    # Update lookup table after each new vertex
                    bm.verts.ensure_lookup_table()

                    # Add to same vertex group as spine
                    vertex_groups[len(bm.verts) - 1] = vertex_groups[spine_indices[i]]

            # Subdivide hip ring (m → lcm) - STORE VERTEX REFERENCES
            hip_subdivided_verts = []
            for i in range(m):
                current_vert = bm.verts[horizontal_ring[i]]
                next_vert = bm.verts[horizontal_ring[(i + 1) % m]]

                # Add current vertex
                hip_subdivided_verts.append(current_vert)

                # Add intermediate vertices for this edge
                subdivisions_per_edge = (lcm // m) - 1
                for j in range(1, subdivisions_per_edge + 1):
                    t = j / (subdivisions_per_edge + 1)
                    intermediate_pos = current_vert.co.lerp(next_vert.co, t)

                    new_vert = bm.verts.new(intermediate_pos)
                    hip_subdivided_verts.append(new_vert)

                    # Update lookup table after each new vertex
                    bm.verts.ensure_lookup_table()

                    # Add to appropriate vertex group
                    if horizontal_ring[i] in vertex_groups:
                        vertex_groups[len(bm.verts) - 1] = vertex_groups[horizontal_ring[i]]

            # Final update of lookup tables
            bm.verts.ensure_lookup_table()
            bm.edges.ensure_lookup_table()
            bm.faces.ensure_lookup_table()

            # Verify we have the correct number of vertices
            script_log(
                f"After subdivision: spine ring has {len(spine_subdivided_verts)} vertices, hip ring has {len(hip_subdivided_verts)} vertices")

            # STEP E2: Create perfect 1:1 QUADS with configurable offset
            quads_created = 0

            # Get the offset from the LowerSpine bone definition
            spine_data = bone_definitions.get("LowerSpine", {})
            optimal_offset = spine_data.get("parent_best_offset", 0) % lcm

            script_log(f"Using LowerSpine parent_best_offset {optimal_offset} for spine-hip bridge alignment")

            # Now create quads with detailed logging
            script_log("=== DEBUG: QUAD CREATION ===")
            for i in range(lcm):
                # Apply offset to hip ring indices
                hip_index = (i + optimal_offset) % lcm
                hip_next_index = (hip_index + 1) % lcm

                v_spine = spine_subdivided_verts[i]
                v_spine_next = spine_subdivided_verts[(i + 1) % lcm]
                v_hip = hip_subdivided_verts[hip_index]
                v_hip_next = hip_subdivided_verts[hip_next_index]

                try:
                    # Create quad: spine -> spine_next -> hip_next -> hip
                    quad_face = bm.faces.new([v_spine, v_spine_next, v_hip_next, v_hip])
                    quads_created += 1

                    # DEBUG: Log detailed information for first few quads
                    if quads_created <= 6:
                        script_log(f"Quad {quads_created}:")
                        script_log(f"  Spine[{i}]: ({v_spine.co.x:.4f}, {v_spine.co.y:.4f}, {v_spine.co.z:.4f})")
                        script_log(f"  Hip[{hip_index}]: ({v_hip.co.x:.4f}, {v_hip.co.y:.4f}, {v_hip.co.z:.4f})")

                except Exception as e:
                    script_log(f"Failed to create quad {i}: {e}")

            script_log(f"Created {quads_created} quads for perfect spine-hip bridge with offset {optimal_offset}")

            # DEBUG: Final verification
            script_log("=== DEBUG: FINAL VERIFICATION ===")
            script_log(f"Total vertices in bmesh: {len(bm.verts)}")
            script_log(f"Total faces in bmesh: {len(bm.faces)}")

        else:
            script_log("WARNING: Cannot create spine-hip bridge - missing spine or hip vertices")

        script_log("Systematic spine-hip bridge with subdivision completed")

        # --------------------------------------------------------------------------------
        # STEP F: Bridge neck to UpperSpine
        script_log("STEP F: Creating bridge between neck and UpperSpine...")

        if "Neck" in ring_data_by_bone and "UpperSpine" in ring_data_by_bone:
            neck_rings = ring_data_by_bone["Neck"]
            spine_rings = ring_data_by_bone["UpperSpine"]

            if neck_rings and spine_rings:
                # Get the bottom ring of the neck (connects to spine)
                neck_bottom_ring = min(neck_rings, key=lambda r: r['position'])
                # Get the HEAD ring of the UpperSpine (connects to neck) - FIXED: use min() not max()
                spine_head_ring = min(spine_rings, key=lambda r: r['position'])

                neck_indices = neck_bottom_ring['vertex_indices']
                spine_indices = spine_head_ring['vertex_indices']

                ring_size = min(len(neck_indices), len(spine_indices))

                # Create vertical edges between neck and spine rings
                vertical_edges = []
                for j in range(ring_size):
                    neck_vert = bm.verts[neck_indices[j]]
                    spine_vert = bm.verts[spine_indices[j]]

                    # Create vertical edge if it doesn't exist
                    edge_exists = False
                    vertical_edge = None
                    for edge in neck_vert.link_edges:
                        if spine_vert in edge.verts:
                            edge_exists = True
                            vertical_edge = edge
                            break

                    if not edge_exists:
                        vertical_edge = bm.edges.new((neck_vert, spine_vert))
                        total_edges_created += 1

                    vertical_edges.append(vertical_edge)

                # Create faces between consecutive vertical edges using contextual_create
                faces_created_neck_spine = 0
                for j in range(ring_size):
                    edge1 = vertical_edges[j]
                    edge2 = vertical_edges[(j + 1) % ring_size]

                    try:
                        # Deselect all first
                        for edge in bm.edges:
                            edge.select = False

                        # Select the two vertical edges we want to bridge
                        edge1.select = True
                        edge2.select = True

                        # Use contextual_create to create face between the edges
                        result = bmesh.ops.contextual_create(bm, geom=[edge1, edge2])

                        if result and 'faces' in result:
                            new_faces = len(result['faces'])
                            faces_created_neck_spine += new_faces
                            total_faces_created += new_faces

                    except Exception as e:
                        # Fallback: manual face creation
                        try:
                            v1 = bm.verts[neck_indices[j]]
                            v4 = bm.verts[spine_indices[j]]
                            v3 = bm.verts[spine_indices[(j + 1) % ring_size]]
                            v2 = bm.verts[neck_indices[(j + 1) % ring_size]]

                            new_face = bm.faces.new([v1, v4, v3, v2])
                            faces_created_neck_spine += 1
                            total_faces_created += 1
                        except Exception as fallback_e:
                            pass

                script_log(f"✓ Created {faces_created_neck_spine} faces bridging neck to UpperSpine HEAD")
        else:
            script_log(f"⚠ Neck or UpperSpine rings not found for bridging")

        # --------------------------------------------------------------------------------
        # PART G: Bridge between LowerSpine and UpperSpine
        # --------------------------------------------------------------------------------
        script_log("PART G: Creating bridge between LowerSpine and UpperSpine...")

        if "LowerSpine" in ring_data_by_bone and "UpperSpine" in ring_data_by_bone:
            lower_spine_rings = ring_data_by_bone["LowerSpine"]
            upper_spine_rings = ring_data_by_bone["UpperSpine"]

            if lower_spine_rings and upper_spine_rings:
                # Get the top ring of LowerSpine (position 1.0 - connects to UpperSpine)
                lower_spine_connect_ring = max(lower_spine_rings, key=lambda r: r['position'])
                # Get the top ring of UpperSpine (position 1.0 - connects to LowerSpine)
                upper_spine_connect_ring = max(upper_spine_rings, key=lambda r: r['position'])

                lower_spine_indices = lower_spine_connect_ring['vertex_indices']
                upper_spine_indices = upper_spine_connect_ring['vertex_indices']

                ring_size = min(len(lower_spine_indices), len(upper_spine_indices))

                script_log(
                    f"LowerSpine ring size: {len(lower_spine_indices)}, UpperSpine ring size: {len(upper_spine_indices)}")

                # Create vertical edges with winding order correction
                vertical_edges = []
                for j in range(ring_size):
                    # LowerSpine uses normal winding order (0, 1, 2, ...)
                    lower_spine_vert = bm.verts[lower_spine_indices[j]]

                    # UpperSpine uses reversed winding order (11, 10, 9, ...)
                    # So vertex j on LowerSpine connects to vertex (ring_size - 1 - j) on UpperSpine
                    upper_spine_index = (ring_size - 1 - j) % ring_size
                    upper_spine_vert = bm.verts[upper_spine_indices[upper_spine_index]]

                    # Create vertical edge if it doesn't exist
                    edge_exists = False
                    vertical_edge = None
                    for edge in lower_spine_vert.link_edges:
                        if upper_spine_vert in edge.verts:
                            edge_exists = True
                            vertical_edge = edge
                            break

                    if not edge_exists:
                        vertical_edge = bm.edges.new((lower_spine_vert, upper_spine_vert))
                        total_edges_created += 1

                    vertical_edges.append(vertical_edge)

                # Create faces between consecutive vertical edges with proper winding
                faces_created_spine_bridge = 0
                for j in range(ring_size):
                    edge1 = vertical_edges[j]
                    edge2 = vertical_edges[(j + 1) % ring_size]

                    if edge1 is None or edge2 is None:
                        continue

                    try:
                        # Get vertices with corrected winding order
                        # LowerSpine: normal order
                        v1_lower = bm.verts[lower_spine_indices[j]]
                        v2_lower = bm.verts[lower_spine_indices[(j + 1) % ring_size]]

                        # UpperSpine: reversed order
                        upper_idx1 = (ring_size - 1 - j) % ring_size
                        upper_idx2 = (ring_size - 1 - (j + 1)) % ring_size
                        v4_upper = bm.verts[upper_spine_indices[upper_idx1]]
                        v3_upper = bm.verts[upper_spine_indices[upper_idx2]]

                        # Create face with proper winding: v1_lower -> v4_upper -> v3_upper -> v2_lower
                        new_face = bm.faces.new([v1_lower, v4_upper, v3_upper, v2_lower])
                        new_face.normal_update()
                        faces_created_spine_bridge += 1
                        total_faces_created += 1

                    except Exception as e:
                        script_log(f"Failed to create spine bridge face {j}: {e}")

                script_log(f"✓ Created {faces_created_spine_bridge} faces bridging LowerSpine to UpperSpine")
        else:
            script_log("⚠ LowerSpine or UpperSpine rings not found for bridging")

        # Update the mesh with all edges and faces
        bm.to_mesh(mesh)
        mesh.update()
        bm.free()

        script_log(f"✓ Created {total_edges_created} edges and {total_faces_created} faces total")
        script_log(f"✓ Final mesh: {len(mesh.vertices)} vertices, {len(mesh.edges)} edges, {len(mesh.polygons)} faces")

        # Position and parent the mesh (already done above, but keep this for positioning)
        cloud_obj.location = armature_obj.location.copy()
        cloud_obj.rotation_euler = armature_obj.rotation_euler.copy()

        bpy.context.view_layer.objects.active = cloud_obj
        cloud_obj.select_set(True)
        # bpy.ops.object.transform_apply(location=True, rotation=True, scale=True) # Removed because it caused the right foot to move to 0,0,0

        # Create vertex groups
        cloud_obj.vertex_groups.clear()
        bone_vertex_groups = {}

        # Include all spine segments in vertex groups
        for bone_name in armature_data.bones.keys():
            if bone_name.startswith("DEF_"):
                vg = cloud_obj.vertex_groups.new(name=bone_name)
                bone_vertex_groups[bone_name] = vg

        for vertex_index, bone_name in vertex_groups.items():
            if bone_name in bone_vertex_groups:
                bone_vertex_groups[bone_name].add([vertex_index], 1.0, 'REPLACE')

        # Apply material
        # flesh_mat = create_transparent_material("#FF6B9D", alpha=0.7)
        # cloud_obj.data.materials.append(flesh_mat)

        cloud_obj.show_wire = True
        cloud_obj.show_all_edges = True

        script_log("✓ Complete flesh mesh creation finished")

        return cloud_obj

    except Exception as e:
        script_log(f"❌ Error creating flesh mesh: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        return None

##########################################################################################

def save_blender_file():
    """Save the Blender file to the correct output location"""
    try:
        from utils import get_show_path, get_scene_folder_name

        show_path = get_show_path(args.show)
        scene_folder_name = get_scene_folder_name(args.show, args.scene)
        outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
        os.makedirs(outputs_dir, exist_ok=True)

        output_blend_file = os.path.join(outputs_dir, f"{args.scene}_kid.blend")
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_file)
        script_log(f"Kid figure animation saved to: {output_blend_file}")

    except Exception as e:
        script_log(f"Error saving Blender file: {e}")


##########################################################################################

def verify_hip_constraints(armature_obj):
    """Verify that all hip-related constraints are properly set up"""
    script_log("=== VERIFYING HIP CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    hip_midpoint_obj = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")

    # Check LowerSpine constraints
    if "DEF_LowerSpine" in armature_obj.pose.bones:
        lower_spine = armature_obj.pose.bones["DEF_LowerSpine"]
        script_log(f"DEF_LowerSpine constraints: {len(lower_spine.constraints)}")
        for constraint in lower_spine.constraints:
            # FIX: Check if constraint has target attribute before accessing it
            if hasattr(constraint, 'target'):
                target_name = constraint.target.name if constraint.target else "None"
            else:
                target_name = "No target"
            script_log(f"  - {constraint.type} -> {target_name}")

    # Check hip constraints
    for hip_name in ["DEF_LeftHip", "DEF_RightHip"]:
        if hip_name in armature_obj.pose.bones:
            hip_bone = armature_obj.pose.bones[hip_name]
            script_log(f"{hip_name} constraints: {len(hip_bone.constraints)}")
            for constraint in hip_bone.constraints:
                # FIX: Check if constraint has target attribute before accessing it
                if hasattr(constraint, 'target'):
                    target_name = constraint.target.name if constraint.target else "None"
                else:
                    target_name = "No target"
                script_log(f"  - {constraint.type} -> {target_name}")

    bpy.ops.object.mode_set(mode='OBJECT')


##########################################################################################

def main_execution():
    """Clean main execution"""
    script_log("=== 4K KID INNER STARTED ===\n")

    try:
        figure_name = "Main"

        # Load configuration and data
        load_config_and_data()

        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        cleanup_existing_objects("Main")

        # Extract bone hierarchy and constraints
        extract_bone_hierarchy_and_controls(bone_definitions)

        # 1. CREATE CONTROL POINTS FIRST (including VIRTUAL_HIP_MIDPOINT)
        script_log("=== CREATING CONTROL POINTS FIRST ===")
        create_control_points(figure_name)

        # 1.5 CREATE FRAME CONTROL POINTS (NEW COORDINATE FRAME SYSTEM)
        script_log("=== CREATING FRAME CONTROL POINTS ===")
        first_frame = mocap_data.get(str(frame_numbers[0]), {})
        create_hip_frame_control_point(first_frame)
        create_shoulder_frame_control_point(first_frame)

        # 2. Set up virtual point constraints
        script_log("=== SETTING UP VIRTUAL POINT CONSTRAINTS ===")
        setup_virtual_point_constraints()

        script_log("=== SETTING UP VIRTUAL FRAME CONSTRAINTS ===")
        setup_virtual_frame_constraints()

        # 3. Create rig with two-segment spine and programmatic root bones (now VIRTUAL_HIP_MIDPOINT exists)
        armature_obj = create_kid_rig(figure_name)

        # 4. Create flesh/vertex cloud
        flesh_obj = create_kid_flesh(armature_obj, figure_name)

        # 5. Align bones with control points
        align_bones_with_control_points(armature_obj, figure_name)

        # 6. Set up direct constraints for limbs FIRST (STRETCH_TO constraints) - SKIPPING HIPS/SHOULDERS
        script_log("=== SETTING UP DIRECT CONSTRAINTS (SKIPPING HIPS/SHOULDERS) ===")
        setup_direct_constraints(armature_obj, figure_name)

        # 7. Set up two-segment spine constraints SECOND
        script_log("=== SETTING UP TWO-SEGMENT SPINE CONSTRAINTS ===")
        setup_two_segment_spine_constraints(armature_obj, figure_name)

        # 8. Set up root bone transform constraints LAST (NEW COORDINATE FRAME SYSTEM)
        script_log("=== SETTING UP ROOT BONE TRANSFORM CONSTRAINTS ===")
        setup_root_bone_transform_constraints(armature_obj)

        # 9. Verify hip constraints
        verify_hip_constraints(armature_obj)

        # 10. Force initial constraint solve
        script_log("=== FORCING INITIAL CONSTRAINT SOLVE ===")
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.visual_transform_apply()
        bpy.ops.object.mode_set(mode='OBJECT')

        # 11. Force scene update after constraint creation
        bpy.context.view_layer.update()

        # 12. SKIP duplicate parenting since create_kid_flesh already handles it
        if flesh_obj:
            script_log("=== SKIPPING DUPLICATE PARENTING - create_kid_flesh already handled it ===")
            # finalize_flesh_mesh and parent_flesh are now no-ops since create_kid_flesh handles everything

        # 13. FINAL SCENE UPDATE BEFORE SAVE
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()

        # 14. Save file
        save_blender_file()

        script_log("=== KID FIGURE CREATION COMPLETED ===")

    except Exception as e:
        script_log(f"ERROR in main execution: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")


##########################################################################################

if __name__ == "__main__":
    main_execution()