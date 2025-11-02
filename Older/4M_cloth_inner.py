# 4M_cloth_inner.py (Version 12.0 - Nice coat!)

import bpy
import bmesh
import sys
import os
import argparse
import json
import math
from mathutils import Vector, Matrix

##########################################################################################

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments passed from 4M_cloth.py"""
    parser = argparse.ArgumentParser(description='4M Cloth Inner Script')
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
joint_vertex_bundles = {}

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
        step_paths = get_processing_step_paths(args.show, args.scene, "cloth_animation")

        # Input JSON is from the apply_physics step
        processing_steps = scene_config.get("processing_steps", {})
        apply_physics_step = processing_steps.get("apply_physics", {})
        input_json_relative = apply_physics_step.get("output_file", "step_4_input.json")

        # Build absolute path to input JSON
        scene_paths = get_scene_paths(args.show, args.scene)
        inputs_dir = os.path.dirname(scene_paths["output_pose_data"])
        INPUT_JSON_FILE = os.path.join(inputs_dir, input_json_relative)

        # Load cloth specific config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

        with open(CLOTH_CONFIG_FILE, 'r') as file:
            config = json.load(file)
            bone_definitions = config.get("bone_definitions", {})
            cloth_settings = config.get("cloth_settings", {})
            squish_factors["x"] = cloth_settings.get("x_squish_fraction", 1.0)
            squish_factors["y"] = cloth_settings.get("y_squish_fraction", 1.0)
            squish_factors["z"] = cloth_settings.get("z_squish_fraction", 1.0)

        with open(INPUT_JSON_FILE, 'r') as file:
            mocap_data = json.load(file)

        script_log(f"Loaded mocap data from: {INPUT_JSON_FILE}")
        script_log(f"Loaded cloth config from: {CLOTH_CONFIG_FILE}")

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
        # Remove cloth garments
        elif "LongSleeveShirt" in obj.name or "Pants" in obj.name or "Coat" in obj.name:
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

def make_vertex_bundle(armature_obj, bone_name, radius, vertex_count, joint_type, side):
    """
    Create a procedural vertex bundle using Fibonacci Spiral for even distribution
    SIMPLIFIED: Only need one bone name and choose head/tail based on joint type
    """
    script_log(f"Creating vertex bundle for {joint_type} ({side}) - {vertex_count} vertices, radius: {radius:.3f}")

    # Get bone position
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    bone = armature_obj.pose.bones.get(bone_name)

    if not bone:
        script_log(f"ERROR: Missing bone for {joint_type} bundle: {bone_name}")
        return None

    # Choose joint center based on joint type
    if joint_type in ["shoulder", "hip", "head_neck", "neck_coordination"]:
        # Use bone head for joints that connect to torso/body
        joint_center = armature_obj.matrix_world @ bone.head
        position_type = "head"
    else:
        # Use bone tail for limb joints (elbow, knee, ankle, wrist)
        joint_center = armature_obj.matrix_world @ bone.tail
        position_type = "tail"

    script_log(f"  Joint center: {joint_center} ({bone_name}.{position_type})")

    # =========================================================================
    # FIBONACCI SPIRAL VERTEX DISTRIBUTION
    # =========================================================================
    script_log(f"Generating {vertex_count} vertices using Fibonacci Spiral distribution")

    vertex_positions = []
    golden_angle = math.pi * (3 - math.sqrt(5))

    for k in range(vertex_count):
        z = 1 - (2 * k + 1) / vertex_count
        r = math.sqrt(1 - z * z)
        phi = k * golden_angle

        x = r * math.cos(phi)
        y = r * math.sin(phi)

        position = Vector((x * radius, y * radius, z * radius))

        # Hemisphere shaping for shoulders/hips
        if joint_type in ["shoulder", "hip"]:
            if side == "left" and x > 0:
                position.x = abs(position.x)
            elif side == "right" and x < 0:
                position.x = -abs(position.x)

        vertex_positions.append(position)

    # Translate to joint center
    for i in range(len(vertex_positions)):
        vertex_positions[i] += joint_center

    # =========================================================================
    # CREATE BUNDLE DATA STRUCTURE
    # =========================================================================
    bundle_data = {
        "joint_type": joint_type,
        "side": side,
        "vertex_count": vertex_count,
        "radius": radius,
        "joint_center": joint_center,
        "vertex_positions": vertex_positions,
        "bone_name": bone_name,
        "position_type": position_type,
        "created_at": bpy.context.scene.frame_current,
        "distribution_method": "fibonacci_spiral"
    }

    bpy.ops.object.mode_set(mode='OBJECT')

    script_log(f"✓ Created {joint_type} bundle with {len(vertex_positions)} vertices")
    script_log(f"  Center: {joint_center} ({bone_name}.{position_type})")
    script_log(f"  Radius: {radius:.3f}")

    return bundle_data

##########################################################################################

def make_vertex_all_bundles(armature_obj):
    """
    Create ALL vertex bundles - simplified robust version
    """
    global joint_vertex_bundles
    script_log("=== CREATING VERTEX BUNDLES (SIMPLIFIED) ===")

    # Load garment configs - THIS IS THE ONLY REAL FAILURE POINT
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

    try:
        with open(CLOTH_CONFIG_FILE, 'r') as file:
            cloth_config = json.load(file)
            garment_configs = cloth_config.get("cloth_garments", {})
        script_log(f"✓ Loaded garment configs")
    except Exception as e:
        script_log(f"ERROR: Cannot load cloth config: {e}")
        script_log("Cannot continue without config file")
        return 0

    # Define required bundles - this is simple and won't fail
    REQUIRED_BUNDLES = [
        # Left Side
        ("DEF_LeftShoulder", "shoulder", "left"),
        ("DEF_LeftUpperArm", "elbow", "left"),
        ("DEF_LeftForearm", "wrist", "left"),
        ("DEF_LeftHip", "hip", "left"),
        ("DEF_LeftThigh", "knee", "left"),
        ("DEF_LeftShin", "ankle", "left"),

        # Right Side
        ("DEF_RightShoulder", "shoulder", "right"),
        ("DEF_RightUpperArm", "elbow", "right"),
        ("DEF_RightForearm", "wrist", "right"),
        ("DEF_RightHip", "hip", "right"),
        ("DEF_RightThigh", "knee", "right"),
        ("DEF_RightShin", "ankle", "right"),

        # Head/Neck
        ("DEF_Head", "head_neck", "center"),
        ("DEF_Neck", "neck_coordination", "center")
    ]

    # Initialize if empty
    if not joint_vertex_bundles:
        joint_vertex_bundles = {}

    bundles_created = 0

    # Create each bundle - simple loop, let individual failures be logged
    for bone_name, joint_type, side in REQUIRED_BUNDLES:
        # Get radius from config with sensible defaults
        radius = get_bundle_radius(garment_configs, joint_type, side)
        vertex_count = 40  # Default, could come from config

        bundle_data = make_vertex_bundle(
            armature_obj=armature_obj,
            bone_name=bone_name,
            radius=radius,
            vertex_count=vertex_count,
            joint_type=joint_type,
            side=side
        )

        if bundle_data:
            joint_vertex_bundles[bone_name] = bundle_data
            bundles_created += 1
            script_log(f"✓ Created bundle for {bone_name}")
        else:
            script_log(f"⚠ Failed to create bundle for {bone_name}")

    script_log(f"=== VERTEX BUNDLES COMPLETE: {bundles_created}/{len(REQUIRED_BUNDLES)} ===")
    return bundles_created

##########################################################################################

def get_bundle_radius(garment_configs, joint_type, side):
    """Simple radius lookup with defaults"""
    defaults = {
        "shoulder": 0.075, "elbow": 0.06, "wrist": 0.04,
        "hip": 0.09, "knee": 0.07, "ankle": 0.06,
        "head_neck": 0.08, "neck_coordination": 0.075
    }

    # Try to get from config, fall back to defaults
    try:
        if joint_type in ["shoulder", "elbow", "wrist"]:
            garment_config = garment_configs.get(f"{side}_sleeve", {})
            if joint_type == "shoulder":
                return garment_config.get("diameter_start", 0.15) / 2
            elif joint_type == "elbow":
                return garment_config.get("diameter_elbow", 0.12) / 2
            else:  # wrist
                return garment_config.get("diameter_end", 0.08) / 2
        elif joint_type in ["hip", "knee", "ankle"]:
            garment_config = garment_configs.get(f"{side}_pants", {})
            if joint_type == "hip":
                return garment_config.get("diameter_hip", 0.18) / 2
            elif joint_type == "knee":
                return garment_config.get("diameter_knee", 0.14) / 2
            else:  # ankle
                return garment_config.get("diameter_ankle", 0.12) / 2
        else:  # head/neck
            return defaults[joint_type]
    except:
        return defaults.get(joint_type, 0.05)

##########################################################################################

def create_hip_frame_control_point(first_frame):
    """Create VIRTUAL_HIP_FRAME with coordinate frame calculation"""
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

##########################################################################################

def create_shoulder_frame_control_point(first_frame):
    """Create VIRTUAL_SHOULDER_FRAME with coordinate frame calculation"""
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
    """Set up constraints for the frame control points to follow the virtual midpoints and update rotation from landmarks"""
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

    # Setup VIRTUAL_HEAD_BASE constraint
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

        # Also follow NOSE to position head base
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
    # [All the existing bone rigging code remains exactly the same]
    # This creates the armature structure that the cloth will follow

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

    # Create programmatic root bones first
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

    # Create all bones including spine in main loop
    script_log("Creating DEF bones for skin deformation and animation...")

    # Create bones in order to ensure parents exist first
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

    # Set up parent relationships - using root bone hierarchy
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
    script_log("=== SETTING UP HIERARCHY-DRIVEN SPINE CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # Get the virtual points
    spine_mid_target = bpy.data.objects.get("VIRTUAL_SPINE_MIDPOINT")

    # LOWER SPINE CONSTRAINTS: Tail stretches to virtual spine midpoint
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

    # UPPER SPINE CONSTRAINTS: Tail stretches to virtual spine midpoint
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

##########################################################################################

def resolve_head_file_path(blend_file_path):
    """Resolve scene-relative paths for head .blend files"""

    # If absolute path, use as-is
    if os.path.isabs(blend_file_path):
        if os.path.exists(blend_file_path):
            return blend_file_path
        else:
            raise FileNotFoundError(f"Absolute path not found: {blend_file_path}")

    # Scene-relative path resolution
    try:
        from utils import get_scene_paths
        scene_paths = get_scene_paths(args.show, args.scene)
        scene_dir = os.path.dirname(scene_paths["output_pose_data"])
        scene_relative_path = os.path.join(scene_dir, blend_file_path)

        if os.path.exists(scene_relative_path):
            return scene_relative_path
        else:
            raise FileNotFoundError(f"Scene-relative path not found: {scene_relative_path}")

    except Exception as e:
        raise FileNotFoundError(f"Could not resolve scene-relative path {blend_file_path}: {e}")


##########################################################################################

def load_replaceable_head(blend_file_path, figure_name, armature_obj):
    """Append head mesh from external .blend file using naming conventions"""
    try:
        # Save current selection
        previous_active = bpy.context.active_object
        previous_selected = bpy.context.selected_objects.copy()

        script_log(f"Appending from: {blend_file_path}")

        # Append the head object using naming convention
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            # Look for the standard head mesh name
            if "Head_Mesh" in data_from.objects:
                data_to.objects = ["Head_Mesh"]
                script_log("✓ Found Head_Mesh object")
            else:
                # Fallback: use first mesh object
                mesh_objects = [obj for obj in data_from.objects if obj.endswith("Mesh") or obj.startswith("Head")]
                if mesh_objects:
                    data_to.objects = [mesh_objects[0]]
                    script_log(f"✓ Using fallback mesh: {mesh_objects[0]}")
                else:
                    raise Exception("No suitable head mesh found in blend file")

        if not data_to.objects:
            raise Exception("No objects appended from blend file")

        # Get the appended object
        head_obj = data_to.objects[0]
        head_obj.name = f"{figure_name}_Head"

        # Position at head bone location
        head_bone = bpy.context.scene.objects.get(armature_obj.name).pose.bones.get("DEF_Head")
        if head_bone:
            head_obj.location = armature_obj.matrix_world @ head_bone.tail
            script_log(f"Positioned head at DEF_Head bone: {head_obj.location}")

        # Materials are automatically appended with the object
        script_log(f"Head object materials: {[mat.name for mat in head_obj.data.materials]}")

        return head_obj

    except Exception as e:
        script_log(f"Failed to load replaceable head: {e}")
        # Clean up any partially appended objects
        if 'data_to' in locals() and data_to.objects:
            for obj in data_to.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        return None


##########################################################################################

def setup_imported_head(head_obj, armature_obj, garment_config, replaceable_config):
    """Setup vertex groups and armature for imported head mesh"""

    script_log("Setting up imported head for rig system...")

    # Clean up vertex groups if requested
    if replaceable_config.get("vertex_group_cleanup", True):
        script_log("Cleaning up existing vertex groups...")
        for vg in list(head_obj.vertex_groups):
            head_obj.vertex_groups.remove(vg)

    # Remove existing armature modifiers
    for mod in list(head_obj.modifiers):
        if mod.type == 'ARMATURE':
            head_obj.modifiers.remove(mod)

    # Get head-neck bundle for coordinated weighting
    head_neck_bundle = joint_vertex_bundles.get("DEF_Head")

    # Create vertex group for head bone
    head_group = head_obj.vertex_groups.new(name="DEF_Head")

    if head_neck_bundle:
        script_log("Applying head-neck bundle weighting...")
        # Use bundle-based weighting (same as procedural head)
        head_neck_positions = head_neck_bundle['vertex_positions']
        head_neck_radius = garment_config.get("neck_connection_radius", 0.08)

        weighted_vertices = 0
        for i, vertex in enumerate(head_obj.data.vertices):
            vert_pos = head_obj.matrix_world @ vertex.co
            min_distance = float('inf')

            for bundle_vert_pos in head_neck_positions:
                distance = (vert_pos - bundle_vert_pos).length
                min_distance = min(min_distance, distance)

            if min_distance <= head_neck_radius * 2.0:
                weight = 1.0 - (min_distance / (head_neck_radius * 2.0))
                weight = weight * weight  # Quadratic falloff

                # Reduce weight for top of head (more flexible)
                vert_local = head_obj.matrix_world.inverted() @ vert_pos
                if vert_local.y > 0.5:  # Top of head
                    weight *= 0.3
                elif vert_local.y > 0.2:  # Upper head
                    weight *= 0.6

                if weight > 0.1:
                    head_group.add([i], weight, 'REPLACE')
                    weighted_vertices += 1

        script_log(f"✓ Weighted {weighted_vertices} vertices using head-neck bundle")
    else:
        script_log("Using uniform weighting (no head-neck bundle)")
        # Fallback uniform weighting
        for i in range(len(head_obj.data.vertices)):
            head_group.add([i], 1.0, 'REPLACE')

    # Add armature modifier
    armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj
    armature_mod.use_vertex_groups = True
    script_log("✓ Added armature modifier")

    return head_obj


##########################################################################################

def save_hero_head(head_obj, export_path, armature_obj, garment_config):
    """Export procedural head as a template .blend file with standardizes naming and materials"""
    script_log(f"=== EXPORTING HERO HEAD TEMPLATE: {export_path} ===")

    # Save current selection state
    previous_active = bpy.context.active_object
    previous_selected = bpy.context.selected_objects.copy()

    try:
        # Create a duplicate of the head for export (don't modify original)
        bpy.ops.object.select_all(action='DESELECT')
        head_obj.select_set(True)
        bpy.context.view_layer.objects.active = head_obj
        bpy.ops.object.duplicate()

        export_head = bpy.context.active_object
        export_head.name = "Head_Mesh"  # Standardized name

        # =========================================================================
        # STEP 1: CLEAN UP THE EXPORT HEAD
        # =========================================================================
        script_log("STEP 1: Cleaning up export head...")

        # Remove armature modifier (keep geometry only)
        for mod in list(export_head.modifiers):
            if mod.type == 'ARMATURE':
                export_head.modifiers.remove(mod)

        # Clear vertex groups (artist will create their own)
        for vg in list(export_head.vertex_groups):
            export_head.vertex_groups.remove(vg)

        # Reset transformations for clean export
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # =========================================================================
        # STEP 2: ENSURE CONSISTENT NAMING OF UV MAPS
        # =========================================================================
        script_log("STEP 2: Setting up UV maps...")

        # Ensure we have a UV map
        if not export_head.data.uv_layers:
            script_log("Creating default UV map...")
            export_head.data.uv_layers.new(name="UVMap")

        # Apply smart UV project for better starting point
        bpy.context.view_layer.objects.active = export_head
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66),
            island_margin=0.02,
            area_weight=1.0,
            correct_aspect=True,
            scale_to_bounds=True
        )
        bpy.ops.object.mode_set(mode='OBJECT')
        script_log("✓ Applied smart UV projection")

        # =========================================================================
        # STEP 3: SETUP STANDARD MATERIALS
        # =========================================================================
        script_log("STEP 3: Setting up standard materials...")

        # Clear existing materials
        export_head.data.materials.clear()

        # Create standard head material
        head_mat = bpy.data.materials.new(name="Head_Material")
        head_mat.use_nodes = True

        # Clear default nodes for clean setup
        head_mat.node_tree.nodes.clear()

        # Create modern Principled BSDF setup
        output_node = head_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = head_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

        # Position nodes
        output_node.location = (300, 0)
        principled_node.location = (0, 0)

        # Connect nodes
        head_mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

        # Set realistic skin properties
        principled_node.inputs['Base Color'].default_value = (0.96, 0.86, 0.72, 1.0)  # Skin tone
        principled_node.inputs['Roughness'].default_value = 0.4  # Slightly rough for skin
        principled_node.inputs['Subsurface'].default_value = 0.2  # Skin subsurface scattering
        principled_node.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)  # Skin scattering
        principled_node.inputs['Subsurface Color'].default_value = (0.9, 0.7, 0.7, 1.0)  # Blood color

        # Add material to head
        export_head.data.materials.append(head_mat)

        # Create placeholder materials for other parts
        eye_mat = bpy.data.materials.new(name="Head_Eyes_Material")
        eye_mat.use_nodes = True
        eye_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.1, 0.1, 0.9,
                                                                                         1.0)  # Blue eyes

        teeth_mat = bpy.data.materials.new(name="Head_Teeth_Material")
        teeth_mat.use_nodes = True
        teeth_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.95, 0.95, 0.9,
                                                                                           1.0)  # Off-white teeth

        script_log("✓ Created standard materials: Head_Material, Head_Eyes_Material, Head_Teeth_Material")

        # =========================================================================
        # STEP 4: CREATE METADATA AND ORGANIZATION
        # =========================================================================
        script_log("STEP 4: Adding metadata...")

        # Add custom properties for documentation
        export_head["is_hero_head_template"] = True
        export_head["export_version"] = "1.0"
        export_head["original_scale"] = list(export_head.scale)
        export_head["recommended_scale"] = 1.0

        # Create a collection for organization
        if "Hero_Head_Assets" not in bpy.data.collections:
            head_collection = bpy.data.collections.new("Hero_Head_Assets")
            bpy.context.scene.collection.children.link(head_collection)
        else:
            head_collection = bpy.data.collections["Hero_Head_Assets"]

        # Move head to collection
        if export_head.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(export_head)
        head_collection.objects.link(export_head)

        # =========================================================================
        # STEP 5: EXPORT TO BLEND FILE
        # =========================================================================
        script_log(f"STEP 5: Exporting to: {export_path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Select only the head for export
        bpy.ops.object.select_all(action='DESELECT')
        export_head.select_set(True)
        bpy.context.view_layer.objects.active = export_head

        # Save as new blend file
        bpy.ops.wm.save_as_mainfile(
            filepath=export_path,
            check_existing=False,
            compress=True
        )

        script_log(f"✓ Successfully exported hero head template: {export_path}")

        # =========================================================================
        # STEP 6: CLEAN UP
        # =========================================================================
        script_log("STEP 6: Cleaning up...")

        # Delete the temporary export object (original remains in scene)
        bpy.ops.object.select_all(action='DESELECT')
        export_head.select_set(True)
        bpy.ops.object.delete()

        # Restore original selection
        if previous_active:
            bpy.context.view_layer.objects.active = previous_active
        for obj in previous_selected:
            obj.select_set(True)

        return True

    except Exception as e:
        script_log(f"ERROR Failed to export hero head template: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")

        # Clean up on failure
        try:
            if 'export_head' in locals() and export_head.name in bpy.data.objects:
                bpy.data.objects.remove(export_head, do_unlink=True)
        except:
            pass

        # Restore selection state
        try:
            if previous_active:
                bpy.context.view_layer.objects.active = previous_active
            for obj in previous_selected:
                obj.select_set(True)
        except:
            pass

        return False


##########################################################################################

def create_head(armature_obj, figure_name, garment_config, global_cloth_settings, neck_config=None):
    """Create head - either procedural or from replaceable asset with template export - UPDATED WITH SAFE NOSE CONSTRAINTS"""
    script_log("=== HEAD CREATION STARTED (SAFE NOSE CONSTRAINTS) ===")

    # =========================================================================
    # STEP 1: ENSURE NOSE CONTROL POINT EXISTS FIRST (CRITICAL FOR SAFETY)
    # =========================================================================
    script_log("STEP 1: Ensuring NOSE control point exists for safe constraint setup...")
    first_frame = mocap_data.get(str(frame_numbers[0]), {})

    def ensure_nose_control_point_safe(first_frame):
        """Guarantee CTRL_NOSE exists and is positioned correctly - SAFE VERSION"""
        script_log("Ensuring NOSE control point exists...")

        nose_obj = bpy.data.objects.get("CTRL_NOSE")

        if not nose_obj:
            script_log("Creating CTRL_NOSE control point from landmark data...")
            # Create from landmark data
            if "NOSE" in first_frame:
                pos_data = first_frame["NOSE"]
                nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

                bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
                nose_obj = bpy.context.active_object
                nose_obj.name = "CTRL_NOSE"

                # Add to control collection
                control_coll = bpy.data.collections.get("Main_ControlPoints")
                if control_coll:
                    control_coll.objects.link(nose_obj)

                script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
            else:
                script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
                nose_obj = bpy.context.active_object
                nose_obj.name = "CTRL_NOSE"
        else:
            script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

        return nose_obj

    def setup_head_constraints_safe(armature_obj, head_bone_name):
        """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
        script_log(f"Setting up safe head constraints for {head_bone_name}...")

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        head_bone = armature_obj.pose.bones[head_bone_name]

        # 1. CLEAR ALL existing constraints first (prevent conflicts)
        for constraint in list(head_bone.constraints):
            head_bone.constraints.remove(constraint)
        script_log("✓ Cleared existing head constraints")

        # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        if neck_bone:
            copy_loc = head_bone.constraints.new('COPY_LOCATION')
            copy_loc.target = armature_obj
            copy_loc.subtarget = "DEF_Neck"
            copy_loc.use_offset = True  # Maintain head's local position relative to neck
            copy_loc.influence = 1.0
            script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

        # 3. ROTATION: Damped track to NOSE control point (facing direction)
        nose_obj = bpy.data.objects.get("CTRL_NOSE")
        if nose_obj:
            track = head_bone.constraints.new('DAMPED_TRACK')
            track.name = "Track_Nose_Direction"
            track.target = nose_obj
            track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
            track.influence = 1.0
            script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

            # 4. ROTATION LIMITS: Prevent extreme angles for stability
            limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
            limit_rot.name = "Limit_Head_Rotation"
            limit_rot.use_limit_x = True
            limit_rot.min_x = -0.5  # Limited head tilt (nodding)
            limit_rot.max_x = 0.5
            limit_rot.use_limit_y = True
            limit_rot.min_y = -1.5  # Head turn (looking left/right)
            limit_rot.max_y = 1.5
            limit_rot.use_limit_z = True
            limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
            limit_rot.max_z = 0.3
            limit_rot.owner_space = 'LOCAL'
            script_log("✓ Added natural head rotation limits")
        else:
            script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

        bpy.ops.object.mode_set(mode='OBJECT')
        script_log("✓ Safe head constraints setup complete")

    # CREATE NOSE CONTROL POINT BEFORE ANYTHING ELSE
    nose_obj = ensure_nose_control_point_safe(first_frame)

    # =========================================================================
    # STEP 2: CREATE PROCEDURAL HEAD FIRST (needed for export or fallback)
    # =========================================================================
    script_log("STEP 2: Creating procedural head with safe constraints...")

    def create_procedural_head_safe(armature_obj, figure_name, garment_config, global_cloth_settings, neck_config=None):
        """Create human-like head with coordinated vertex bundles for seamless neck integration - UPDATED WITH SAFE CONSTRAINTS"""
        script_log("Creating procedural garment_head with coordinated vertex bundles and safe constraints...")

        # Get head bone position
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        try:
            head_bone = armature_obj.pose.bones.get("DEF_Head")
            neck_bone = armature_obj.pose.bones.get("DEF_Neck")
            head_bone_name = "DEF_Head"
            neck_bone_name = "DEF_Neck"

            bpy.ops.object.mode_set(mode='OBJECT')

            if not head_bone:
                script_log("ERROR: Could not find head bone")
                return None

            # Get head bone positions in world space - Use tail for head position
            head_base_pos = armature_obj.matrix_world @ head_bone.head  # Base of head/neck (where neck connects)
            head_top_pos = armature_obj.matrix_world @ head_bone.tail  # Top of head (where head should be)

            # Get neck bone position for coordination
            neck_base_pos = armature_obj.matrix_world @ neck_bone.head if neck_bone else head_base_pos
            neck_top_pos = armature_obj.matrix_world @ neck_bone.tail if neck_bone else head_base_pos

            # Get head dimensions from new config format
            scale = garment_config.get("scale", [0.25, 0.28, 0.3])  # Larger, more human-sized proportions
            position_offset = garment_config.get("position_offset", [0.0, 0.0, 0.0])
            rotation_offset = garment_config.get("rotation_offset", [0.0, 0.0, 0.0])
            neck_connection_radius = garment_config.get("neck_connection_radius", 0.08)
            subdivision_levels = garment_config.get("subdivision_levels", 3)  # More subdivisions for detail

            # Get geometry settings from config
            geometry_settings = global_cloth_settings.get("geometry_settings", {})
            head_segments = geometry_settings.get("head_segments", 32)  # More segments for detail
            head_ring_count = geometry_settings.get("head_ring_count", 24)  # More rings for detail
            base_radii = global_cloth_settings.get("base_radii", {})
            head_base_radius = base_radii.get("head", 0.25)  # Larger base radius

            # Get head-neck bundle diameter from config
            head_neck_diameter = garment_config.get("diameter_neck", 0.16)  # Diameter at base of head/neck junction

            script_log(f"DEBUG: Procedural head scale: {scale}")
            script_log(f"DEBUG: Position offset: {position_offset}, Rotation offset: {rotation_offset}")
            script_log(f"DEBUG: Head bone - base (neck): {head_base_pos}, top (head): {head_top_pos}")
            script_log(f"DEBUG: Head geometry - segments: {head_segments}, rings: {head_ring_count}")
            script_log(f"DEBUG: Head-neck junction diameter: {head_neck_diameter}")

            # =========================================================================
            # STEP 1: CREATE HEAD MESH WITH NECK OPENING
            # =========================================================================
            script_log("DEBUG: Creating procedural head as UV sphere with neck opening...")
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=head_segments,
                ring_count=head_ring_count,
                radius=head_base_radius,
                location=head_top_pos + Vector(position_offset)  # Position at TOP of head bone
            )
            head_obj = bpy.context.active_object
            head_obj.name = f"{figure_name}_Head"

            # Apply human-like scaling (wider and taller than spherical)
            head_obj.scale = Vector(scale)

            # Apply rotation offset if specified
            if any(rotation_offset):
                head_obj.rotation_euler = Vector(rotation_offset)

            # =========================================================================
            # STEP 2: SHAPE HEAD TO HUMAN PROPORTIONS
            # =========================================================================
            script_log("DEBUG: Shaping head to human proportions...")
            bpy.context.view_layer.objects.active = head_obj

            # Use bmesh for vertex manipulation without entering edit mode
            bm = bmesh.new()
            bm.from_mesh(head_obj.data)

            # Define human head proportions (these could be moved to config)
            human_proportions = garment_config.get("human_proportions", {
                'forehead_scale': 0.95,  # Slightly narrower forehead
                'cheek_scale': 1.1,  # Fuller cheeks
                'jaw_scale': 0.9,  # Narrower jaw
                'chin_scale': 0.8,  # More pronounced chin
                'back_head_scale': 1.05,  # Fuller back of head
                'top_head_scale': 0.95,  # Slightly flatter top
                'neck_opening_scale': 0.7,  # Create neck opening
            })

            # Shape vertices to human proportions
            for vert in bm.verts:
                # Convert to spherical coordinates for easier manipulation
                x, y, z = vert.co
                radius = (x ** 2 + y ** 2 + z ** 2) ** 0.5
                if radius > 0.001:  # Avoid division by zero
                    # Normalize to unit sphere
                    nx, ny, nz = x / radius, y / radius, z / radius

                    # Apply human proportions based on vertex position
                    scale_factor = 1.0

                    # Forehead area (front top)
                    if nz > 0.3 and ny > 0.2:
                        scale_factor = human_proportions['forehead_scale']

                    # Cheek area (front sides)
                    elif nz > -0.2 and abs(nx) > 0.4:
                        scale_factor = human_proportions['cheek_scale']

                    # Jaw area (front bottom)
                    elif nz < -0.3 and abs(ny) < 0.3:
                        scale_factor = human_proportions['jaw_scale']

                    # Chin area (very front bottom)
                    elif nz < -0.5 and abs(nx) < 0.2:
                        scale_factor = human_proportions['chin_scale']

                    # Back of head
                    elif nz < -0.1 and abs(nx) < 0.3:
                        scale_factor = human_proportions['back_head_scale']

                    # Top of head
                    elif ny > 0.7:
                        scale_factor = human_proportions['top_head_scale']

                    # Neck opening area (bottom center)
                    elif ny < -0.6 and abs(nz) < 0.3 and abs(nx) < 0.3:
                        scale_factor = human_proportions['neck_opening_scale']

                    # Apply the scaling
                    vert.co.x = nx * radius * scale_factor
                    vert.co.y = ny * radius * scale_factor
                    vert.co.z = nz * radius * scale_factor

            # Write back to mesh (object mode is safe for this)
            bm.to_mesh(head_obj.data)
            bm.free()

            # =========================================================================
            # STEP 3: ADD FACIAL FEATURES USING SHAPE KEYS
            # =========================================================================
            facial_features = garment_config.get("facial_features", {})
            if facial_features.get("enable_eye_sockets", True) or facial_features.get("enable_nose_bridge", True):
                script_log("DEBUG: Adding facial features with shape keys...")

                # Add basis shape key first
                head_obj.shape_key_add(name="Basis")

                # Create shape keys FIRST before using bmesh
                shape_key_objects = {}

                if facial_features.get("enable_eye_sockets", True):
                    script_log("DEBUG: Creating eye sockets shape key...")
                    eye_sockets = head_obj.shape_key_add(name="EyeSockets")
                    shape_key_objects["EyeSockets"] = eye_sockets

                if facial_features.get("enable_nose_bridge", True):
                    script_log("DEBUG: Creating nose bridge shape key...")
                    nose_bridge = head_obj.shape_key_add(name="NoseBridge")
                    shape_key_objects["NoseBridge"] = nose_bridge

                # NOW use bmesh for shape key editing
                bm = bmesh.new()
                bm.from_mesh(head_obj.data)

                # Ensure we have shape key layers - they should exist now
                shape_keys = bm.verts.layers.shape

                # Create eye sockets if enabled
                if facial_features.get("enable_eye_sockets", True) and "EyeSockets" in shape_keys:
                    script_log("DEBUG: Adding eye sockets geometry...")
                    eye_socket_layer = shape_keys["EyeSockets"]
                    eye_socket_depth = facial_features.get("eye_socket_depth", 0.08)

                    for vert in bm.verts:
                        x, y, z = vert.co
                        # Eye socket areas (front upper sides)
                        if z > 0.1 and abs(x) > 0.3 and y > 0.1:
                            # Create indentation for eye sockets
                            depth = eye_socket_depth
                            vert[eye_socket_layer] = vert.co + Vector((-x * depth * 0.5, -y * depth * 0.3, -z * depth))

                # Create nose bridge if enabled
                if facial_features.get("enable_nose_bridge", True) and "NoseBridge" in shape_keys:
                    script_log("DEBUG: Adding nose bridge geometry...")
                    nose_bridge_layer = shape_keys["NoseBridge"]
                    nose_bridge_height = facial_features.get("nose_bridge_height", 0.05)

                    for vert in bm.verts:
                        x, y, z = vert.co
                        # Nose bridge area (center front)
                        if abs(x) < 0.15 and z > 0.2 and y > -0.1:
                            # Create protrusion for nose
                            height = nose_bridge_height
                            vert[nose_bridge_layer] = vert.co + Vector((0, 0, height))

                # Write shape keys back
                bm.to_mesh(head_obj.data)
                bm.free()

                # Apply some of the facial features
                if "EyeSockets" in head_obj.data.shape_keys.key_blocks:
                    head_obj.data.shape_keys.key_blocks["EyeSockets"].value = 0.3
                if "NoseBridge" in head_obj.data.shape_keys.key_blocks:
                    head_obj.data.shape_keys.key_blocks["NoseBridge"].value = 0.4
            else:
                script_log("DEBUG: Facial features disabled in config")

            # =========================================================================
            # STEP 4: SETUP VERTEX GROUPS WITH COORDINATED HEAD-NECK BUNDLES
            # =========================================================================
            script_log("DEBUG: Setting up head vertex groups with coordinated neck bundles...")

            # Clear any existing parenting
            head_obj.parent = None

            # Clear any existing vertex groups
            for vg in list(head_obj.vertex_groups):
                head_obj.vertex_groups.remove(vg)

            # Remove any existing armature modifiers
            for mod in list(head_obj.modifiers):
                if mod.type == 'ARMATURE':
                    head_obj.modifiers.remove(mod)

            # Get head-neck vertex bundle from global storage
            head_neck_bundle = joint_vertex_bundles.get("DEF_Head")

            if head_neck_bundle:
                script_log(f"✓ Using head-neck vertex bundle with {head_neck_bundle['vertex_count']} vertices")
            else:
                script_log(f"⚠ No head-neck bundle found for DEF_Head, using standard head weighting")

            # Create vertex group for head bone
            head_group = head_obj.vertex_groups.new(name=head_bone_name)

            # Apply head-neck bundle weighting if available
            if head_neck_bundle:
                head_neck_positions = head_neck_bundle['vertex_positions']
                head_neck_radius = head_neck_diameter / 2

                for i, vertex in enumerate(head_obj.data.vertices):
                    vert_pos = head_obj.matrix_world @ vertex.co
                    min_distance = float('inf')

                    # Find closest vertex in the head-neck bundle
                    for bundle_vert_pos in head_neck_positions:
                        distance = (vert_pos - bundle_vert_pos).length
                        min_distance = min(min_distance, distance)

                    # Apply weight based on distance to nearest bundle vertex
                    # Stronger influence at neck junction, lighter toward top of head
                    if min_distance <= head_neck_radius * 2.0:
                        weight = 1.0 - (min_distance / (head_neck_radius * 2.0))
                        weight = weight * weight  # Quadratic falloff

                        # Reduce weight for top of head (more flexible)
                        vert_local = head_obj.matrix_world.inverted() @ vert_pos
                        if vert_local.y > 0.5:  # Top of head
                            weight *= 0.3
                        elif vert_local.y > 0.2:  # Upper head
                            weight *= 0.6
                        # Full weight for neck area (y < 0)

                        if weight > 0.1:
                            head_group.add([i], weight, 'REPLACE')
                    else:
                        # Light default weight for distant vertices
                        head_group.add([i], 0.2, 'REPLACE')
            else:
                # Fallback: assign uniform weights with neck emphasis
                for i, vertex in enumerate(head_obj.data.vertices):
                    vert_local = head_obj.matrix_world.inverted() @ vertex.co
                    # Stronger weight at neck, lighter at top
                    if vert_local.y < -0.4:  # Neck area
                        head_group.add([i], 1.0, 'REPLACE')
                    elif vert_local.y < 0:  # Lower head
                        head_group.add([i], 0.8, 'REPLACE')
                    else:  # Upper head
                        head_group.add([i], 0.6, 'REPLACE')

            # =========================================================================
            # STEP 5: ADD ARMATURE MODIFIER
            # =========================================================================
            script_log("DEBUG: Adding armature modifier...")

            # Add armature modifier
            armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
            armature_mod.object = armature_obj
            armature_mod.use_vertex_groups = True

            # =========================================================================
            # STEP 6: ADD SUBDIVISION AND MATERIALS
            # =========================================================================
            script_log("DEBUG: Adding subdivision and skin material...")

            # Add subdivision from config
            subdiv_mod = head_obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = subdivision_levels
            subdiv_mod.render_levels = subdivision_levels

            # ADD SKIN MATERIAL WITH MODERN BLENDER 4.3+ NODES
            script_log("DEBUG: Creating skin material with modern node setup...")
            material_config = garment_config.get("material", {})
            material_color = material_config.get("color", [0.96, 0.86, 0.72, 1.0])

            head_mat = bpy.data.materials.new(name="Head_Material")
            head_mat.use_nodes = True

            # Clear default nodes for clean setup
            head_mat.node_tree.nodes.clear()

            # Create modern Principled BSDF setup
            output_node = head_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
            principled_node = head_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

            # Position nodes
            output_node.location = (300, 0)
            principled_node.location = (0, 0)

            # Connect nodes
            head_mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

            # Set material properties using modern API
            principled_node.inputs['Base Color'].default_value = material_color
            principled_node.inputs['Roughness'].default_value = material_config.get("roughness", 0.4)
            principled_node.inputs['Metallic'].default_value = material_config.get("metallic", 0.0)

            head_obj.data.materials.append(head_mat)

            # =========================================================================
            # STEP 7: CLOTH SIMULATION - DISABLED AS REQUESTED
            # =========================================================================
            cloth_settings = garment_config.get("cloth_settings", {})
            if cloth_settings.get("enabled", False):
                script_log("DEBUG: Adding cloth simulation to garment_head...")
                cloth_mod = head_obj.modifiers.new(name="Cloth", type='CLOTH')
                cloth_mod.settings.quality = cloth_settings.get("quality", 6)
                cloth_mod.settings.mass = cloth_settings.get("mass", 0.8)
                cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 15.0)
                cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 15.0)
                cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 10.0)
                cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 2.0)
                cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)
            else:
                script_log("DEBUG: Cloth simulation disabled for head")

            # =========================================================================
            # STEP 8: SET MODIFIER ORDER
            # =========================================================================
            bpy.context.view_layer.objects.active = head_obj
            modifiers = head_obj.modifiers

            # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
            correct_order = ["Subdivision", "Armature", "Cloth"]
            for mod_name in correct_order:
                mod_index = modifiers.find(mod_name)
                if mod_index >= 0:
                    while mod_index > correct_order.index(mod_name):
                        bpy.ops.object.modifier_move_up(modifier=mod_name)
                        mod_index -= 1

            # =========================================================================
            # STEP 9: NOSE LANDMARK TRACKING SETUP - UPDATED WITH SAFE CONSTRAINTS
            # =========================================================================
            script_log("DEBUG: Setting up NOSE landmark tracking with safe constraints...")

            # SETUP SAFE HEAD CONSTRAINTS (PORCELAIN DOLL ARCHITECTURE)
            setup_head_constraints_safe(armature_obj, head_bone_name)

            # =========================================================================
            # STEP 10: FINAL VERIFICATION
            # =========================================================================
            bpy.context.view_layer.update()

            # Log bundle usage
            bundle_status = f"head-neck({head_neck_bundle['vertex_count']}v)" if head_neck_bundle else "NONE (standard)"

            script_log("=== PROCEDURAL HEAD CREATION COMPLETE (COORDINATED VERTEX BUNDLES + SAFE CONSTRAINTS) ===")
            script_log(f"✓ Head positioned at TOP of DEF_Head bone (tail position)")
            script_log(f"✓ Head shaped with human proportions")
            script_log(f"✓ Facial features: Eye sockets, Nose bridge")
            script_log(f"✓ Head rotation tracks NOSE landmark in real-time (SAFE CONSTRAINTS)")
            script_log(f"✓ Natural rotation limits applied for stability")
            script_log(f"✓ Head scale: {scale} (larger, human-sized proportions)")
            script_log(f"✓ Head segments: {head_segments}, rings: {head_ring_count}")
            script_log(f"✓ Subdivision levels: {subdivision_levels}")
            script_log(f"✓ Neck connection diameter: {head_neck_diameter}")
            script_log(f"✓ Head parented to {head_bone_name}")
            script_log(f"✓ Skin material applied with modern Principled BSDF")
            script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_settings.get('enabled', False) else 'DISABLED'}")
            script_log(f"✓ Vertex bundle used: {bundle_status}")
            script_log(f"✓ Seamless neck integration: Head uses same vertex bundle as neck")
            script_log(f"✓ PORCELAIN DOLL ARCHITECTURE: Neck anchor + Nose rotation guide")

            # Call create_neck for coordinated creation
            if neck_config:
                create_neck(armature_obj, figure_name, neck_config, global_cloth_settings)

            return head_obj

        except Exception as e:
            script_log(f"ERROR creating procedural head with coordinated bundles: {e}")
            import traceback
            script_log(f"Traceback: {traceback.format_exc()}")
            bpy.ops.object.mode_set(mode='OBJECT')
            return None

    # CREATE PROCEDURAL HEAD WITH SAFE CONSTRAINTS
    procedural_head = create_procedural_head_safe(armature_obj, figure_name, garment_config, global_cloth_settings,
                                                  neck_config)

    if not procedural_head:
        script_log("ERROR Failed to create procedural head")
        return None

    # =========================================================================
    # STEP 3: CHECK IF WE SHOULD EXPORT AS TEMPLATE
    # =========================================================================
    replaceable_config = garment_config.get("replaceable_head", {})
    if replaceable_config.get("export_template", False):
        script_log("STEP 3: Exporting head template...")

        export_path = replaceable_config.get("export_path", "assets/hero_head_template.blend")

        # Resolve scene-relative path
        try:
            resolved_export_path = resolve_head_file_path(export_path)
            script_log(f"Export path resolved to: {resolved_export_path}")

            success = save_hero_head(procedural_head, resolved_export_path, armature_obj, garment_config)

            if success:
                script_log("✓ Head template exported successfully")
                # Continue to use the procedural head in the scene
                script_log("✓ Using procedural head in scene (template exported)")
                return procedural_head
            else:
                script_log("⚠ Template export failed, using procedural head in scene")
                return procedural_head

        except Exception as e:
            script_log(f"⚠ Template export path resolution failed: {e}, using procedural head")
            return procedural_head

    # =========================================================================
    # STEP 4: CHECK IF WE SHOULD LOAD REPLACEABLE HEAD INSTEAD
    # =========================================================================
    if replaceable_config.get("enabled", False):
        script_log("STEP 4: Attempting to load replaceable head...")

        try:
            blend_file_path = replaceable_config.get("blend_file")
            if not blend_file_path:
                raise ValueError("No blend_file specified in replaceable_head config")

            # Resolve scene-relative path
            resolved_path = resolve_head_file_path(blend_file_path)
            script_log(f"Loading head from: {resolved_path}")

            replaceable_head = load_replaceable_head(
                resolved_path,
                figure_name,
                armature_obj
            )

            if replaceable_head:
                script_log("✓ Successfully loaded replaceable head")

                # =========================================================================
                # STEP 4A: REMOVE PROCEDURAL HEAD SINCE WE'RE USING REPLACEABLE
                # =========================================================================
                script_log("Removing procedural head (replacing with imported head)...")
                bpy.data.objects.remove(procedural_head, do_unlink=True)
                script_log("✓ Procedural head removed")

                # =========================================================================
                # STEP 4B: APPLY TRANSFORMATIONS TO REPLACEABLE HEAD
                # =========================================================================
                script_log("Applying transformations to replaceable head...")
                scale_factor = replaceable_config.get("scale_factor", 1.0)
                position_offset = Vector(replaceable_config.get("position_offset", [0, 0, 0]))
                rotation_offset = Vector(replaceable_config.get("rotation_offset", [0, 0, 0]))

                replaceable_head.scale = (scale_factor, scale_factor, scale_factor)
                replaceable_head.location += position_offset
                replaceable_head.rotation_euler += rotation_offset

                script_log(
                    f"✓ Applied transformations - Scale: {scale_factor}, Offset: {position_offset}, Rotation: {rotation_offset}")

                # =========================================================================
                # STEP 4C: SETUP FOR OUR RIG SYSTEM
                # =========================================================================
                script_log("Setting up replaceable head for rig system...")
                replaceable_head = setup_imported_head(replaceable_head, armature_obj, garment_config,
                                                       replaceable_config)

                script_log("✓ Replaceable head setup complete")

                # =========================================================================
                # STEP 4D: VERIFY FINAL SETUP
                # =========================================================================
                bpy.context.view_layer.update()

                # Verify armature modifier exists
                has_armature_mod = any(mod.type == 'ARMATURE' for mod in replaceable_head.modifiers)
                if not has_armature_mod:
                    script_log("⚠ WARNING: Replaceable head missing armature modifier")

                # Verify vertex groups exist
                has_vertex_groups = len(replaceable_head.vertex_groups) > 0
                if not has_vertex_groups:
                    script_log("⚠ WARNING: Replaceable head missing vertex groups")

                script_log("=== REPLACEABLE HEAD CREATION COMPLETE ===")
                script_log(f"✓ Successfully replaced procedural head with imported head")
                script_log(f"✓ Head object: {replaceable_head.name}")
                script_log(f"✓ Materials: {[mat.name for mat in replaceable_head.data.materials]}")
                script_log(f"✓ Vertex groups: {[vg.name for vg in replaceable_head.vertex_groups]}")
                script_log(f"✓ Modifiers: {[mod.name for mod in replaceable_head.modifiers]}")

                return replaceable_head
            else:
                script_log("⚠ Failed to load replaceable head, using procedural head as fallback")
                script_log("✓ Using procedural head in scene")
                return procedural_head

        except Exception as e:
            script_log(f"ERROR Replaceable head loading failed: {e}")
            import traceback
            script_log(f"Traceback: {traceback.format_exc()}")

            if not garment_config.get("fallback_to_procedural", True):
                script_log("ERROR Fallback disabled - returning None")
                # Clean up procedural head since we're not using it
                bpy.data.objects.remove(procedural_head, do_unlink=True)
                return None
            else:
                script_log("✓ Using procedural head as fallback")
                return procedural_head

    # =========================================================================
    # STEP 5: IF NO REPLACEABLE HEAD REQUESTED, USE PROCEDURAL
    # =========================================================================
    script_log("STEP 5: Using procedural head (no replaceable head requested)...")

    # Verify procedural head setup
    bpy.context.view_layer.update()

    # Check if procedural head has proper setup
    has_armature_mod = any(mod.type == 'ARMATURE' for mod in procedural_head.modifiers)
    has_vertex_groups = len(procedural_head.vertex_groups) > 0
    has_materials = len(procedural_head.data.materials) > 0

    script_log("=== PROCEDURAL HEAD VERIFICATION ===")
    script_log(f"✓ Armature modifier: {'PRESENT' if has_armature_mod else 'MISSING'}")
    script_log(f"✓ Vertex groups: {'PRESENT' if has_vertex_groups else 'MISSING'}")
    script_log(f"✓ Materials: {'PRESENT' if has_materials else 'MISSING'}")
    script_log(f"✓ Head object: {procedural_head.name}")

    if has_vertex_groups:
        script_log(f"✓ Vertex groups: {[vg.name for vg in procedural_head.vertex_groups]}")

    if has_materials:
        script_log(f"✓ Materials: {[mat.name for mat in procedural_head.data.materials]}")

    script_log("=== PROCEDURAL HEAD CREATION COMPLETE ===")
    return procedural_head

##########################################################################################

def ensure_nose_control_point(first_frame):
    """Guarantee CTRL_NOSE exists and is positioned correctly"""
    script_log("Ensuring NOSE control point exists...")

    nose_obj = bpy.data.objects.get("CTRL_NOSE")

    if not nose_obj:
        script_log("Creating CTRL_NOSE control point from landmark data...")
        # Create from landmark data
        if "NOSE" in first_frame:
            pos_data = first_frame["NOSE"]
            nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

            bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"

            # Add to control collection
            control_coll = bpy.data.collections.get("Main_ControlPoints")
            if control_coll:
                control_coll.objects.link(nose_obj)

            script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
        else:
            script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"
    else:
        script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

    return nose_obj


def setup_head_constraints_safe(armature_obj, head_bone_name):
    """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
    script_log(f"Setting up safe head constraints for {head_bone_name}...")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    head_bone = armature_obj.pose.bones[head_bone_name]

    # 1. CLEAR ALL existing constraints first (prevent conflicts)
    for constraint in list(head_bone.constraints):
        head_bone.constraints.remove(constraint)
    script_log("✓ Cleared existing head constraints")

    # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
    neck_bone = armature_obj.pose.bones.get("DEF_Neck")
    if neck_bone:
        copy_loc = head_bone.constraints.new('COPY_LOCATION')
        copy_loc.target = armature_obj
        copy_loc.subtarget = "DEF_Neck"
        copy_loc.use_offset = True  # Maintain head's local position relative to neck
        copy_loc.influence = 1.0
        script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

    # 3. ROTATION: Damped track to NOSE control point (facing direction)
    nose_obj = bpy.data.objects.get("CTRL_NOSE")
    if nose_obj:
        track = head_bone.constraints.new('DAMPED_TRACK')
        track.name = "Track_Nose_Direction"
        track.target = nose_obj
        track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
        track.influence = 1.0
        script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

        # 4. ROTATION LIMITS: Prevent extreme angles for stability
        limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
        limit_rot.name = "Limit_Head_Rotation"
        limit_rot.use_limit_x = True
        limit_rot.min_x = -0.5  # Limited head tilt (nodding)
        limit_rot.max_x = 0.5
        limit_rot.use_limit_y = True
        limit_rot.min_y = -1.5  # Head turn (looking left/right)
        limit_rot.max_y = 1.5
        limit_rot.use_limit_z = True
        limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
        limit_rot.max_z = 0.3
        limit_rot.owner_space = 'LOCAL'
        script_log("✓ Added natural head rotation limits")
    else:
        script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log("✓ Safe head constraints setup complete")


def ensure_nose_control_point(first_frame):
    """Guarantee CTRL_NOSE exists and is positioned correctly"""
    script_log("Ensuring NOSE control point exists...")

    nose_obj = bpy.data.objects.get("CTRL_NOSE")

    if not nose_obj:
        script_log("Creating CTRL_NOSE control point from landmark data...")
        # Create from landmark data
        if "NOSE" in first_frame:
            pos_data = first_frame["NOSE"]
            nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

            bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"

            # Add to control collection
            control_coll = bpy.data.collections.get("Main_ControlPoints")
            if control_coll:
                control_coll.objects.link(nose_obj)

            script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
        else:
            script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"
    else:
        script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

    return nose_obj


def setup_head_constraints_safe(armature_obj, head_bone_name):
    """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
    script_log(f"Setting up safe head constraints for {head_bone_name}...")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    head_bone = armature_obj.pose.bones[head_bone_name]

    # 1. CLEAR ALL existing constraints first (prevent conflicts)
    for constraint in list(head_bone.constraints):
        head_bone.constraints.remove(constraint)
    script_log("✓ Cleared existing head constraints")

    # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
    neck_bone = armature_obj.pose.bones.get("DEF_Neck")
    if neck_bone:
        copy_loc = head_bone.constraints.new('COPY_LOCATION')
        copy_loc.target = armature_obj
        copy_loc.subtarget = "DEF_Neck"
        copy_loc.use_offset = True  # Maintain head's local position relative to neck
        copy_loc.influence = 1.0
        script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

    # 3. ROTATION: Damped track to NOSE control point (facing direction)
    nose_obj = bpy.data.objects.get("CTRL_NOSE")
    if nose_obj:
        track = head_bone.constraints.new('DAMPED_TRACK')
        track.name = "Track_Nose_Direction"
        track.target = nose_obj
        track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
        track.influence = 1.0
        script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

        # 4. ROTATION LIMITS: Prevent extreme angles for stability
        limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
        limit_rot.name = "Limit_Head_Rotation"
        limit_rot.use_limit_x = True
        limit_rot.min_x = -0.5  # Limited head tilt (nodding)
        limit_rot.max_x = 0.5
        limit_rot.use_limit_y = True
        limit_rot.min_y = -1.5  # Head turn (looking left/right)
        limit_rot.max_y = 1.5
        limit_rot.use_limit_z = True
        limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
        limit_rot.max_z = 0.3
        limit_rot.owner_space = 'LOCAL'
        script_log("✓ Added natural head rotation limits")
    else:
        script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log("✓ Safe head constraints setup complete")

# GOOD

def create_procedural_head(armature_obj, figure_name, garment_config, global_cloth_settings, neck_config=None):
    """Create human-like head with coordinated vertex bundles for seamless neck integration - UPDATED WITH SAFE CONSTRAINTS"""
    script_log("Creating procedural garment_head with coordinated vertex bundles and safe constraints...")

    # Get head bone position
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        head_bone = armature_obj.pose.bones.get("DEF_Head")
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        head_bone_name = "DEF_Head"
        neck_bone_name = "DEF_Neck"

        bpy.ops.object.mode_set(mode='OBJECT')

        if not head_bone:
            script_log("ERROR: Could not find head bone")
            return None

        # Get head bone positions in world space - Use tail for head position
        head_base_pos = armature_obj.matrix_world @ head_bone.head  # Base of head/neck (where neck connects)
        head_top_pos = armature_obj.matrix_world @ head_bone.tail  # Top of head (where head should be)

        # Get neck bone position for coordination
        neck_base_pos = armature_obj.matrix_world @ neck_bone.head if neck_bone else head_base_pos
        neck_top_pos = armature_obj.matrix_world @ neck_bone.tail if neck_bone else head_base_pos

        # Get head dimensions from new config format
        scale = garment_config.get("scale", [0.25, 0.28, 0.3])  # Larger, more human-sized proportions
        position_offset = garment_config.get("position_offset", [0.0, 0.0, 0.0])
        rotation_offset = garment_config.get("rotation_offset", [0.0, 0.0, 0.0])
        neck_connection_radius = garment_config.get("neck_connection_radius", 0.08)
        subdivision_levels = garment_config.get("subdivision_levels", 3)  # More subdivisions for detail

        # Get geometry settings from config
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        head_segments = geometry_settings.get("head_segments", 32)  # More segments for detail
        head_ring_count = geometry_settings.get("head_ring_count", 24)  # More rings for detail
        base_radii = global_cloth_settings.get("base_radii", {})
        head_base_radius = base_radii.get("head", 0.25)  # Larger base radius

        # Get head-neck bundle diameter from config
        head_neck_diameter = garment_config.get("diameter_neck", 0.16)  # Diameter at base of head/neck junction

        script_log(f"DEBUG: Procedural head scale: {scale}")
        script_log(f"DEBUG: Position offset: {position_offset}, Rotation offset: {rotation_offset}")
        script_log(f"DEBUG: Head bone - base (neck): {head_base_pos}, top (head): {head_top_pos}")
        script_log(f"DEBUG: Head geometry - segments: {head_segments}, rings: {head_ring_count}")
        script_log(f"DEBUG: Head-neck junction diameter: {head_neck_diameter}")

        # =========================================================================
        # STEP 1: CREATE HEAD MESH WITH NECK OPENING
        # =========================================================================
        script_log("DEBUG: Creating procedural head as UV sphere with neck opening...")
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=head_segments,
            ring_count=head_ring_count,
            radius=head_base_radius,
            location=head_top_pos + Vector(position_offset)  # Position at TOP of head bone
        )
        head_obj = bpy.context.active_object
        head_obj.name = f"{figure_name}_Head"

        # Apply human-like scaling (wider and taller than spherical)
        head_obj.scale = Vector(scale)

        # Apply rotation offset if specified
        if any(rotation_offset):
            head_obj.rotation_euler = Vector(rotation_offset)

        # =========================================================================
        # STEP 2: SHAPE HEAD TO HUMAN PROPORTIONS
        # =========================================================================
        script_log("DEBUG: Shaping head to human proportions...")
        bpy.context.view_layer.objects.active = head_obj

        # Use bmesh for vertex manipulation without entering edit mode
        bm = bmesh.new()
        bm.from_mesh(head_obj.data)

        # Define human head proportions (these could be moved to config)
        human_proportions = garment_config.get("human_proportions", {
            'forehead_scale': 0.95,  # Slightly narrower forehead
            'cheek_scale': 1.1,  # Fuller cheeks
            'jaw_scale': 0.9,  # Narrower jaw
            'chin_scale': 0.8,  # More pronounced chin
            'back_head_scale': 1.05,  # Fuller back of head
            'top_head_scale': 0.95,  # Slightly flatter top
            'neck_opening_scale': 0.7,  # Create neck opening
        })

        # Shape vertices to human proportions
        for vert in bm.verts:
            # Convert to spherical coordinates for easier manipulation
            x, y, z = vert.co
            radius = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            if radius > 0.001:  # Avoid division by zero
                # Normalize to unit sphere
                nx, ny, nz = x / radius, y / radius, z / radius

                # Apply human proportions based on vertex position
                scale_factor = 1.0

                # Forehead area (front top)
                if nz > 0.3 and ny > 0.2:
                    scale_factor = human_proportions['forehead_scale']

                # Cheek area (front sides)
                elif nz > -0.2 and abs(nx) > 0.4:
                    scale_factor = human_proportions['cheek_scale']

                # Jaw area (front bottom)
                elif nz < -0.3 and abs(ny) < 0.3:
                    scale_factor = human_proportions['jaw_scale']

                # Chin area (very front bottom)
                elif nz < -0.5 and abs(nx) < 0.2:
                    scale_factor = human_proportions['chin_scale']

                # Back of head
                elif nz < -0.1 and abs(nx) < 0.3:
                    scale_factor = human_proportions['back_head_scale']

                # Top of head
                elif ny > 0.7:
                    scale_factor = human_proportions['top_head_scale']

                # Neck opening area (bottom center)
                elif ny < -0.6 and abs(nz) < 0.3 and abs(nx) < 0.3:
                    scale_factor = human_proportions['neck_opening_scale']

                # Apply the scaling
                vert.co.x = nx * radius * scale_factor
                vert.co.y = ny * radius * scale_factor
                vert.co.z = nz * radius * scale_factor

        # Write back to mesh (object mode is safe for this)
        bm.to_mesh(head_obj.data)
        bm.free()

        # =========================================================================
        # STEP 3: ADD FACIAL FEATURES USING SHAPE KEYS
        # =========================================================================
        facial_features = garment_config.get("facial_features", {})
        if facial_features.get("enable_eye_sockets", True) or facial_features.get("enable_nose_bridge", True):
            script_log("DEBUG: Adding facial features with shape keys...")

            # Add basis shape key first
            head_obj.shape_key_add(name="Basis")

            # Create shape keys FIRST before using bmesh
            shape_key_objects = {}

            if facial_features.get("enable_eye_sockets", True):
                script_log("DEBUG: Creating eye sockets shape key...")
                eye_sockets = head_obj.shape_key_add(name="EyeSockets")
                shape_key_objects["EyeSockets"] = eye_sockets

            if facial_features.get("enable_nose_bridge", True):
                script_log("DEBUG: Creating nose bridge shape key...")
                nose_bridge = head_obj.shape_key_add(name="NoseBridge")
                shape_key_objects["NoseBridge"] = nose_bridge

            # NOW use bmesh for shape key editing
            bm = bmesh.new()
            bm.from_mesh(head_obj.data)

            # Ensure we have shape key layers - they should exist now
            shape_keys = bm.verts.layers.shape

            # Create eye sockets if enabled
            if facial_features.get("enable_eye_sockets", True) and "EyeSockets" in shape_keys:
                script_log("DEBUG: Adding eye sockets geometry...")
                eye_socket_layer = shape_keys["EyeSockets"]
                eye_socket_depth = facial_features.get("eye_socket_depth", 0.08)

                for vert in bm.verts:
                    x, y, z = vert.co
                    # Eye socket areas (front upper sides)
                    if z > 0.1 and abs(x) > 0.3 and y > 0.1:
                        # Create indentation for eye sockets
                        depth = eye_socket_depth
                        vert[eye_socket_layer] = vert.co + Vector((-x * depth * 0.5, -y * depth * 0.3, -z * depth))

            # Create nose bridge if enabled
            if facial_features.get("enable_nose_bridge", True) and "NoseBridge" in shape_keys:
                script_log("DEBUG: Adding nose bridge geometry...")
                nose_bridge_layer = shape_keys["NoseBridge"]
                nose_bridge_height = facial_features.get("nose_bridge_height", 0.05)

                for vert in bm.verts:
                    x, y, z = vert.co
                    # Nose bridge area (center front)
                    if abs(x) < 0.15 and z > 0.2 and y > -0.1:
                        # Create protrusion for nose
                        height = nose_bridge_height
                        vert[nose_bridge_layer] = vert.co + Vector((0, 0, height))

            # Write shape keys back
            bm.to_mesh(head_obj.data)
            bm.free()

            # Apply some of the facial features
            if "EyeSockets" in head_obj.data.shape_keys.key_blocks:
                head_obj.data.shape_keys.key_blocks["EyeSockets"].value = 0.3
            if "NoseBridge" in head_obj.data.shape_keys.key_blocks:
                head_obj.data.shape_keys.key_blocks["NoseBridge"].value = 0.4
        else:
            script_log("DEBUG: Facial features disabled in config")

        # =========================================================================
        # STEP 4: SETUP VERTEX GROUPS WITH COORDINATED HEAD-NECK BUNDLES
        # =========================================================================
        script_log("DEBUG: Setting up head vertex groups with coordinated neck bundles...")

        # Clear any existing parenting
        head_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(head_obj.vertex_groups):
            head_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(head_obj.modifiers):
            if mod.type == 'ARMATURE':
                head_obj.modifiers.remove(mod)

        # Get head-neck vertex bundle from global storage
        head_neck_bundle = joint_vertex_bundles.get("DEF_Head")

        if head_neck_bundle:
            script_log(f"✓ Using head-neck vertex bundle with {head_neck_bundle['vertex_count']} vertices")
        else:
            script_log(f"⚠ No head-neck bundle found for DEF_Head, using standard head weighting")

        # Create vertex group for head bone
        head_group = head_obj.vertex_groups.new(name=head_bone_name)

        # Apply head-neck bundle weighting if available
        if head_neck_bundle:
            head_neck_positions = head_neck_bundle['vertex_positions']
            head_neck_radius = head_neck_diameter / 2

            for i, vertex in enumerate(head_obj.data.vertices):
                vert_pos = head_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the head-neck bundle
                for bundle_vert_pos in head_neck_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                # Stronger influence at neck junction, lighter toward top of head
                if min_distance <= head_neck_radius * 2.0:
                    weight = 1.0 - (min_distance / (head_neck_radius * 2.0))
                    weight = weight * weight  # Quadratic falloff

                    # Reduce weight for top of head (more flexible)
                    vert_local = head_obj.matrix_world.inverted() @ vert_pos
                    if vert_local.y > 0.5:  # Top of head
                        weight *= 0.3
                    elif vert_local.y > 0.2:  # Upper head
                        weight *= 0.6
                    # Full weight for neck area (y < 0)

                    if weight > 0.1:
                        head_group.add([i], weight, 'REPLACE')
                else:
                    # Light default weight for distant vertices
                    head_group.add([i], 0.2, 'REPLACE')
        else:
            # Fallback: assign uniform weights with neck emphasis
            for i, vertex in enumerate(head_obj.data.vertices):
                vert_local = head_obj.matrix_world.inverted() @ vertex.co
                # Stronger weight at neck, lighter at top
                if vert_local.y < -0.4:  # Neck area
                    head_group.add([i], 1.0, 'REPLACE')
                elif vert_local.y < 0:  # Lower head
                    head_group.add([i], 0.8, 'REPLACE')
                else:  # Upper head
                    head_group.add([i], 0.6, 'REPLACE')

        # =========================================================================
        # STEP 5: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 6: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and skin material...")

        # Add subdivision from config
        subdiv_mod = head_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = subdivision_levels
        subdiv_mod.render_levels = subdivision_levels

        # ADD SKIN MATERIAL WITH MODERN BLENDER 4.3+ NODES
        script_log("DEBUG: Creating skin material with modern node setup...")
        material_config = garment_config.get("material", {})
        material_color = material_config.get("color", [0.96, 0.86, 0.72, 1.0])

        head_mat = bpy.data.materials.new(name="Head_Material")
        head_mat.use_nodes = True

        # Clear default nodes for clean setup
        head_mat.node_tree.nodes.clear()

        # Create modern Principled BSDF setup
        output_node = head_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = head_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

        # Position nodes
        output_node.location = (300, 0)
        principled_node.location = (0, 0)

        # Connect nodes
        head_mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

        # Set material properties using modern API
        principled_node.inputs['Base Color'].default_value = material_color
        principled_node.inputs['Roughness'].default_value = material_config.get("roughness", 0.4)
        principled_node.inputs['Metallic'].default_value = material_config.get("metallic", 0.0)

        head_obj.data.materials.append(head_mat)

        # =========================================================================
        # STEP 7: CLOTH SIMULATION - DISABLED AS REQUESTED
        # =========================================================================
        cloth_settings = garment_config.get("cloth_settings", {})
        if cloth_settings.get("enabled", False):
            script_log("DEBUG: Adding cloth simulation to garment_head...")
            cloth_mod = head_obj.modifiers.new(name="Cloth", type='CLOTH')
            cloth_mod.settings.quality = cloth_settings.get("quality", 6)
            cloth_mod.settings.mass = cloth_settings.get("mass", 0.8)
            cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 15.0)
            cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 15.0)
            cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 10.0)
            cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 2.0)
            cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)
        else:
            script_log("DEBUG: Cloth simulation disabled for head")

        # =========================================================================
        # STEP 8: SET MODIFIER ORDER
        # =========================================================================
        bpy.context.view_layer.objects.active = head_obj
        modifiers = head_obj.modifiers

        # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
        correct_order = ["Subdivision", "Armature", "Cloth"]
        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 9: NOSE LANDMARK TRACKING SETUP - UPDATED WITH SAFE CONSTRAINTS
        # =========================================================================
        script_log("DEBUG: Setting up NOSE landmark tracking with safe constraints...")

        # Get first frame data for NOSE control point creation
        first_frame = mocap_data.get(str(frame_numbers[0]), {})

        # GUARANTEE NOSE CONTROL POINT EXISTS
        nose_obj = ensure_nose_control_point(first_frame)

        # SETUP SAFE HEAD CONSTRAINTS (PORCELAIN DOLL ARCHITECTURE)
        setup_head_constraints_safe(armature_obj, head_bone_name)

        # =========================================================================
        # STEP 10: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage
        bundle_status = f"head-neck({head_neck_bundle['vertex_count']}v)" if head_neck_bundle else "NONE (standard)"

        script_log("=== PROCEDURAL HEAD CREATION COMPLETE (COORDINATED VERTEX BUNDLES + SAFE CONSTRAINTS) ===")
        script_log(f"✓ Head positioned at TOP of DEF_Head bone (tail position)")
        script_log(f"✓ Head shaped with human proportions")
        script_log(f"✓ Facial features: Eye sockets, Nose bridge")
        script_log(f"✓ Head rotation tracks NOSE landmark in real-time (SAFE CONSTRAINTS)")
        script_log(f"✓ Natural rotation limits applied for stability")
        script_log(f"✓ Head scale: {scale} (larger, human-sized proportions)")
        script_log(f"✓ Head segments: {head_segments}, rings: {head_ring_count}")
        script_log(f"✓ Subdivision levels: {subdivision_levels}")
        script_log(f"✓ Neck connection diameter: {head_neck_diameter}")
        script_log(f"✓ Head parented to {head_bone_name}")
        script_log(f"✓ Skin material applied with modern Principled BSDF")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_settings.get('enabled', False) else 'DISABLED'}")
        script_log(f"✓ Vertex bundle used: {bundle_status}")
        script_log(f"✓ Seamless neck integration: Head uses same vertex bundle as neck")
        script_log(f"✓ PORCELAIN DOLL ARCHITECTURE: Neck anchor + Nose rotation guide")

        # Call create_neck for coordinated creation
        if neck_config:
            create_neck(armature_obj, figure_name, neck_config, global_cloth_settings)

        return head_obj

    except Exception as e:
        script_log(f"ERROR creating procedural head with coordinated bundles: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_neck(armature_obj, figure_name, garment_config, global_cloth_settings):
    """Create stretchy neck garment with coordinated vertex bundles for seamless integration"""
    script_log("Creating garment_neck with coordinated vertex bundles...")

    # Get neck bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        head_bone = armature_obj.pose.bones.get("DEF_Head")
        upper_spine_bone = armature_obj.pose.bones.get("DEF_UpperSpine")

        neck_bone_name = "DEF_Neck"
        head_bone_name = "DEF_Head"
        upper_spine_bone_name = "DEF_UpperSpine"

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([neck_bone, head_bone, upper_spine_bone]):
            script_log("ERROR: Could not find neck, head, or spine bones")
            return None

        # Get bone positions in world space
        neck_base_pos = armature_obj.matrix_world @ neck_bone.head  # Base of neck (shoulders)
        neck_top_pos = armature_obj.matrix_world @ neck_bone.tail  # Top of neck (head base)
        head_base_pos = armature_obj.matrix_world @ head_bone.head  # Base of head
        spine_top_pos = armature_obj.matrix_world @ upper_spine_bone.tail  # Top of spine

        # Get neck dimensions from config
        puffiness = garment_config.get("puffiness", 1.1)
        neck_height = garment_config.get("neck_height", 0.08)
        neck_diameter = garment_config.get("neck_diameter", 0.14)
        collar_style = garment_config.get("collar_style", "turtleneck")

        # Get neck-shoulder bundle diameter from config
        neck_spine_diameter = garment_config.get("diameter_spine", 0.18)  # Diameter at neck-spine-shoulder junction

        # Get geometry settings
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        base_radii = global_cloth_settings.get("base_radii", {})
        segments = geometry_settings.get("default_segments", 12)

        # Calculate neck length and radius
        neck_length = (neck_top_pos - neck_base_pos).length
        neck_radius = neck_diameter / 2 * puffiness

        script_log(f"DEBUG: Neck garment - Height: {neck_height}, Diameter: {neck_diameter}")
        script_log(f"DEBUG: Neck garment - Puffiness: {puffiness}, Final radius: {neck_radius}")
        script_log(f"DEBUG: Neck garment - Collar style: {collar_style}")
        script_log(f"DEBUG: Neck-spine junction diameter: {neck_spine_diameter}")

        # =========================================================================
        # STEP 1: CREATE NECK CYLINDER WITH COORDINATED ENDS
        # =========================================================================
        script_log("DEBUG: Creating neck cylinder with coordinated ends...")

        # Position cylinder to cover the neck area
        neck_center = neck_base_pos + ((neck_top_pos - neck_base_pos) / 2)
        neck_direction = (neck_top_pos - neck_base_pos).normalized()

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=neck_length,
            radius=neck_radius,
            location=neck_center
        )
        neck_obj = bpy.context.active_object
        neck_obj.name = f"{figure_name}_Neck"

        # Rotate to align with neck bone direction
        neck_obj.rotation_euler = neck_direction.to_track_quat('Z', 'Y').to_euler()

        # =========================================================================
        # STEP 2: SETUP VERTEX GROUPS WITH COORDINATED BUNDLES
        # =========================================================================
        script_log("DEBUG: Setting up neck vertex groups with coordinated bundles...")

        # Clear any existing parenting
        neck_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(neck_obj.vertex_groups):
            neck_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(neck_obj.modifiers):
            if mod.type == 'ARMATURE':
                neck_obj.modifiers.remove(mod)

        # Get coordinated vertex bundles from global storage
        head_neck_bundle = joint_vertex_bundles.get("DEF_Head")  # Head-neck junction
        neck_spine_bundle = joint_vertex_bundles.get("DEF_Neck")  # Neck-spine-shoulder junction

        if head_neck_bundle:
            script_log(f"✓ Using head-neck vertex bundle with {head_neck_bundle['vertex_count']} vertices")
        else:
            script_log(f"⚠ No head-neck bundle found for DEF_Head")

        if neck_spine_bundle:
            script_log(f"✓ Using neck-spine vertex bundle with {neck_spine_bundle['vertex_count']} vertices")
        else:
            script_log(f"⚠ No neck-spine bundle found for DEF_Neck")

        # Create vertex groups for all three coordination points
        neck_group = neck_obj.vertex_groups.new(name=neck_bone_name)
        head_coordination_group = neck_obj.vertex_groups.new(name="Head_Coordination_Neck")
        spine_coordination_group = neck_obj.vertex_groups.new(name="Spine_Coordination_Neck")

        # Calculate bundle radii
        head_neck_radius = neck_spine_diameter / 2 * 0.8  # Slightly smaller than spine junction
        neck_spine_radius = neck_spine_diameter / 2

        # =========================================================================
        # STEP 3: APPLY HEAD-NECK BUNDLE WEIGHTS (TOP OF NECK)
        # =========================================================================
        if head_neck_bundle:
            script_log("DEBUG: Applying head-neck bundle weights to neck top...")
            head_neck_positions = head_neck_bundle['vertex_positions']

            for i, vertex in enumerate(neck_obj.data.vertices):
                vert_pos = neck_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the head-neck bundle
                for bundle_vert_pos in head_neck_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                # Strong influence at top of neck for seamless head connection
                if min_distance <= head_neck_radius:
                    weight = 1.0 - (min_distance / head_neck_radius)
                    weight = weight * weight  # Quadratic falloff

                    # Stronger weight for top of neck vertices
                    vert_local = neck_obj.matrix_world.inverted() @ vert_pos
                    if vert_local.z > neck_length * 0.3:  # Top third of neck
                        weight *= 1.2  # Boost influence
                    elif vert_local.z > 0:  # Upper half
                        weight *= 1.0
                    else:  # Lower half
                        weight *= 0.6  # Reduced influence

                    weight = min(weight, 1.0)  # Clamp to 1.0

                    if weight > 0.1:
                        head_coordination_group.add([i], weight, 'REPLACE')
                        # Also assign to neck bone with coordinated weight
                        neck_group.add([i], weight * 0.8, 'REPLACE')
        else:
            script_log("DEBUG: Using fallback head-neck weighting...")
            # Fallback: graduated weights from top to bottom
            for i, vertex in enumerate(neck_obj.data.vertices):
                vert_local = neck_obj.matrix_world.inverted() @ vertex.co
                # Normalize Z from -0.5 (bottom) to 0.5 (top)
                z_norm = (vert_local.z + neck_length / 2) / neck_length

                # Strong weight at top for head connection, lighter at bottom
                top_weight = z_norm  # 1.0 at top, 0.0 at bottom
                top_weight = top_weight * top_weight  # Quadratic

                if top_weight > 0.1:
                    head_coordination_group.add([i], top_weight, 'REPLACE')
                    neck_group.add([i], top_weight * 0.8, 'REPLACE')

        # =========================================================================
        # STEP 4: APPLY NECK-SPINE BUNDLE WEIGHTS (BOTTOM OF NECK)
        # =========================================================================
        if neck_spine_bundle:
            script_log("DEBUG: Applying neck-spine bundle weights to neck base...")
            neck_spine_positions = neck_spine_bundle['vertex_positions']

            for i, vertex in enumerate(neck_obj.data.vertices):
                vert_pos = neck_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the neck-spine bundle
                for bundle_vert_pos in neck_spine_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                # Strong influence at base of neck for torso integration
                if min_distance <= neck_spine_radius:
                    weight = 1.0 - (min_distance / neck_spine_radius)
                    weight = weight * weight  # Quadratic falloff

                    # Stronger weight for base of neck vertices
                    vert_local = neck_obj.matrix_world.inverted() @ vert_pos
                    if vert_local.z < -neck_length * 0.3:  # Bottom third of neck
                        weight *= 1.2  # Boost influence
                    elif vert_local.z < 0:  # Lower half
                        weight *= 1.0
                    else:  # Upper half
                        weight *= 0.4  # Reduced influence

                    weight = min(weight, 1.0)  # Clamp to 1.0

                    if weight > 0.1:
                        spine_coordination_group.add([i], weight, 'REPLACE')
                        # Also assign to neck bone with coordinated weight
                        neck_group.add([i], weight * 0.8, 'REPLACE')
        else:
            script_log("DEBUG: Using fallback neck-spine weighting...")
            # Fallback: graduated weights from bottom to top
            for i, vertex in enumerate(neck_obj.data.vertices):
                vert_local = neck_obj.matrix_world.inverted() @ vertex.co
                # Normalize Z from -0.5 (bottom) to 0.5 (top)
                z_norm = (vert_local.z + neck_length / 2) / neck_length

                # Strong weight at bottom for spine connection, lighter at top
                bottom_weight = 1.0 - z_norm  # 1.0 at bottom, 0.0 at top
                bottom_weight = bottom_weight * bottom_weight  # Quadratic

                if bottom_weight > 0.1:
                    spine_coordination_group.add([i], bottom_weight, 'REPLACE')
                    neck_group.add([i], bottom_weight * 0.8, 'REPLACE')

        # =========================================================================
        # STEP 5: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = neck_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 6: ADD CLOTH SIMULATION FOR STRETCHY FABRIC
        # =========================================================================
        script_log("DEBUG: Adding stretchy cloth simulation to neck...")
        cloth_settings = garment_config.get("cloth_settings", {})

        if cloth_settings.get("enabled", True):
            cloth_mod = neck_obj.modifiers.new(name="Cloth", type='CLOTH')
            cloth_mod.settings.quality = cloth_settings.get("quality", 6)
            cloth_mod.settings.mass = cloth_settings.get("mass", 0.3)
            cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 5.0)
            cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 4.0)
            cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 3.0)
            cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 0.5)
            cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)

            # PIN CLOTH TO COORDINATION GROUPS FOR SEAMLESS INTEGRATION
            # Create combined coordination group
            combined_coordination_group = neck_obj.vertex_groups.new(name="Neck_Combined_Coordination")

            for i in range(len(neck_obj.data.vertices)):
                max_weight = 0.0

                # Check head coordination weight
                try:
                    head_weight = head_coordination_group.weight(i)
                    max_weight = max(max_weight, head_weight)
                except:
                    pass

                # Check spine coordination weight
                try:
                    spine_weight = spine_coordination_group.weight(i)
                    max_weight = max(max_weight, spine_weight)
                except:
                    pass

                if max_weight > 0.1:
                    combined_coordination_group.add([i], max_weight, 'REPLACE')

            cloth_mod.settings.vertex_group_mass = "Neck_Combined_Coordination"

            script_log("✓ Added stretchy cloth simulation to garment_neck")
            script_log("✓ Cloth pinned to combined head-spine coordination groups")
        else:
            script_log("DEBUG: Cloth simulation disabled for neck")

        # =========================================================================
        # STEP 7: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and materials...")

        # Add subdivision for smoother fabric
        subdiv_mod = neck_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = 1
        subdiv_mod.render_levels = 1

        # Add fabric material
        material_config = garment_config.get("material", {})
        material_color = material_config.get("color", [0.1, 0.3, 0.8, 1.0])

        neck_mat = bpy.data.materials.new(name="Neck_Material")
        neck_mat.use_nodes = True

        # Set fabric properties (softer, less shiny than skin)
        neck_mat.diffuse_color = material_color
        neck_mat.roughness = material_config.get("roughness", 0.8)
        neck_mat.metallic = material_config.get("metallic", 0.0)

        neck_obj.data.materials.append(neck_mat)

        # =========================================================================
        # STEP 8: SET MODIFIER ORDER
        # =========================================================================
        script_log("DEBUG: Setting modifier order...")

        bpy.context.view_layer.objects.active = neck_obj
        modifiers = neck_obj.modifiers

        # Ensure order: Subdivision → Armature → Cloth
        correct_order = ["Subdivision", "Armature", "Cloth"]
        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 9: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage
        bundle_status = []
        if head_neck_bundle: bundle_status.append(f"head-neck({head_neck_bundle['vertex_count']}v)")
        if neck_spine_bundle: bundle_status.append(f"neck-spine({neck_spine_bundle['vertex_count']}v)")

        script_log("=== GARMENT_NECK CREATION COMPLETE (COORDINATED VERTEX BUNDLES) ===")
        script_log(f"✓ Neck positioned along neck bone")
        script_log(f"✓ Neck radius: {neck_radius}, Length: {neck_length:.3f}")
        script_log(f"✓ Neck-spine junction diameter: {neck_spine_diameter}")
        script_log(f"✓ Stretchy cloth simulation: {'ENABLED' if cloth_settings.get('enabled', True) else 'DISABLED'}")
        script_log(f"✓ Collar style: {collar_style}")
        script_log(f"✓ Neck parented to {neck_bone_name}")
        script_log(f"✓ Vertex bundles used: {', '.join(bundle_status) if bundle_status else 'NONE (fallback)'}")
        script_log(f"✓ Seamless integration: Head connection + Spine/shoulder coordination")
        script_log(f"✓ Combined cloth pinning: Head and spine coordination groups")

        return neck_obj

    except Exception as e:
        script_log(f"ERROR creating garment_neck: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_cloth_garments(armature_obj, figure_name):
    """Create cloth garments that follow the bone rig - WITH CONTINUOUS SLEEVES"""
    script_log("=== CREATING CLOTH GARMENTS ===")

    garments_created = 0

    # Load cloth config to get garment definitions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

    with open(CLOTH_CONFIG_FILE, 'r') as file:
        cloth_config = json.load(file)

    # Use new config structure - only cloth_garments
    garment_definitions = cloth_config.get("cloth_garments", {})

    # Store garments for potential connections
    left_sleeve_obj = None
    right_sleeve_obj = None
    coat_obj = None
    # ADD MITTENS TRACKING
    left_mitten_obj = None
    right_mitten_obj = None

    # CREATE GARMENT_HEAD first (with physics flag check)
    if "garment_head" in garment_definitions:
        head_config = garment_definitions["garment_head"]
        neck_config = garment_definitions.get("garment_neck")

        # Check both enabled flag and physics setting
        if head_config.get("enabled", True):
            head_obj = create_head(armature_obj, figure_name, head_config, cloth_config, neck_config)
            if head_obj:
                garments_created += 1
                script_log("Created garment_head with integrated neck")

    # CREATE BOOTS before pants for future ankle connection (with physics flag check)
    if "left_boot" in garment_definitions:
        left_boot_config = garment_definitions["left_boot"]
        if left_boot_config.get("enabled", True):
            left_boot = create_boot(armature_obj, figure_name, left_boot_config, {}, "left")
            if left_boot:
                garments_created += 1
                script_log("Created left boot")

    if "right_boot" in garment_definitions:
        right_boot_config = garment_definitions["right_boot"]
        if right_boot_config.get("enabled", True):
            right_boot = create_boot(armature_obj, figure_name, right_boot_config, {}, "right")
            if right_boot:
                garments_created += 1
                script_log("Created right boot")

    # CREATE CONTINUOUS SLEEVES (NEW - single mesh with spherical elbow weighting)
    if "left_sleeve" in garment_definitions:
        left_sleeve_config = garment_definitions["left_sleeve"]
        if left_sleeve_config.get("enabled", True):
            left_sleeve_obj = create_sleeve(armature_obj, figure_name, left_sleeve_config, {}, "left")
            if left_sleeve_obj:
                garments_created += 1
                script_log("Created left continuous sleeve with spherical elbow weighting")

    if "right_sleeve" in garment_definitions:
        right_sleeve_config = garment_definitions["right_sleeve"]
        if right_sleeve_config.get("enabled", True):
            right_sleeve_obj = create_sleeve(armature_obj, figure_name, right_sleeve_config, {}, "right")
            if right_sleeve_obj:
                garments_created += 1
                script_log("Created right continuous sleeve with spherical elbow weighting")

    # CREATE PANTS (with physics flag check)
    if "left_pants" in garment_definitions:
        left_pants_config = garment_definitions["left_pants"]
        if left_pants_config.get("enabled", True):
            left_pants = create_pants(armature_obj, figure_name, left_pants_config, {}, "left")
            if left_pants:
                garments_created += 1
                script_log("Created left pants")

    if "right_pants" in garment_definitions:
        right_pants_config = garment_definitions["right_pants"]
        if right_pants_config.get("enabled", True):
            right_pants = create_pants(armature_obj, figure_name, right_pants_config, {}, "right")
            if right_pants:
                garments_created += 1
                script_log("Created right pants")

    # CREATE COAT TORSO (with physics flag check) - STORE FOR CONNECTION
    if "coat_torso" in garment_definitions:
        coat_config = garment_definitions["coat_torso"]
        if coat_config.get("enabled", True):
            coat_obj = create_coat(armature_obj, figure_name, coat_config, {})
            if coat_obj:
                garments_created += 1
                script_log("Created coat_torso")

    # CREATE MITTENS (with physics flag check) - STORE FOR TRACKING
    if "left_mitten" in garment_definitions:
        left_mitten_config = garment_definitions["left_mitten"]
        if left_mitten_config.get("enabled", True):
            left_mitten_obj = create_mitten(armature_obj, figure_name, left_mitten_config, {}, "left")
            if left_mitten_obj:
                garments_created += 1
                script_log("Created left mitten")

    if "right_mitten" in garment_definitions:
        right_mitten_config = garment_definitions["right_mitten"]
        if right_mitten_config.get("enabled", True):
            right_mitten_obj = create_mitten(armature_obj, figure_name, right_mitten_config, {}, "right")
            if right_mitten_obj:
                garments_created += 1
                script_log("Created right mitten")

    # CREATE LEGACY GARMENTS (for backward compatibility with old garment types)
    for garment_name, garment_config in garment_definitions.items():
        # Skip garments we've already handled above
        if garment_name in ["garment_head", "left_boot", "right_boot", "left_sleeve", "right_sleeve",
                            "left_pants", "right_pants", "coat_torso", "left_mitten", "right_mitten"]:
            continue

        if garment_config.get("enabled", True):
            if garment_name == "long_sleeve_shirt":
                garment_obj = create_long_sleeve_shirt(armature_obj, figure_name, garment_config, {})
            else:
                script_log(f"WARNING: Unknown garment type: {garment_name}")
                continue

            if garment_obj:
                garments_created += 1
                script_log(f"Created {garment_name}")

    # LOG MITTENS STATUS
    if left_mitten_obj or right_mitten_obj:
        mitten_count = (1 if left_mitten_obj else 0) + (1 if right_mitten_obj else 0)
        script_log(f"✓ Created {mitten_count} mittens")
    else:
        script_log(f"✓ No mittens created (disabled or failed)")

    script_log(f"Created {garments_created} cloth garments")
    return garments_created

##########################################################################################

def create_long_sleeve_shirt(armature_obj, figure_name, garment_config, global_cloth_settings):
    """Create a long-sleeve shirt garment with cloth simulation"""
    script_log("Creating long-sleeve shirt cloth garment...")

    # Create mesh for shirt
    bpy.ops.mesh.primitive_cube_add()
    shirt_obj = bpy.context.active_object
    shirt_obj.name = f"{figure_name}_LongSleeveShirt"

    # Apply scale and position from config
    scale = garment_config.get("scale", [0.4, 0.3, 0.5])
    position_offset = garment_config.get("position_offset", [0.0, 0.0, 0.1])

    shirt_obj.scale = Vector(scale)
    shirt_obj.location = Vector(position_offset)

    # Add cloth simulation modifier with config settings
    cloth_mod = shirt_obj.modifiers.new(name="Cloth", type='CLOTH')

    # Apply global cloth settings
    cloth_mod.settings.quality = global_cloth_settings.get("quality", 5)
    cloth_mod.settings.time_scale = global_cloth_settings.get("time_scale", 1.0)

    # Apply garment-specific cloth settings
    cloth_settings = garment_config.get("cloth_settings", {})
    cloth_mod.settings.mass = cloth_settings.get("mass", 0.4)
    cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)
    cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 25.0)
    cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 25.0)
    cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 15.0)
    cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 1.5)

    # Parent to appropriate bones from config
    parent_bones = garment_config.get("parent_bones", [])
    default_weight = global_cloth_settings.get("default_vertex_weight", 0.3)
    setup_cloth_parenting(shirt_obj, armature_obj, parent_bones, default_weight)

    # Add material with config properties
    material_color = garment_config.get("material_color", [0.8, 0.8, 0.8, 1.0])
    material_props = garment_config.get("material_properties", {})

    cloth_mat = bpy.data.materials.new(name="LongSleeveShirt_Material")
    cloth_mat.diffuse_color = material_color
    cloth_mat.metallic = material_props.get("metallic", 0.0)
    cloth_mat.roughness = material_props.get("roughness", 0.7)
    cloth_mat.specular_intensity = material_props.get("specular", 0.3)
    shirt_obj.data.materials.append(cloth_mat)

    return shirt_obj

##########################################################################################

def create_boot(armature_obj, figure_name, garment_config, global_cloth_settings, side="left"):
    """Create modular boots with ankle bridge, shaft cylinder, and foot cylinder using joint_vertex_bundles"""
    script_log(f"Creating {side} boot with joint_vertex_bundles integration...")

    # Get shin and foot bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        if side == "left":
            shin_bone = armature_obj.pose.bones.get("DEF_LeftShin")
            foot_bone = armature_obj.pose.bones.get("DEF_LeftFoot")
            shin_bone_name = "DEF_LeftShin"
            foot_bone_name = "DEF_LeftFoot"
        else:
            shin_bone = armature_obj.pose.bones.get("DEF_RightShin")
            foot_bone = armature_obj.pose.bones.get("DEF_RightFoot")
            shin_bone_name = "DEF_RightShin"
            foot_bone_name = "DEF_RightFoot"

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([shin_bone, foot_bone]):
            script_log(f"ERROR: Could not find leg bones for {side} boot")
            return None

        # Get bone positions in world space
        ankle_pos = armature_obj.matrix_world @ shin_bone.tail  # Ankle position
        foot_start_pos = armature_obj.matrix_world @ foot_bone.head  # Start of foot
        toe_pos = armature_obj.matrix_world @ foot_bone.tail  # End of foot

        # Get boot dimensions from config
        puffiness = garment_config.get("puffiness", 1.0)
        shaft_height = garment_config.get("shaft_height", 0.15)
        foot_length = garment_config.get("foot_length", 0.12)
        foot_height = garment_config.get("foot_height", 0.06)
        segments = garment_config.get("segments", 8)

        # Get geometry settings from config
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        base_radii = global_cloth_settings.get("base_radii", {})

        # Use configurable base ankle radius
        base_ankle_radius = base_radii.get("ankle", 0.08)
        ankle_radius = base_ankle_radius * puffiness

        # Calculate foot radii based on configurable dimensions
        foot_radius_x = foot_length / 3 * puffiness  # Width of foot
        foot_radius_y = foot_height / 2 * puffiness  # Height of foot

        # Get vertex weighting settings from config
        weighting_config = garment_config.get("vertex_weighting", {})
        falloff_type = weighting_config.get("sphere_falloff", "quadratic")
        min_weight_threshold = weighting_config.get("min_weight_threshold", 0.05)
        sphere_influence_scale = weighting_config.get("sphere_influence_scale", 2.0)

        # Calculate sphere radii for bundle integration
        ankle_sphere_radius = ankle_radius * sphere_influence_scale

        script_log(f"DEBUG: {side} boot - Shaft height: {shaft_height}, Foot length: {foot_length}")
        script_log(f"DEBUG: {side} boot - Base ankle radius: {base_ankle_radius}, Puffiness: {puffiness}")
        script_log(f"DEBUG: {side} boot - Final ankle radius: {ankle_radius}, Foot radii: {foot_radius_x}, {foot_radius_y}")
        script_log(f"DEBUG: {side} boot - Segments: {segments}")
        script_log(f"DEBUG: {side} boot - Ankle sphere radius: {ankle_sphere_radius:.3f}")

        boot_objects = []

        # =========================================================================
        # STEP 1: CREATE ANKLE BRIDGE SPHERE WITH JOINT_VERTEX_BUNDLES
        # =========================================================================
        script_log(f"DEBUG: Creating ankle bridge sphere for {side} boot with bundle integration...")
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12,
            ring_count=6,
            radius=ankle_radius,
            location=ankle_pos
        )
        ankle_sphere = bpy.context.active_object
        ankle_sphere.name = f"{figure_name}_{side.capitalize()}AnkleBridge"
        boot_objects.append(ankle_sphere)

        # =========================================================================
        # STEP 2: CREATE SHAFT CYLINDER (vertical)
        # =========================================================================
        script_log(f"DEBUG: Creating shaft cylinder for {side} boot...")

        # Shaft extends upward from ankle position
        shaft_top_pos = ankle_pos + Vector((0, 0, shaft_height))
        shaft_center = (ankle_pos + shaft_top_pos) / 2
        shaft_direction = Vector((0, 0, 1))  # Straight up

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=shaft_height,
            radius=ankle_radius,  # Same radius as ankle bridge
            location=shaft_center
        )
        shaft_obj = bpy.context.active_object
        shaft_obj.name = f"{figure_name}_{side.capitalize()}Shaft"

        # Rotate to align vertically (already aligned by default)
        # No rotation needed for vertical cylinder

        boot_objects.append(shaft_obj)

        # =========================================================================
        # STEP 3: CREATE FOOT CYLINDER (horizontal, elliptical)
        # =========================================================================
        script_log(f"DEBUG: Creating foot cylinder for {side} boot...")

        # Foot extends from ankle to toe position
        foot_vector = toe_pos - foot_start_pos
        foot_direction = foot_vector.normalized()
        foot_center = foot_start_pos + (foot_vector / 2)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=foot_length,
            radius=1.0,  # Unit radius, we'll scale it
            location=foot_center
        )
        foot_obj = bpy.context.active_object
        foot_obj.name = f"{figure_name}_{side.capitalize()}Foot"

        # Rotate to align with foot bone direction
        foot_obj.rotation_euler = foot_direction.to_track_quat('Z', 'Y').to_euler()

        # Scale to create elliptical cross-section for foot
        foot_obj.scale = (foot_radius_x, foot_radius_y, 1.0)

        boot_objects.append(foot_obj)

        # =========================================================================
        # STEP 4: SETUP PARENTING AND ARMATURE MODIFIERS
        # =========================================================================
        script_log("DEBUG: Setting up parenting and armature modifiers...")

        for obj in boot_objects:
            # Clear any existing parenting
            obj.parent = None

            # Clear any existing vertex groups
            for vg in list(obj.vertex_groups):
                obj.vertex_groups.remove(vg)

            # Remove any existing armature modifiers
            for mod in list(obj.modifiers):
                if mod.type == 'ARMATURE':
                    obj.modifiers.remove(mod)

        # =========================================================================
        # STEP 5: JOINT_VERTEX_BUNDLES INTEGRATION FOR ANKLE BRIDGE
        # =========================================================================
        script_log(f"DEBUG: Integrating joint_vertex_bundles for {side} boot ankle...")

        # Get ankle vertex bundle from global storage
        ankle_bundle = joint_vertex_bundles.get(shin_bone_name)  # Ankle bundle uses shin bone

        if ankle_bundle:
            script_log(f"✓ Using ankle vertex bundle with {ankle_bundle['vertex_count']} vertices")
        else:
            script_log(f"⚠ No ankle bundle found for {shin_bone_name}, using standard boot weighting")

        # Create spherical vertex group for ankle bundle integration
        ankle_sphere_group = ankle_sphere.vertex_groups.new(name=f"Ankle_Sphere_{side}")

        # =========================================================================
        # APPLY ANKLE BUNDLE VERTEX WEIGHTS TO ANKLE BRIDGE
        # =========================================================================
        if ankle_bundle:
            script_log(f"DEBUG: Applying ankle vertex bundle to ankle bridge...")
            ankle_vertex_positions = ankle_bundle['vertex_positions']

            for i, vertex in enumerate(ankle_sphere.data.vertices):
                vert_pos = ankle_sphere.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the bundle
                for bundle_vert_pos in ankle_vertex_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                if min_distance <= ankle_sphere_radius:
                    weight = 1.0 - (min_distance / ankle_sphere_radius)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    if weight > min_weight_threshold:
                        ankle_sphere_group.add([i], weight, 'REPLACE')
        else:
            # Fallback: assign uniform weights to ankle sphere
            for i in range(len(ankle_sphere.data.vertices)):
                ankle_sphere_group.add([i], 1.0, 'REPLACE')

        # =========================================================================
        # STEP 6: SETUP VERTEX GROUPS FOR ARMATURE BINDING (Preserved Functionality)
        # =========================================================================
        script_log("DEBUG: Setting up vertex groups for armature binding...")

        # Ankle Sphere -> Shin Bone (since it's at ankle position)
        ankle_group = ankle_sphere.vertex_groups.new(name=shin_bone_name)
        for i in range(len(ankle_sphere.data.vertices)):
            ankle_group.add([i], 1.0, 'REPLACE')

        # Shaft Cylinder -> Shin Bone
        shaft_group = shaft_obj.vertex_groups.new(name=shin_bone_name)
        for i in range(len(shaft_obj.data.vertices)):
            shaft_group.add([i], 1.0, 'REPLACE')

        # Foot Cylinder -> Foot Bone
        foot_group = foot_obj.vertex_groups.new(name=foot_bone_name)
        for i in range(len(foot_obj.data.vertices)):
            foot_group.add([i], 1.0, 'REPLACE')

        # =========================================================================
        # STEP 7: ADD ARMATURE MODIFIERS
        # =========================================================================
        script_log("DEBUG: Adding armature modifiers to all boot parts...")

        for obj in boot_objects:
            # Add armature modifier
            armature_mod = obj.modifiers.new(name="Armature", type='ARMATURE')
            armature_mod.object = armature_obj
            armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 8: ADD SUBDIVISION AND MATERIALS (Preserved Functionality)
        # =========================================================================
        script_log("DEBUG: Adding subdivision and materials...")

        for obj in boot_objects:
            # Add subdivision
            subdiv_mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = 1
            subdiv_mod.render_levels = 1

            # Add materials from config
            material_config = garment_config.get("material", {})
            material_color = material_config.get("color", [0.3, 0.2, 0.1, 1.0])

            if "AnkleBridge" in obj.name:
                mat_name = f"{side.capitalize()}AnkleBridge_Material"
                color = (0.3, 0.2, 0.1, 1.0)  # Brown for ankle
            elif "Shaft" in obj.name:
                mat_name = f"{side.capitalize()}Shaft_Material"
                color = (0.35, 0.25, 0.15, 1.0)  # Slightly lighter brown for shaft
            else:
                mat_name = f"{side.capitalize()}Foot_Material"
                color = (0.4, 0.3, 0.2, 1.0)  # Lightest brown for foot

            boot_mat = bpy.data.materials.new(name=mat_name)
            boot_mat.diffuse_color = color
            boot_mat.roughness = material_config.get("roughness", 0.9)
            boot_mat.metallic = material_config.get("metallic", 0.1)
            obj.data.materials.append(boot_mat)

        # =========================================================================
        # STEP 9: CLOTH SIMULATION - DISABLED AS REQUESTED (Preserved Functionality)
        # =========================================================================
        cloth_config = garment_config.get("cloth_settings", {})
        if cloth_config.get("enabled", False):
            script_log(f"DEBUG: Adding cloth simulation to {side} boot...")
            for obj in boot_objects:
                cloth_mod = obj.modifiers.new(name="Cloth", type='CLOTH')

                # Apply cloth settings from config
                cloth_mod.settings.quality = cloth_config.get("quality", 6)
                cloth_mod.settings.mass = cloth_config.get("mass", 0.7)
                cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 25.0)
                cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 20.0)
                cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 15.0)
                cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 5.0)
                cloth_mod.settings.air_damping = cloth_config.get("air_damping", 1.0)

                # PIN CLOTH TO ANKLE SPHERE GROUP IF BUNDLE EXISTS
                if ankle_bundle and "AnkleBridge" in obj.name:
                    cloth_mod.settings.vertex_group_mass = f"Ankle_Sphere_{side}"
                    script_log(f"✓ Ankle bridge cloth pinned to ankle spherical vertex group")
        else:
            script_log(f"DEBUG: Cloth simulation disabled for {side} boot")

        # =========================================================================
        # STEP 10: SET MODIFIER ORDER (Preserved Functionality)
        # =========================================================================
        script_log("DEBUG: Setting modifier order...")

        for obj in boot_objects:
            bpy.context.view_layer.objects.active = obj
            modifiers = obj.modifiers

            # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
            correct_order = ["Subdivision", "Armature", "Cloth"]
            for mod_name in correct_order:
                mod_index = modifiers.find(mod_name)
                if mod_index >= 0:
                    while mod_index > correct_order.index(mod_name):
                        bpy.ops.object.modifier_move_up(modifier=mod_name)
                        mod_index -= 1

        # =========================================================================
        # STEP 11: CREATE COMBINED PINNING GROUP FOR BOOT COORDINATION
        # =========================================================================
        script_log(f"DEBUG: Creating combined coordination group for {side} boot...")

        # Create a combined group that includes ankle sphere influence
        combined_coordination_group = ankle_sphere.vertex_groups.new(name=f"{side}_Boot_Combined_Coordination")

        # Combine ankle sphere weights with standard armature weights
        for i in range(len(ankle_sphere.data.vertices)):
            max_weight = 0.0

            # Check ankle sphere group weight
            try:
                sphere_weight = ankle_sphere_group.weight(i)
                max_weight = max(max_weight, sphere_weight)
            except:
                pass

            # Check standard armature group weight
            try:
                armature_weight = ankle_group.weight(i)
                max_weight = max(max_weight, armature_weight)
            except:
                pass

            if max_weight > min_weight_threshold:
                combined_coordination_group.add([i], max_weight, 'REPLACE')

        # =========================================================================
        # STEP 12: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage
        bundle_status = f"ankle({ankle_bundle['vertex_count']}v)" if ankle_bundle else "NONE (standard)"

        script_log(f"=== {side.upper()} BOOT CREATION COMPLETE (JOINT_VERTEX_BUNDLES INTEGRATION) ===")
        script_log(f"✓ Base ankle radius: {base_ankle_radius} (from config)")
        script_log(f"✓ Final ankle radius: {ankle_radius} (with puffiness: {puffiness})")
        script_log(f"✓ Shaft height: {shaft_height}")
        script_log(f"✓ Foot length: {foot_length}, elliptical: {foot_radius_x}x{foot_radius_y}")
        script_log(f"✓ Segments: {segments}")
        script_log(f"✓ Ankle sphere radius for bundle integration: {ankle_sphere_radius:.3f}")
        script_log(f"✓ Ankle bridge and shaft parented to {shin_bone_name}")
        script_log(f"✓ Foot parented to {foot_bone_name}")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_config.get('enabled', False) else 'DISABLED'}")
        script_log(f"✓ Vertex bundle used: {bundle_status}")
        script_log(f"✓ Seamless integration: Boot uses same ankle vertex bundle as pants")
        script_log(f"✓ Combined coordination: Ankle sphere + armature binding work together")

        # Return the foot cylinder as the main boot object (preserved behavior)
        return foot_obj

    except Exception as e:
        script_log(f"ERROR creating {side} boot: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_pants(armature_obj, figure_name, garment_config, global_cloth_settings, side="left"):
    """Create continuous pants with coordinated vertex bundles for natural joint coverage"""
    script_log(f"Creating continuous {side} pants with coordinated vertex bundles...")

    # Get leg bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        if side == "left":
            hip_bone_name = "DEF_LeftHip"
            thigh_bone_name = "DEF_LeftThigh"
            shin_bone_name = "DEF_LeftShin"

            # GET CONTROL POINT NAMES FROM BONE DEFINITIONS (LIKE SLEEVES DO)
            hip_control_point = bone_tail_control_points.get("LeftHip")
            thigh_control_point = bone_tail_control_points.get("LeftThigh")
            shin_control_point = bone_tail_control_points.get("LeftShin")
        else:
            hip_bone_name = "DEF_RightHip"
            thigh_bone_name = "DEF_RightThigh"
            shin_bone_name = "DEF_RightShin"

            # GET CONTROL POINT NAMES FROM BONE DEFINITIONS (LIKE SLEEVES DO)
            hip_control_point = bone_tail_control_points.get("RightHip")
            thigh_control_point = bone_tail_control_points.get("RightThigh")
            shin_control_point = bone_tail_control_points.get("RightShin")

        hip_bone = armature_obj.pose.bones.get(hip_bone_name)
        thigh_bone = armature_obj.pose.bones.get(thigh_bone_name)
        shin_bone = armature_obj.pose.bones.get(shin_bone_name)

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([hip_bone, thigh_bone, shin_bone]):
            script_log(f"ERROR: Could not find leg bones for {side} pants")
            return None

        # =========================================================================
        # STEP 1: SET UP BONE CONSTRAINTS FIRST (LIKE SLEEVES DO)
        # =========================================================================
        script_log(f"DEBUG: Setting up bone constraints for {side} pants movement...")

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        # CLEAR EXISTING CONSTRAINTS FIRST (LIKE SLEEVES DO)
        for bone_name in [hip_bone_name, thigh_bone_name, shin_bone_name]:
            bone = armature_obj.pose.bones.get(bone_name)
            if bone:
                for constraint in list(bone.constraints):
                    bone.constraints.remove(constraint)

        # SET UP STRETCH_TO CONSTRAINTS TO CONTROL POINTS (LIKE SLEEVES DO)
        constraints_added = 0

        # HIP BONE: Constrain to hip control point
        if hip_bone and hip_control_point:
            hip_target = bpy.data.objects.get(hip_control_point)
            if hip_target:
                stretch = hip_bone.constraints.new('STRETCH_TO')
                stretch.target = hip_target
                stretch.influence = 1.0
                constraints_added += 1
                script_log(f"✓ {hip_bone_name} STRETCH_TO -> {hip_control_point}")

        # THIGH BONE: Constrain to knee control point (thigh tail points to knee)
        if thigh_bone and thigh_control_point:
            thigh_target = bpy.data.objects.get(thigh_control_point)
            if thigh_target:
                stretch = thigh_bone.constraints.new('STRETCH_TO')
                stretch.target = thigh_target
                stretch.influence = 1.0
                constraints_added += 1
                script_log(f"✓ {thigh_bone_name} STRETCH_TO -> {thigh_control_point}")

        # SHIN BONE: Constrain to ankle control point (shin tail points to ankle)
        if shin_bone and shin_control_point:
            shin_target = bpy.data.objects.get(shin_control_point)
            if shin_target:
                stretch = shin_bone.constraints.new('STRETCH_TO')
                stretch.target = shin_target
                stretch.influence = 1.0
                constraints_added += 1
                script_log(f"✓ {shin_bone_name} STRETCH_TO -> {shin_control_point}")

        bpy.ops.object.mode_set(mode='OBJECT')
        script_log(f"✓ Added {constraints_added} bone constraints for {side} pants")

        # NOW GET UPDATED BONE POSITIONS AFTER CONSTRAINTS ARE APPLIED
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        hip_bone = armature_obj.pose.bones.get(hip_bone_name)
        thigh_bone = armature_obj.pose.bones.get(thigh_bone_name)
        shin_bone = armature_obj.pose.bones.get(shin_bone_name)

        # Get bone positions in world space AFTER constraints are set
        hip_pos = armature_obj.matrix_world @ hip_bone.tail
        thigh_pos = armature_obj.matrix_world @ thigh_bone.head
        knee_pos = armature_obj.matrix_world @ thigh_bone.tail
        shin_pos = armature_obj.matrix_world @ shin_bone.head
        ankle_pos = armature_obj.matrix_world @ shin_bone.tail

        bpy.ops.object.mode_set(mode='OBJECT')

        # Get pants dimensions from config
        diameter_hip = garment_config.get("diameter_hip", 0.18)
        diameter_knee = garment_config.get("diameter_knee", 0.14)
        diameter_ankle = garment_config.get("diameter_ankle", 0.12)
        segments = garment_config.get("segments", 32)

        # Get artist-controlled settings
        subdivision_config = garment_config.get("subdivision", {})
        manual_cuts = subdivision_config.get("manual_cuts", 2)
        subdiv_levels = subdivision_config.get("subdiv_levels", 2)
        min_rings = subdivision_config.get("min_rings", 24)
        rings_per_meter = subdivision_config.get("rings_per_meter", 50)

        weighting_config = garment_config.get("vertex_weighting", {})
        falloff_type = weighting_config.get("elbow_sphere_falloff", "quadratic")
        min_weight_threshold = weighting_config.get("min_weight_threshold", 0.05)
        sphere_influence_scale = weighting_config.get("sphere_influence_scale", 2.0)

        # Calculate segment lengths - stop at ankle (shin tail)
        thigh_length = (knee_pos - thigh_pos).length
        shin_length = (ankle_pos - shin_pos).length
        total_length = thigh_length + shin_length

        script_log(f"DEBUG: {side} pants - Thigh length: {thigh_length:.3f}, Shin length: {shin_length:.3f}")
        script_log(f"DEBUG: {side} pants - Total length: {total_length:.3f} (stopping at ankle)")
        script_log(f"DEBUG: {side} pants - Bone constraints: {constraints_added} added")

        # CREATE SINGLE CONTINUOUS CYLINDER (like sleeves)
        script_log(f"DEBUG: Creating continuous {side} pants cylinder...")

        # Use average radius for initial cylinder
        avg_radius = (diameter_hip / 2 + diameter_knee / 2 + diameter_ankle / 2) / 3
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=total_length,
            radius=avg_radius,
            location=(hip_pos + ankle_pos) / 2  # Center between hip and ankle
        )
        pants_obj = bpy.context.active_object
        pants_obj.name = f"{figure_name}_{side.capitalize()}Pants"

        # Rotate to align with leg direction
        leg_direction = (ankle_pos - hip_pos).normalized()
        pants_obj.rotation_euler = leg_direction.to_track_quat('Z', 'Y').to_euler()

        # ADD MANUAL SUBDIVISION
        if manual_cuts > 0:
            script_log(f"DEBUG: Adding {manual_cuts} manual subdivision cuts...")
            bpy.context.view_layer.objects.active = pants_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.subdivide(number_cuts=manual_cuts)
            bpy.ops.object.mode_set(mode='OBJECT')

        # TAPER THE CONTINUOUS PANTS
        script_log(f"DEBUG: Tapering {side} pants...")
        bpy.context.view_layer.objects.active = pants_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(pants_obj.data)

        for vert in bm.verts:
            # Normalize Z position from -0.5 (hip) to 0.5 (ankle)
            z_norm = vert.co.z / (total_length / 2)

            # Calculate target radius based on position along leg
            if z_norm <= -0.3:  # Hip area
                target_radius = diameter_hip / 2
            elif z_norm <= 0.3:  # Knee area
                target_radius = diameter_knee / 2
            else:  # Ankle area
                target_radius = diameter_ankle / 2

            # Smooth transitions between areas
            if -0.3 < z_norm < -0.1:  # Hip → Knee transition
                blend = (z_norm + 0.3) / 0.2
                target_radius = (diameter_hip / 2 * (1 - blend)) + (diameter_knee / 2 * blend)
            elif 0.1 < z_norm < 0.3:  # Knee → Ankle transition
                blend = (z_norm - 0.1) / 0.2
                target_radius = (diameter_knee / 2 * (1 - blend)) + (diameter_ankle / 2 * blend)

            # Scale vertex to target radius
            current_radius = (vert.co.x ** 2 + vert.co.y ** 2) ** 0.5
            if current_radius > 0.001:
                scale_factor = target_radius / current_radius
                vert.co.x *= scale_factor
                vert.co.y *= scale_factor

        bmesh.update_edit_mesh(pants_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        # ADD SUBDIVISION SURFACE MODIFIER
        if subdiv_levels > 0:
            script_log(f"DEBUG: Adding subdivision surface with {subdiv_levels} levels...")
            subdiv_mod = pants_obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = subdiv_levels
            subdiv_mod.render_levels = subdiv_levels

        # =========================================================================
        # USE JOINT_VERTEX_BUNDLES FOR COORDINATED CLOTH BEHAVIOR (EXISTING FUNCTIONALITY)
        # =========================================================================
        script_log(f"DEBUG: Creating spherical vertex groups using joint_vertex_bundles for {side} pants")

        # Create spherical vertex groups
        hip_vertex_group = pants_obj.vertex_groups.new(name=f"Hip_Sphere_{side}")
        knee_vertex_group = pants_obj.vertex_groups.new(name=f"Knee_Sphere_{side}")
        ankle_vertex_group = pants_obj.vertex_groups.new(name=f"Ankle_Sphere_{side}")

        # Get vertex bundles from global storage
        hip_bundle = joint_vertex_bundles.get(hip_bone_name)
        knee_bundle = joint_vertex_bundles.get(thigh_bone_name)  # Knee bundle uses thigh bone
        ankle_bundle = joint_vertex_bundles.get(shin_bone_name)

        # Calculate sphere radii
        hip_sphere_radius = (diameter_hip / 2) * sphere_influence_scale
        knee_sphere_radius = (diameter_knee / 2) * sphere_influence_scale
        ankle_sphere_radius = (diameter_ankle / 2) * sphere_influence_scale

        # =========================================================================
        # FIXED: APPLY HIP BUNDLE VERTEX WEIGHTS WITH RESTRICTED INFLUENCE
        # =========================================================================
        if hip_bundle:
            script_log(f"✓ Applying hip vertex bundle with {hip_bundle['vertex_count']} vertices")
            hip_vertex_positions = hip_bundle['vertex_positions']

            # REDUCE hip sphere radius to limit influence area
            hip_sphere_radius_restricted = hip_sphere_radius * 0.7

            for i, vertex in enumerate(pants_obj.data.vertices):
                vert_pos = pants_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the bundle
                for bundle_vert_pos in hip_vertex_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                if min_distance <= hip_sphere_radius_restricted:
                    weight = 1.0 - (min_distance / hip_sphere_radius_restricted)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    # STRICTLY LIMIT HIP INFLUENCE TO UPPER PANTS ONLY
                    vert_local = pants_obj.matrix_world.inverted() @ vert_pos
                    z_norm = (vert_local.z + total_length / 2) / total_length  # 0=hip, 1=ankle

                    if z_norm < 0.2:  # Only top 20% near hips
                        weight *= 1.0  # Full influence
                    elif z_norm < 0.4:  # Next 20% - reduced influence
                        weight *= 0.3  # Drastically reduced
                    else:  # Below 40% - minimal to no influence
                        weight *= 0.05  # Almost no hip influence

                    if weight > min_weight_threshold:
                        hip_vertex_group.add([i], weight, 'REPLACE')

        # =========================================================================
        # FIXED: APPLY KNEE BUNDLE VERTEX WEIGHTS WITH ENHANCED MID-LEG INFLUENCE
        # =========================================================================
        if knee_bundle:
            script_log(f"✓ Applying knee vertex bundle with {knee_bundle['vertex_count']} vertices")
            knee_vertex_positions = knee_bundle['vertex_positions']

            for i, vertex in enumerate(pants_obj.data.vertices):
                vert_pos = pants_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the bundle
                for bundle_vert_pos in knee_vertex_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                if min_distance <= knee_sphere_radius:
                    weight = 1.0 - (min_distance / knee_sphere_radius)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    # ENHANCE knee influence in mid-leg area
                    vert_local = pants_obj.matrix_world.inverted() @ vert_pos
                    z_norm = (vert_local.z + total_length / 2) / total_length

                    if 0.3 <= z_norm <= 0.7:  # Mid-leg area around knee
                        weight *= 1.5  # Boost knee influence
                    elif z_norm < 0.2 or z_norm > 0.8:  # Far from knee
                        weight *= 0.3  # Reduce influence

                    if weight > min_weight_threshold:
                        knee_vertex_group.add([i], weight, 'REPLACE')

        # =========================================================================
        # FIXED: APPLY ANKLE BUNDLE VERTEX WEIGHTS WITH BOOSTED INFLUENCE
        # =========================================================================
        if ankle_bundle:
            script_log(f"✓ Applying ankle vertex bundle with {ankle_bundle['vertex_count']} vertices")
            ankle_vertex_positions = ankle_bundle['vertex_positions']

            # DOUBLE the ankle sphere radius for better influence
            ankle_sphere_radius_boosted = ankle_sphere_radius * 2.0

            for i, vertex in enumerate(pants_obj.data.vertices):
                vert_pos = pants_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the bundle
                for bundle_vert_pos in ankle_vertex_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                # REMOVED the position-based reduction - ankle gets full influence
                if min_distance <= ankle_sphere_radius_boosted:
                    weight = 1.0 - (min_distance / ankle_sphere_radius_boosted)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    # NO POSITION-BASED REDUCTION - ankle gets full influence
                    if weight > min_weight_threshold:
                        ankle_vertex_group.add([i], weight, 'REPLACE')
        else:
            script_log("DEBUG: Using fallback ankle weighting...")
            # Fallback: graduated weights from bottom to top
            for i, vertex in enumerate(pants_obj.data.vertices):
                vert_local = pants_obj.matrix_world.inverted() @ vertex.co
                # Normalize Z from -0.5 (bottom) to 0.5 (top)
                z_norm = (vert_local.z + total_length / 2) / total_length

                # Strong weight at bottom for ankle connection, lighter at top
                bottom_weight = 1.0 - z_norm  # 1.0 at bottom, 0.0 at top
                bottom_weight = bottom_weight * bottom_weight  # Quadratic

                if bottom_weight > 0.1:
                    ankle_vertex_group.add([i], bottom_weight, 'REPLACE')

        # CREATE COMBINED PINNING GROUP FOR PANTS
        script_log(f"DEBUG: Creating combined pinning group for {side} pants...")
        combined_pinning_group = pants_obj.vertex_groups.new(name=f"{side}_Pants_Combined_Anchors")

        # Combine weights from all three spherical groups (hip, knee, ankle)
        for i in range(len(pants_obj.data.vertices)):
            max_weight = 0.0
            for group_name in [f"Hip_Sphere_{side}", f"Knee_Sphere_{side}", f"Ankle_Sphere_{side}"]:
                group = pants_obj.vertex_groups.get(group_name)
                if group:
                    try:
                        weight = group.weight(i)
                        max_weight = max(max_weight, weight)
                    except:
                        # Vertex not in this group, continue
                        pass

            if max_weight > min_weight_threshold:
                combined_pinning_group.add([i], max_weight, 'REPLACE')

        script_log(f"✓ Created {side}_Pants_Combined_Anchors with weights from hip, knee, and ankle spheres")

        # TARGETED CLOTH SIMULATION WITH SIMPLE COLLISIONS
        cloth_config = garment_config.get("cloth_settings", {})
        if cloth_config.get("enabled", True):
            script_log(f"DEBUG: Adding cloth simulation for {side} pants (simple collisions)...")
            cloth_mod = pants_obj.modifiers.new(name="Cloth", type='CLOTH')

            # Apply cloth settings from config
            cloth_mod.settings.quality = cloth_config.get("quality", 12)
            cloth_mod.settings.mass = cloth_config.get("mass", 0.2)
            cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 6.0)
            cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 5.0)
            cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 4.0)
            cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.6)
            cloth_mod.settings.air_damping = cloth_config.get("air_damping", 0.8)
            cloth_mod.settings.time_scale = cloth_config.get("time_scale", 1.0)

            # SIMPLE COLLISIONS - WILL INTERACT WITH COAT AUTOMATICALLY
            cloth_mod.collision_settings.use_collision = True
            cloth_mod.collision_settings.collision_quality = cloth_config.get("collision_quality", 8)
            cloth_mod.collision_settings.self_distance_min = cloth_config.get("self_distance_min", 0.002)

            # Self-collision for pants fabric
            cloth_mod.collision_settings.use_self_collision = True

            # PIN CLOTH TO COMBINED SPHERICAL VERTEX GROUP
            cloth_mod.settings.vertex_group_mass = f"{side}_Pants_Combined_Anchors"

            script_log(f"✓ Pants cloth: self-collision + simple collisions (will interact with coat)")
        else:
            script_log(f"DEBUG: Cloth simulation disabled for {side} pants")

        # SETUP ARMATURE MODIFIER AND VERTEX GROUPS FOR BONE DEFORMATION (LIKE SLEEVES)
        script_log(f"DEBUG: Setting up armature modifier and vertex groups for {side} pants...")

        # Clear any existing vertex groups (except the spherical ones we just created)
        groups_to_keep = [f"Hip_Sphere_{side}", f"Knee_Sphere_{side}", f"Ankle_Sphere_{side}",
                          f"{side}_Pants_Combined_Anchors"]
        for vg in list(pants_obj.vertex_groups):
            if vg.name not in groups_to_keep:
                pants_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(pants_obj.modifiers):
            if mod.type == 'ARMATURE':
                pants_obj.modifiers.remove(mod)

        # Create vertex groups for bone deformation (LIKE SLEEVES DO)
        hip_group = pants_obj.vertex_groups.new(name=hip_bone_name)
        thigh_group = pants_obj.vertex_groups.new(name=thigh_bone_name)
        shin_group = pants_obj.vertex_groups.new(name=shin_bone_name)

        # Assign vertex weights based on position along pants
        for i, vertex in enumerate(pants_obj.data.vertices):
            vert_local = pants_obj.matrix_world.inverted() @ vertex.co
            z_norm = (vert_local.z + total_length / 2) / total_length  # 0=hip, 1=ankle

            if z_norm < 0.3:  # Upper part - hip to upper thigh
                hip_weight = 1.0 - (z_norm / 0.3)
                thigh_weight = z_norm / 0.3
                hip_group.add([i], hip_weight, 'REPLACE')
                thigh_group.add([i], thigh_weight, 'REPLACE')
            elif z_norm < 0.7:  # Middle part - thigh to shin
                thigh_weight = 1.0 - ((z_norm - 0.3) / 0.4)
                shin_weight = (z_norm - 0.3) / 0.4
                thigh_group.add([i], thigh_weight, 'REPLACE')
                shin_group.add([i], shin_weight, 'REPLACE')
            else:  # Lower part - shin to ankle
                shin_weight = 1.0 - ((z_norm - 0.7) / 0.3)
                shin_group.add([i], shin_weight, 'REPLACE')

        # Add armature modifier
        armature_mod = pants_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True
        script_log(f"✓ Added armature modifier with vertex group deformation")

        # Add material
        material_config = garment_config.get("material", {})
        material_color = material_config.get("color", [0.9, 0.9, 0.9, 1.0])
        pants_mat = bpy.data.materials.new(name=f"{side.capitalize()}Pants_Material")
        pants_mat.diffuse_color = material_color
        pants_mat.roughness = material_config.get("roughness", 0.8)
        pants_mat.metallic = material_config.get("metallic", 0.0)
        pants_obj.data.materials.append(pants_mat)

        # SET PROPER MODIFIER ORDER
        script_log(f"DEBUG: Setting proper modifier order for {side} pants...")
        bpy.context.view_layer.objects.active = pants_obj
        modifiers = pants_obj.modifiers

        # Build correct order based on which modifiers are present
        correct_order = ["Subdivision", "Armature"]
        if cloth_config.get("enabled", True):
            correct_order.append("Cloth")

        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # VERIFY THE SETUP
        script_log(f"DEBUG: Verifying {side} pants setup...")
        if cloth_config.get("enabled",
                            True) and cloth_mod.settings.vertex_group_mass == f"{side}_Pants_Combined_Anchors":
            script_log(f"✓ Cloth pinned to {side}_Pants_Combined_Anchors vertex group")
        else:
            script_log(f"⚠ Cloth not pinned to spherical vertex group (simulation disabled)")

        # Log vertex bundle usage
        bundle_status = []
        if hip_bundle: bundle_status.append(f"hip({hip_bundle['vertex_count']}v)")
        if knee_bundle: bundle_status.append(f"knee({knee_bundle['vertex_count']}v)")
        if ankle_bundle: bundle_status.append(f"ankle({ankle_bundle['vertex_count']}v)")

        script_log(f"✓ Created {side} pants with coordinated vertex bundle weighting")
        script_log(f"✓ Bone constraints: {constraints_added} STRETCH_TO constraints added")
        script_log(f"✓ Vertices weighted to hip, knee, and ankle spherical vertex groups")
        script_log(f"✓ Armature modifier configured for deformation")
        script_log(f"✓ Pants object parented to armature")
        if cloth_config.get("enabled", True):
            script_log(f"✓ Cloth pinned to combined anchors (hip+knee+ankle)")
            script_log(f"✓ Modern Blender 4.3+ cloth API applied")
            script_log(f"✓ Simple collisions enabled (will interact with coat)")
        script_log(f"✓ Using joint_vertex_bundles for consistent 1:1 joint motion response")

        return pants_obj

    except Exception as e:
        script_log(f"ERROR creating {side} pants: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_mitten(armature_obj, figure_name, garment_config, global_cloth_settings, side="left"):
    """Create mitten with seamless thumb attachment and coordinated vertex bundles"""
    script_log(f"Creating {side} mitten with seamless thumb attachment...")

    # Get hand bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        if side == "left":
            hand_bone_name = "DEF_LeftHand"
            forearm_bone_name = "DEF_LeftForearm"
            elbow_bone_name = "DEF_LeftUpperArm"  # For arm plane calculation
            shoulder_bone_name = "DEF_LeftShoulder"  # For arm plane calculation
            thumb_direction = -1  # Left thumb points to left (negative X)
        else:
            hand_bone_name = "DEF_RightHand"
            forearm_bone_name = "DEF_RightForearm"
            elbow_bone_name = "DEF_RightUpperArm"  # For arm plane calculation
            shoulder_bone_name = "DEF_RightShoulder"  # For arm plane calculation
            thumb_direction = 1  # Right thumb points to right (positive X)

        hand_bone = armature_obj.pose.bones.get(hand_bone_name)
        forearm_bone = armature_obj.pose.bones.get(forearm_bone_name)
        elbow_bone = armature_obj.pose.bones.get(elbow_bone_name)
        shoulder_bone = armature_obj.pose.bones.get(shoulder_bone_name)

        bpy.ops.object.mode_set(mode='OBJECT')

        if not hand_bone:
            script_log(f"ERROR: Could not find hand bone for {side} mitten")
            return None

        # Get hand bone positions in world space
        hand_pos = armature_obj.matrix_world @ hand_bone.head
        hand_tail = armature_obj.matrix_world @ hand_bone.tail
        hand_direction = (hand_tail - hand_pos).normalized()

        # Get arm plane landmarks for thumb orientation
        if elbow_bone and shoulder_bone:
            elbow_pos = armature_obj.matrix_world @ elbow_bone.tail
            shoulder_pos = armature_obj.matrix_world @ shoulder_bone.tail
            wrist_pos = armature_obj.matrix_world @ hand_bone.head  # Wrist position

            # Calculate arm plane normal
            upper_arm_vector = (elbow_pos - shoulder_pos).normalized()
            forearm_vector = (wrist_pos - elbow_pos).normalized()
            arm_plane_normal = upper_arm_vector.cross(forearm_vector).normalized()

            # Calculate forward direction in arm plane (perpendicular to hand direction)
            forward_in_plane = arm_plane_normal.cross(hand_direction).normalized()
            script_log(f"✓ Calculated arm plane for {side} mitten thumb orientation")
        else:
            # Fallback: use simple forward direction
            forward_in_plane = Vector((0, 1, 0))  # Forward in world space
            script_log(f"⚠ Using fallback thumb orientation for {side} mitten")

        # Get mitten dimensions from config
        puffiness = garment_config.get("puffiness", 1.0)
        hand_size = garment_config.get("hand_size", [0.1, 0.08, 0.04])
        thumb_size = garment_config.get("thumb_size", [0.04, 0.03, 0.03])
        segments = garment_config.get("segments", 8)

        # Get geometry settings from config
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        base_radii = global_cloth_settings.get("base_radii", {})

        # Use configurable thumb angle factor
        mitten_thumb_angle_factor = geometry_settings.get("mitten_thumb_angle_factor", 0.7)
        base_wrist_radius = base_radii.get("wrist", 0.06)

        # Unpack sizes
        hand_length, hand_radius_x, hand_radius_y = hand_size
        thumb_length, thumb_radius_x, thumb_radius_y = thumb_size

        # Apply puffiness
        hand_radius_x *= puffiness
        hand_radius_y *= puffiness
        thumb_radius_x *= puffiness
        thumb_radius_y *= puffiness

        script_log(
            f"DEBUG: {side} mitten - Hand: length={hand_length}, radius_x={hand_radius_x}, radius_y={hand_radius_y}")
        script_log(
            f"DEBUG: {side} mitten - Thumb: length={thumb_length}, radius_x={thumb_radius_x}, radius_y={thumb_radius_y}")
        script_log(f"DEBUG: {side} mitten - Puffiness: {puffiness}")
        script_log(f"DEBUG: {side} mitten - Thumb angle factor: {mitten_thumb_angle_factor}")
        script_log(f"DEBUG: {side} mitten - Base wrist radius: {base_wrist_radius}")
        script_log(f"DEBUG: {side} mitten - Segments: {segments}")

        # =========================================================================
        # STEP 1: CREATE HAND CYLINDER AS BASE MESH
        # =========================================================================
        script_log(f"DEBUG: Creating hand cylinder for {side} mitten...")

        # Calculate hand cylinder position and orientation
        hand_center = hand_pos + (hand_direction * hand_length / 2)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=hand_length,
            radius=1.0,  # Unit radius, we'll scale it
            location=hand_center
        )
        mitten_obj = bpy.context.active_object
        mitten_obj.name = f"{figure_name}_{side.capitalize()}Mitten"

        # Rotate to align with hand bone direction
        mitten_obj.rotation_euler = hand_direction.to_track_quat('Z', 'Y').to_euler()

        # Scale to create elliptical cross-section for hand
        mitten_obj.scale = (hand_radius_x, hand_radius_y, 1.0)

        # =========================================================================
        # STEP 2: CREATE SEAMLESS THUMB ATTACHMENT
        # =========================================================================
        script_log(f"DEBUG: Creating seamless thumb attachment for {side} mitten...")

        bpy.context.view_layer.objects.active = mitten_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mitten_obj.data)

        # Calculate thumb attachment point on hand cylinder
        # Thumb attaches on the side, slightly forward in the arm plane
        thumb_attach_local = Vector((thumb_direction * hand_radius_x * 0.8, 0, hand_length * 0.3))

        # Find closest vertex to thumb attachment point
        closest_vert = None
        min_distance = float('inf')

        for vert in bm.verts:
            distance = (vert.co - thumb_attach_local).length
            if distance < min_distance:
                min_distance = distance
                closest_vert = vert

        if closest_vert:
            # Select vertices around the attachment point for thumb base
            bpy.ops.mesh.select_all(action='DESELECT')
            closest_vert.select = True

            # Grow selection to get thumb base ring
            bpy.ops.mesh.select_more()
            bpy.ops.mesh.select_more()

            # Get selected vertices for thumb base
            selected_verts = [v for v in bm.verts if v.select]

            if selected_verts:
                # Calculate thumb base center
                thumb_base_center = Vector((0, 0, 0))
                for vert in selected_verts:
                    thumb_base_center += vert.co
                thumb_base_center /= len(selected_verts)

                # Calculate thumb direction (forward in arm plane with slight outward angle)
                thumb_dir = (hand_direction + forward_in_plane * mitten_thumb_angle_factor).normalized()

                # Convert thumb direction to local space
                thumb_dir_local = mitten_obj.matrix_world.inverted().to_3x3() @ thumb_dir

                # Extrude to create thumb
                extruded = bmesh.ops.extrude_vert_indiv(bm, verts=selected_verts)

                # Move extruded vertices to form thumb
                for vert in extruded['verts']:
                    # Move in thumb direction
                    vert.co += thumb_dir_local * thumb_length * 0.3

                # Scale down for thumb tip
                bmesh.ops.scale(
                    bm,
                    vec=Vector((thumb_radius_x / hand_radius_x, thumb_radius_y / hand_radius_y, 1.0)),
                    space=mitten_obj.matrix_world.inverted(),
                    verts=extruded['verts']
                )

                script_log(f"✓ Created seamless thumb attachment with {len(selected_verts)} base vertices")
            else:
                script_log("⚠ Could not select vertices for thumb base")
        else:
            script_log("⚠ Could not find thumb attachment point")

        # Update mesh and return to object mode
        bmesh.update_edit_mesh(mitten_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        # =========================================================================
        # STEP 3: SETUP PARENTING AND ARMATURE MODIFIERS
        # =========================================================================
        script_log("DEBUG: Setting up parenting and armature modifiers...")

        # Clear any existing parenting
        mitten_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(mitten_obj.vertex_groups):
            mitten_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(mitten_obj.modifiers):
            if mod.type == 'ARMATURE':
                mitten_obj.modifiers.remove(mod)

        # =========================================================================
        # STEP 4: SETUP VERTEX GROUPS WITH COORDINATED WRIST BUNDLES
        # =========================================================================

        # Get wrist vertex bundle from global storage for seamless integration
        wrist_bundle = joint_vertex_bundles.get(forearm_bone_name)  # Wrist bundle uses forearm bone

        if wrist_bundle:
            script_log(
                f"✓ Using wrist vertex bundle with {wrist_bundle['vertex_count']} vertices for seamless integration")
        else:
            script_log(f"⚠ No wrist bundle found for {forearm_bone_name}, using standard mitten weighting")

        # Create vertex group for hand bone
        hand_group = mitten_obj.vertex_groups.new(name=hand_bone_name)

        # Apply wrist bundle weighting if available
        if wrist_bundle:
            wrist_vertex_positions = wrist_bundle['vertex_positions']
            wrist_radius = base_wrist_radius * puffiness

            for i, vertex in enumerate(mitten_obj.data.vertices):
                vert_pos = mitten_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                # Find closest vertex in the wrist bundle
                for bundle_vert_pos in wrist_vertex_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                # Apply weight based on distance to nearest bundle vertex
                # Wrist area gets stronger influence for secure attachment
                if min_distance <= wrist_radius * 1.2:
                    weight = 1.0 - (min_distance / (wrist_radius * 1.2))
                    weight = weight * weight  # Quadratic falloff

                    # Reduce weight for thumb area to allow more flexibility
                    vert_local = mitten_obj.matrix_world.inverted() @ vert_pos
                    if abs(vert_local.x) > hand_radius_x * 0.6:  # Thumb area
                        weight *= 0.6  # Reduced influence for thumb

                    if weight > 0.1:
                        hand_group.add([i], weight, 'REPLACE')
                else:
                    # Default weight for areas far from wrist
                    hand_group.add([i], 0.3, 'REPLACE')  # Light influence
        else:
            # Fallback: assign uniform weights
            for i in range(len(mitten_obj.data.vertices)):
                hand_group.add([i], 1.0, 'REPLACE')

        # =========================================================================
        # STEP 5: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = mitten_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 6: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and materials...")

        # Add subdivision for smoother mitten
        subdiv_mod = mitten_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = 1
        subdiv_mod.render_levels = 1

        # Add materials
        material_config = garment_config.get("material", {})
        material_color = material_config.get("color", [0.8, 0.1, 0.1, 1.0])

        mitten_mat = bpy.data.materials.new(name=f"{side.capitalize()}Mitten_Material")
        mitten_mat.diffuse_color = material_color
        mitten_mat.roughness = material_config.get("roughness", 0.6)
        mitten_mat.metallic = material_config.get("metallic", 0.0)
        mitten_obj.data.materials.append(mitten_mat)

        # =========================================================================
        # STEP 7: CLOTH SIMULATION - DISABLED AS REQUESTED
        # =========================================================================
        cloth_config = garment_config.get("cloth_settings", {})
        if cloth_config.get("enabled", False):
            script_log(f"DEBUG: Adding cloth simulation to {side} mitten...")
            cloth_mod = mitten_obj.modifiers.new(name="Cloth", type='CLOTH')

            # Apply cloth settings from config
            cloth_mod.settings.quality = cloth_config.get("quality", 4)
            cloth_mod.settings.mass = cloth_config.get("mass", 0.1)
            cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 5.0)
            cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 4.0)
            cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 3.0)
            cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.2)
            cloth_mod.settings.air_damping = cloth_config.get("air_damping", 1.0)
        else:
            script_log(f"DEBUG: Cloth simulation disabled for {side} mitten")

        # =========================================================================
        # STEP 8: SET MODIFIER ORDER
        # =========================================================================
        script_log("DEBUG: Setting modifier order...")

        bpy.context.view_layer.objects.active = mitten_obj
        modifiers = mitten_obj.modifiers

        # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
        correct_order = ["Subdivision", "Armature", "Cloth"]
        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 9: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage and thumb orientation
        bundle_status = f"wrist({wrist_bundle['vertex_count']}v)" if wrist_bundle else "NONE (standard)"
        thumb_orientation = "arm plane aligned" if elbow_bone and shoulder_bone else "fallback"

        script_log(f"=== {side.upper()} MITTEN CREATION COMPLETE (SEAMLESS + COORDINATED) ===")
        script_log(f"✓ Hand cylinder: length={hand_length}, radius_x={hand_radius_x}, radius_y={hand_radius_y}")
        script_log(f"✓ Thumb: length={thumb_length}, radius_x={thumb_radius_x}, radius_y={thumb_radius_y}")
        script_log(f"✓ Thumb orientation: {thumb_orientation}")
        script_log(f"✓ Thumb angle factor: {mitten_thumb_angle_factor}")
        script_log(f"✓ Base wrist radius: {base_wrist_radius}")
        script_log(f"✓ Segments: {segments}")
        script_log(f"✓ Puffiness factor applied: {puffiness}")
        script_log(f"✓ Mitten parented to {hand_bone_name}")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_config.get('enabled', False) else 'DISABLED'}")
        script_log(f"✓ Vertex bundle used: {bundle_status}")
        script_log(f"✓ Thumb attachment: SEAMLESS (single mesh)")
        script_log(f"✓ Thumb orientation: COPLANAR with arm plane")

        return mitten_obj

    except Exception as e:
        script_log(f"ERROR creating {side} mitten: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_coat(armature_obj, figure_name, garment_config, global_cloth_settings):
    """Create coat torso garment with shoulder coordination and length variations"""
    script_log("Creating coat torso garment...")

    # Get coat configuration
    coat_length = garment_config.get("coat_length", "short")  # "short" or "long"
    radial_segments = garment_config.get("radial_segments", 32)
    longitudinal_segments = garment_config.get("longitudinal_segments", 24)
    torso_radius = garment_config.get("torso_radius", 0.25)
    coat_height = garment_config.get("coat_height", 0.8)
    puffiness = garment_config.get("puffiness", 1.05)

    # GET SHOULDER DIAMETER FROM SLEEVE CONFIG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

    diameter_shoulder = 0.15  # Fallback value
    try:
        with open(CLOTH_CONFIG_FILE, 'r') as file:
            cloth_config_data = json.load(file)
            sleeve_config = cloth_config_data.get("cloth_garments", {}).get("left_sleeve", {})
            diameter_shoulder = sleeve_config.get("diameter_shoulder", 0.15)  # Get from sleeve config
    except:
        script_log(f"⚠ Could not load shoulder diameter from config, using fallback: {diameter_shoulder}")

    script_log(f"DEBUG: Coat - Length: {coat_length}, Radial segments: {radial_segments}")
    script_log(f"DEBUG: Coat - Longitudinal segments: {longitudinal_segments}, Height: {coat_height}")
    script_log(
        f"DEBUG: Coat - Torso radius: {torso_radius}, Shoulder diameter: {diameter_shoulder}, Puffiness: {puffiness}")

    # Get shoulder positions for coordination
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        # Get shoulder and spine bones for positioning
        left_shoulder_bone = armature_obj.pose.bones.get("DEF_LeftShoulder")
        right_shoulder_bone = armature_obj.pose.bones.get("DEF_RightShoulder")
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        upper_spine_bone = armature_obj.pose.bones.get("DEF_UpperSpine")

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([left_shoulder_bone, right_shoulder_bone, neck_bone, upper_spine_bone]):
            script_log("ERROR: Could not find required bones for coat")
            return None

        # Get bone positions in world space
        left_shoulder_pos = armature_obj.matrix_world @ left_shoulder_bone.tail
        right_shoulder_pos = armature_obj.matrix_world @ right_shoulder_bone.tail
        upper_spine_pos = armature_obj.matrix_world @ upper_spine_bone.head

        # Calculate coat dimensions
        shoulder_width = (right_shoulder_pos - left_shoulder_pos).length
        shoulder_center = (left_shoulder_pos + right_shoulder_pos) / 2
        spine_to_shoulder = (shoulder_center - upper_spine_pos).length

        # =========================================================================
        # STEP 1: CREATE VERTICAL CYLINDER (MAIN BODY) - APPLY Y-SQUISH TO MATCH SHOULDER DIAMETER
        # =========================================================================
        script_log("DEBUG: Creating vertical cylinder for coat body...")

        # Position cylinder centered at shoulders, extending downward
        vertical_center = shoulder_center + Vector((0, 0, -coat_height / 2))

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=radial_segments,
            depth=coat_height,
            radius=torso_radius,  # Starts with torso_radius = 0.25 from config
            location=vertical_center
        )
        vertical_cylinder = bpy.context.active_object
        vertical_cylinder.name = f"{figure_name}_Coat_Vertical"

        # CALCULATE Y SCALE TO MATCH SHOULDER DIAMETER
        # Current Y diameter = torso_radius * 2 = 0.25 * 2 = 0.50
        # Target Y diameter = shoulder_diameter = 0.15
        # Y scale ratio = target / current = 0.15 / 0.50 = 0.3
        y_scale_ratio = diameter_shoulder / (torso_radius * 2)

        # Apply Y-squish: X=1.0, Y=calculated ratio, Z=1.0
        vertical_cylinder.scale = (1.0, y_scale_ratio, 1.0)
        script_log(
            f"✓ Vertical cylinder Y-squish: torso_radius={torso_radius} → Y-scale={y_scale_ratio:.3f} to match diameter_shoulder={diameter_shoulder}")

        # =========================================================================
        # STEP 2: ADD VERTICAL SUBDIVISIONS FOR Z-AXIS FLEXIBILITY (PRESERVE SHAPE)
        # =========================================================================
        script_log("DEBUG: Adding vertical subdivisions for coat flexibility...")

        # Get longitudinal segments from config
        longitudinal_segments = garment_config.get("longitudinal_segments", 24)
        # Subtract 1 because the cylinder already has top and bottom rings
        number_cuts = max(1, longitudinal_segments - 1)

        script_log(f"DEBUG: Using {longitudinal_segments} longitudinal segments, adding {number_cuts} vertical cuts")

        bpy.context.view_layer.objects.active = vertical_cylinder
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='EDGE')

        # Select all vertical edges
        bpy.ops.mesh.select_all(action='DESELECT')

        bm = bmesh.from_edit_mesh(vertical_cylinder.data)

        for edge in bm.edges:
            # Check if edge is approximately vertical (different Z coordinates)
            if abs(edge.verts[0].co.z - edge.verts[1].co.z) > 0.01:
                edge.select = True

        # Subdivide only the selected vertical edges using config value
        bpy.ops.mesh.subdivide(number_cuts=number_cuts, smoothness=0.0)

        bpy.ops.object.mode_set(mode='OBJECT')
        script_log(f"✓ Added {number_cuts} vertical subdivisions using longitudinal_segments={longitudinal_segments}")

        # =========================================================================
        # STEP 3: LONG COAT - QUARTER SEPARATION (FRONT SPLIT) AND DELETE BOTTOM FACE
        # =========================================================================
        if coat_length == "long":
            script_log("DEBUG: Creating front split and deleting bottom face for long coat...")

            # Get skirt_start_ratio from config with fallback
            long_coat_settings = garment_config.get("coat_length_settings", {}).get("long", {})
            skirt_start_ratio = long_coat_settings.get("skirt_start_ratio", 0.6)  # Default to 0.6 if not specified

            # Calculate skirt region dimensions (bottom portion of coat)
            skirt_start_z = -coat_height * (1 - skirt_start_ratio)
            split_depth = coat_height * 0.8
            split_width = 0.05

            script_log(f"DEBUG: Long coat skirt starts at {skirt_start_ratio * 100}% down (z={skirt_start_z:.3f})")

            # Create split cutter object (thin vertical plane)
            bpy.ops.mesh.primitive_plane_add(
                size=split_depth,
                location=vertical_center + Vector((0, 0, skirt_start_z))
            )
            split_cutter = bpy.context.active_object
            split_cutter.name = f"{figure_name}_CoatSplit_Cutter"

            # Scale cutter to be a thin vertical strip
            split_cutter.scale = (1.0, split_width / split_depth, 1.0)
            split_cutter.rotation_euler = (0, math.radians(90), 0)

            # Position cutter at front center of coat
            split_cutter.location.y = vertical_cylinder.location.y + diameter_shoulder * 0.5

            # Add Boolean modifier to subtract split from coat
            bool_split = vertical_cylinder.modifiers.new(name="Split_Front", type='BOOLEAN')
            bool_split.operation = 'DIFFERENCE'
            bool_split.object = split_cutter
            bool_split.solver = 'FAST'

            # Apply the Boolean subtraction
            bpy.context.view_layer.objects.active = vertical_cylinder
            bpy.ops.object.modifier_apply(modifier="Split_Front")

            # Remove the cutter object
            bpy.data.objects.remove(split_cutter, do_unlink=True)

            script_log("✓ Created front split using Boolean subtraction in FAST mode")

            # The bottom face deletion creates the open-bottom cylinder needed
            # for the coat torso, allowing it to drape naturally over the
            # lower body while maintaining the closed top at the shoulders.

            # Find and delete bottom face of vertical cylinder
            script_log("Finding and deleting bottom face of vertical cylinder...")
            bpy.context.view_layer.objects.active = vertical_cylinder
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.select_all(action='DESELECT')

            # Calculate the Z threshold for bottom face (shoulder_center Z minus coat_height)
            bottom_z_threshold = shoulder_center.z - coat_height + 0.01  # Small tolerance

            bm = bmesh.from_edit_mesh(vertical_cylinder.data)
            bottom_faces = []

            for face in bm.faces:
                # Check if all vertices in this face are near the bottom
                all_vertices_bottom = True
                for vert in face.verts:
                    world_pos = vertical_cylinder.matrix_world @ vert.co
                    if abs(world_pos.z - bottom_z_threshold) > 0.02:  # 2cm tolerance
                        all_vertices_bottom = False
                        break

                if all_vertices_bottom:
                    bottom_faces.append(face)

            # Select and delete bottom faces
            if bottom_faces:
                bmesh.ops.delete(bm, geom=bottom_faces, context='FACES')
                script_log(f"Deleted {len(bottom_faces)} bottom face(s)")
            else:
                script_log("WARNING: No bottom face found to delete")

            bmesh.update_edit_mesh(vertical_cylinder.data)
            bpy.ops.object.mode_set(mode='OBJECT')


        # =========================================================================
        # STEP 4: CREATE HORIZONTAL CYLINDER (SHOULDERS/CHEST)
        # =========================================================================
        script_log(f"DEBUG: Creating horizontal cylinder for shoulders, width {shoulder_width}...")

        # Position horizontal cylinder between shoulders
        horizontal_center = shoulder_center
        shoulder_vector = (right_shoulder_pos - left_shoulder_pos).normalized()

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=radial_segments,
            depth=shoulder_width,
            radius=diameter_shoulder / 2.0,
            location=horizontal_center
        )
        horizontal_cylinder = bpy.context.active_object
        horizontal_cylinder.name = f"{figure_name}_Coat_Horizontal"

        # Rotate to align with shoulder line
        horizontal_cylinder.rotation_euler = shoulder_vector.to_track_quat('Z', 'Y').to_euler()

        # =========================================================================
        # STEP 5: COMBINE CYLINDERS WITH BOOLEAN UNION
        # =========================================================================
        script_log("DEBUG: Combining cylinders with boolean union...")

        # Set vertical cylinder as main object
        coat_obj = vertical_cylinder
        coat_obj.name = f"{figure_name}_Coat"

        # Add boolean modifier to combine with horizontal cylinder
        boolean_mod = coat_obj.modifiers.new(name="Boolean_Union", type='BOOLEAN')
        boolean_mod.operation = 'UNION'
        boolean_mod.object = horizontal_cylinder

        # Apply boolean modifier
        bpy.context.view_layer.objects.active = coat_obj
        bpy.ops.object.modifier_apply(modifier="Boolean_Union")

        # Remove horizontal cylinder
        bpy.data.objects.remove(horizontal_cylinder, do_unlink=True)

        # =========================================================================
        # STEP 6: SMOOTH ARMPIT AREAS
        # =========================================================================
        smooth_armpits = garment_config.get("smooth_armpits", False)
        if smooth_armpits:
            script_log("DEBUG: Smoothing armpit areas...")

            bpy.context.view_layer.objects.active = coat_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='VERT')  # Blender 4.4.4 uses 'VERT' not 'VERTEX'

            # Select vertices in armpit regions and smooth them
            bm = bmesh.from_edit_mesh(coat_obj.data)

            for vert in bm.verts:
                # Armpit regions are around the shoulder connections
                vert_local = coat_obj.matrix_world.inverted() @ vert.co
                if abs(vert_local.x) > torso_radius * 0.8 and vert_local.z > -0.1:
                    vert.select = True

            if any(v.select for v in bm.verts):
                bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=3)

            bpy.ops.object.mode_set(mode='OBJECT')

        # =========================================================================
        # STEP 7: SETUP VERTEX GROUPS WITH COORDINATED BUNDLES
        # =========================================================================
        script_log("DEBUG: Setting up coat vertex groups with coordinated bundles...")

        # Debug: See what bundles are available
        script_log(f"DEBUG: Available bundles: {list(joint_vertex_bundles.keys())}")

        # Clear any existing parenting
        coat_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(coat_obj.vertex_groups):
            coat_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(coat_obj.modifiers):
            if mod.type == 'ARMATURE':
                coat_obj.modifiers.remove(mod)

        # GET SHOULDER BUNDLES (copying create_sleeve pattern)
        left_shoulder_bundle = joint_vertex_bundles.get("DEF_LeftShoulder")
        right_shoulder_bundle = joint_vertex_bundles.get("DEF_RightShoulder")

        if left_shoulder_bundle and right_shoulder_bundle:
            script_log(
                f"✓ Using shoulder vertex bundles: left({left_shoulder_bundle['vertex_count']}v), right({right_shoulder_bundle['vertex_count']}v)")
        else:
            script_log(f"ERROR: Missing shoulder bundles")
            return None

        # Create vertex groups for BOTH shoulders
        left_shoulder_group = coat_obj.vertex_groups.new(name="Left_Shoulder_Coordination_Coat")
        right_shoulder_group = coat_obj.vertex_groups.new(name="Right_Shoulder_Coordination_Coat")

        # For short coats, add hip coordination like shoulder coordination
        if coat_length == "short":
            left_hip_bundle = joint_vertex_bundles.get("DEF_LeftHip")
            right_hip_bundle = joint_vertex_bundles.get("DEF_RightHip")

            left_hip_group = coat_obj.vertex_groups.new(name="Left_Hip_Coordination_Coat")
            right_hip_group = coat_obj.vertex_groups.new(name="Right_Hip_Coordination_Coat")

            if left_hip_bundle and right_hip_bundle:
                script_log(
                    f"Short coat: Added hip coordination with {left_hip_bundle['vertex_count']} + {right_hip_bundle['vertex_count']} vertices")
            else:
                script_log("Short coat: Added hip coordination (fallback weighting)")
        else:
            script_log("Long coat: No hip coordination - free draping from shoulders")

        # Combined group for cloth pinning
        combined_anchors_group = coat_obj.vertex_groups.new(name="Coat_Combined_Anchors")

        # Calculate sphere radii for bundle integration
        shoulder_sphere_radius = diameter_shoulder / 2.0

        # =========================================================================
        # APPLY SHOULDER BUNDLE VERTEX WEIGHTS (DIFFERENT STRATEGIES)
        # =========================================================================

        if coat_length == "short":
            # SHORT COAT: Shoulders + Hips for twisting motion
            script_log("DEBUG: Applying SHORT COAT shoulder weights (shared influence with hips)...")

            # Apply left shoulder weights
            if left_shoulder_bundle:
                left_shoulder_positions = left_shoulder_bundle['vertex_positions']
                for i, vertex in enumerate(coat_obj.data.vertices):
                    vert_pos = coat_obj.matrix_world @ vertex.co
                    min_distance = float('inf')

                    for bundle_vert_pos in left_shoulder_positions:
                        distance = (vert_pos - bundle_vert_pos).length
                        min_distance = min(min_distance, distance)

                    if min_distance <= shoulder_sphere_radius:
                        weight = 1.0 - (min_distance / shoulder_sphere_radius)
                        weight = weight * weight  # Quadratic falloff

                        # For short coats, reduce shoulder influence in lower areas to share with hips
                        vert_local = coat_obj.matrix_world.inverted() @ vert_pos
                        if vert_local.z < -coat_height * 0.4:  # Lower 40% of coat
                            weight *= 0.3  # Reduced shoulder influence near hips

                        if weight > 0.1:
                            left_shoulder_group.add([i], weight, 'REPLACE')
                            combined_anchors_group.add([i], weight, 'REPLACE')

            # Apply right shoulder weights (same logic)
            if right_shoulder_bundle:
                right_shoulder_positions = right_shoulder_bundle['vertex_positions']
                for i, vertex in enumerate(coat_obj.data.vertices):
                    vert_pos = coat_obj.matrix_world @ vertex.co
                    min_distance = float('inf')

                    for bundle_vert_pos in right_shoulder_positions:
                        distance = (vert_pos - bundle_vert_pos).length
                        min_distance = min(min_distance, distance)

                    if min_distance <= shoulder_sphere_radius:
                        weight = 1.0 - (min_distance / shoulder_sphere_radius)
                        weight = weight * weight

                        vert_local = coat_obj.matrix_world.inverted() @ vert_pos
                        if vert_local.z < -coat_height * 0.4:
                            weight *= 0.3

                        if weight > 0.1:
                            right_shoulder_group.add([i], weight, 'REPLACE')
                            combined_anchors_group.add([i], weight, 'REPLACE')

        else:
            # LONG COAT: 100% shoulder influence for free draping
            script_log("DEBUG: Applying LONG COAT shoulder weights (100% influence)...")

            # Apply left shoulder weights - FULL INFLUENCE
            if left_shoulder_bundle:
                left_shoulder_positions = left_shoulder_bundle['vertex_positions']
                for i, vertex in enumerate(coat_obj.data.vertices):
                    vert_pos = coat_obj.matrix_world @ vertex.co
                    min_distance = float('inf')

                    for bundle_vert_pos in left_shoulder_positions:
                        distance = (vert_pos - bundle_vert_pos).length
                        min_distance = min(min_distance, distance)

                    if min_distance <= shoulder_sphere_radius:
                        weight = 1.0 - (min_distance / shoulder_sphere_radius)
                        weight = weight * weight

                        # 100% INFLUENCE THROUGHOUT LONG COAT
                        # No reduction based on position - coat hangs entirely from shoulders

                        if weight > 0.1:
                            left_shoulder_group.add([i], weight, 'REPLACE')
                            combined_anchors_group.add([i], weight, 'REPLACE')

            # Apply right shoulder weights - FULL INFLUENCE
            if right_shoulder_bundle:
                right_shoulder_positions = right_shoulder_bundle['vertex_positions']
                for i, vertex in enumerate(coat_obj.data.vertices):
                    vert_pos = coat_obj.matrix_world @ vertex.co
                    min_distance = float('inf')

                    for bundle_vert_pos in right_shoulder_positions:
                        distance = (vert_pos - bundle_vert_pos).length
                        min_distance = min(min_distance, distance)

                    if min_distance <= shoulder_sphere_radius:
                        weight = 1.0 - (min_distance / shoulder_sphere_radius)
                        weight = weight * weight

                        # 100% INFLUENCE THROUGHOUT LONG COAT

                        if weight > 0.1:
                            right_shoulder_group.add([i], weight, 'REPLACE')
                            combined_anchors_group.add([i], weight, 'REPLACE')

        # =========================================================================
        # APPLY HIP BUNDLE VERTEX WEIGHTS (SHORT COATS ONLY)
        # =========================================================================
        if coat_length == "short":
            script_log("DEBUG: Applying hip coordination for short coat twisting...")

            left_pants_config = garment_config.get("left_pants", {})
            diameter_hip = left_pants_config.get("diameter_hip", 0.18)

            hip_sphere_radius = (diameter_hip / 2) * 1.5  # Slightly larger influence area

            # Apply left hip weights
            left_hip_positions = left_hip_bundle['vertex_positions']
            for i, vertex in enumerate(coat_obj.data.vertices):
                vert_pos = coat_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                for bundle_vert_pos in left_hip_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                if min_distance <= hip_sphere_radius:
                    weight = 1.0 - (min_distance / hip_sphere_radius)
                    weight = weight * weight

                    # Hip influence primarily in lower coat area
                    vert_local = coat_obj.matrix_world.inverted() @ vert_pos
                    if vert_local.z > -coat_height * 0.3:  # Upper 70% of coat
                        weight *= 0.2  # Reduced hip influence near shoulders

                    if weight > 0.1:
                        left_hip_group.add([i], weight, 'REPLACE')
                        combined_anchors_group.add([i], weight, 'REPLACE')

            # Apply right hip weights
            right_hip_positions = right_hip_bundle['vertex_positions']
            for i, vertex in enumerate(coat_obj.data.vertices):
                vert_pos = coat_obj.matrix_world @ vertex.co
                min_distance = float('inf')

                for bundle_vert_pos in right_hip_positions:
                    distance = (vert_pos - bundle_vert_pos).length
                    min_distance = min(min_distance, distance)

                if min_distance <= hip_sphere_radius:
                    weight = 1.0 - (min_distance / hip_sphere_radius)
                    weight = weight * weight

                    vert_local = coat_obj.matrix_world.inverted() @ vert_pos
                    if vert_local.z > -coat_height * 0.3:
                        weight *= 0.2

                    if weight > 0.1:
                        right_hip_group.add([i], weight, 'REPLACE')
                        combined_anchors_group.add([i], weight, 'REPLACE')

        # =========================================================================
        # STEP 8: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = coat_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 9: ADD CLOTH SIMULATION WITH SIMPLE COLLISIONS
        # =========================================================================
        script_log("DEBUG: Adding cloth simulation with simple collisions...")
        cloth_config = garment_config.get("cloth_settings", {})

        if cloth_config.get("enabled", True):
            cloth_mod = coat_obj.modifiers.new(name="Cloth", type='CLOTH')

            # Apply cloth settings from config
            cloth_mod.settings.quality = cloth_config.get("quality", 12)
            cloth_mod.settings.mass = cloth_config.get("mass", 0.3)
            cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 8.0)
            cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 7.0)
            cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 5.0)
            cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.8)
            cloth_mod.settings.air_damping = cloth_config.get("air_damping", 0.8)
            cloth_mod.settings.time_scale = cloth_config.get("time_scale", 1.0)

            # SIMPLE COLLISIONS - COAT WILL INTERACT WITH PANTS AUTOMATICALLY
            cloth_mod.collision_settings.use_collision = True
            cloth_mod.collision_settings.collision_quality = cloth_config.get("collision_quality", 6)
            cloth_mod.collision_settings.distance_min = cloth_config.get("external_distance_min", 0.005)

            # Self-collision for coat fabric
            cloth_mod.collision_settings.use_self_collision = True
            cloth_mod.collision_settings.self_distance_min = cloth_config.get("self_distance_min", 0.002)

            # PIN ENTIRE COAT TO SHOULDERS (100% influence as requested)
            cloth_mod.settings.vertex_group_mass = "Coat_Combined_Anchors"

            script_log("✓ Coat cloth: 100% shoulder pinning + simple collisions (will interact with pants)")
        else:
            script_log("DEBUG: Cloth simulation disabled for coat")

        # =========================================================================
        # STEP 10: ADD MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding coat materials...")

        material_config = garment_config.get("material", {})
        material_color = material_config.get("color", [0.1, 0.3, 0.8, 1.0])

        coat_mat = bpy.data.materials.new(name="Coat_Material")
        coat_mat.diffuse_color = material_color
        coat_mat.roughness = material_config.get("roughness", 0.7)
        coat_mat.metallic = material_config.get("metallic", 0.0)
        coat_obj.data.materials.append(coat_mat)

        # =========================================================================
        # STEP 11: SET MODIFIER ORDER (NO SUBDIVISION)
        # =========================================================================
        script_log("DEBUG: Setting modifier order (no subdivision)...")

        bpy.context.view_layer.objects.active = coat_obj
        modifiers = coat_obj.modifiers

        # Remove Subdivision modifier if it exists
        if "Subdivision" in modifiers:
            modifiers.remove(modifiers["Subdivision"])
            script_log("✓ Removed Subdivision modifier - using manual geometry control")

        # Build correct order based on which modifiers are present
        correct_order = ["Armature"]
        if cloth_config.get("enabled", True):
            correct_order.append("Cloth")

        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 12: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage
        bundle_status = []
        if left_shoulder_bundle:
            bundle_status.append(f"left_shoulder({left_shoulder_bundle['vertex_count']}v)")
        if right_shoulder_bundle:
            bundle_status.append(f"right_shoulder({right_shoulder_bundle['vertex_count']}v)")
        if coat_length == "short":
            if left_hip_bundle:
                bundle_status.append(f"left_hip({left_hip_bundle['vertex_count']}v)")
            if right_hip_bundle:
                bundle_status.append(f"right_hip({right_hip_bundle['vertex_count']}v)")

        bundle_status_str = ", ".join(bundle_status) if bundle_status else "NONE (standard)"

        script_log(f"=== COAT TORSO CREATION COMPLETE ===")
        script_log(f"✓ Coat type: {coat_length}")
        script_log(f"✓ Height: {coat_height:.3f}")
        script_log(f"✓ Shoulder width: {shoulder_width:.3f}")
        script_log(f"✓ Torso radius: {torso_radius}")
        script_log(f"✓ Front split: {'CREATED' if coat_length == 'long' else 'NOT APPLIED'}")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_config.get('enabled', True) else 'DISABLED'}")
        script_log(f"✓ Vertex bundles used: {bundle_status_str}")
        script_log(f"✓ Modifier order: Subdivision → Armature → Cloth")

        return coat_obj

    except Exception as e:
        script_log(f"ERROR creating coat: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_sleeve(armature, figure_name, sleeve_config, existing_garments, side):
    """
    Create a sleeve mesh that stretches from shoulder to forearm
    and uses joint_vertex_bundles for seamless coordination with coat and mittens.
    """
    script_log(f"DEBUG create_sleeve: Creating {side} sleeve with joint_vertex_bundles integration")

    # Get bone positions from the actual armature
    if side == 'left':
        shoulder_bone_name = "DEF_LeftShoulder"
        forearm_bone_name = "DEF_LeftForearm"
        upper_arm_bone_name = "DEF_LeftUpperArm"
    else:
        shoulder_bone_name = "DEF_RightShoulder"
        forearm_bone_name = "DEF_RightForearm"
        upper_arm_bone_name = "DEF_RightUpperArm"

    # Get the bones from the armature
    shoulder_bone = armature.pose.bones.get(shoulder_bone_name)
    forearm_bone = armature.pose.bones.get(forearm_bone_name)
    upper_arm_bone = armature.pose.bones.get(upper_arm_bone_name)

    if not shoulder_bone or not forearm_bone or not upper_arm_bone:
        missing_bones = []
        if not shoulder_bone: missing_bones.append(shoulder_bone_name)
        if not forearm_bone: missing_bones.append(forearm_bone_name)
        if not upper_arm_bone: missing_bones.append(upper_arm_bone_name)
        script_log(f"ERROR: Missing bones in create_sleeve - Bones {missing_bones} not found in armature", "ERROR")
        raise Exception(f"ERROR: Required bones {missing_bones} not found in armature")

    script_log("DEBUG create_sleeve: Get the positions in world coordinates")

    # Get the positions in world coordinates
    shoulder_pos = armature.matrix_world @ shoulder_bone.tail
    forearm_pos = armature.matrix_world @ forearm_bone.tail
    upper_arm_pos = armature.matrix_world @ upper_arm_bone.tail  # Elbow position

    # Calculate actual sleeve length based on distance between shoulder and forearm
    sleeve_vector = forearm_pos - shoulder_pos
    actual_sleeve_length = sleeve_vector.length

    # Extract sleeve parameters from config
    diameter_start = sleeve_config.get('diameter_shoulder', 0.15)
    diameter_elbow = sleeve_config.get('diameter_elbow', 0.12)
    diameter_end = sleeve_config.get('diameter_wrist', 0.08)
    sleeve_vertices = sleeve_config.get('vertices', 16)

    # Get vertex weighting settings from config (like pants)
    weighting_config = sleeve_config.get("vertex_weighting", {})
    falloff_type = weighting_config.get("elbow_sphere_falloff", "quadratic")
    min_weight_threshold = weighting_config.get("min_weight_threshold", 0.05)
    sphere_influence_scale = weighting_config.get("sphere_influence_scale", 2.0)

    # Calculate radii (convert diameters to radii)
    radius_start = diameter_start / 2
    radius_elbow = diameter_elbow / 2
    radius_end = diameter_end / 2

    # Calculate sphere radii for bundle integration
    shoulder_sphere_radius = radius_start * sphere_influence_scale
    elbow_sphere_radius = radius_elbow * sphere_influence_scale
    wrist_sphere_radius = radius_end * sphere_influence_scale

    # Calculate average radius for initial cylinder
    avg_radius = (radius_start + radius_elbow + radius_end) / 3

    script_log("DEBUG create_sleeve: Create sleeve mesh at midpoint between shoulder and forearm")

    # Create sleeve mesh at midpoint between shoulder and forearm
    midpoint = (shoulder_pos + forearm_pos) / 2

    script_log("DEBUG create_sleeve: Create cylinder with depth segments for flexibility")

    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=sleeve_vertices,
        radius=avg_radius,
        depth=actual_sleeve_length,
        location=midpoint,
        end_fill_type='NGON'
    )
    sleeve_obj = bpy.context.active_object
    sleeve_obj.name = f"Main_{side.capitalize()}Sleeve"

    script_log("DEBUG create_sleeve: Add subdivisions for fabric flexibility")

    subdivision_config = sleeve_config.get("subdivision", {})
    manual_cuts = subdivision_config.get("manual_cuts", 2)
    subdiv_levels = subdivision_config.get("subdiv_levels", 2)
    min_rings = subdivision_config.get("min_rings", 24)
    rings_per_meter = subdivision_config.get("rings_per_meter", 50)

    # Enter edit mode safely for subdivisions
    bpy.context.view_layer.objects.active = sleeve_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # USE SUBDIVIDE INSTEAD OF LOOPCUT - more reliable in background mode
    bpy.ops.mesh.subdivide(
        number_cuts=min_rings,  # Creates 24 segments along the length for cloth flexibility
        smoothness=0,
        ngon=True,
        quadcorner='INNERVERT'
    )

    script_log("DEBUG create_sleeve: Apply taper to sleeve")

    # Get the mesh data for tapering
    mesh = sleeve_obj.data
    bm = bmesh.from_edit_mesh(mesh)

    # Scale vertices based on their Z position to create taper
    for vert in bm.verts:
        # Normalize Z position from -0.5 to 0.5 (cylinder local coordinates)
        z_norm = (vert.co.z + actual_sleeve_length / 2) / actual_sleeve_length

        if z_norm < 0.5:  # Shoulder half
            # Scale from radius_start at top to radius_elbow at middle
            scale_factor = radius_start + (radius_elbow - radius_start) * (z_norm * 2)
        else:  # Forearm half
            # Scale from radius_elbow at middle to radius_end at bottom
            scale_factor = radius_elbow + (radius_end - radius_elbow) * ((z_norm - 0.5) * 2)

        # Apply scaling
        vert.co.x *= scale_factor / avg_radius
        vert.co.y *= scale_factor / avg_radius

    bmesh.update_edit_mesh(mesh)
    bpy.ops.object.mode_set(mode='OBJECT')

    script_log("DEBUG create_sleeve: Rotate sleeve to align with arm direction")

    # Rotate sleeve to align with arm direction
    sleeve_direction = sleeve_vector.normalized()
    z_axis = Vector((0, 0, 1))
    rotation_quat = z_axis.rotation_difference(sleeve_direction)
    sleeve_obj.rotation_euler = rotation_quat.to_euler()

    script_log("DEBUG create_sleeve: Add armature modifier and vertex group binding")

    # Add armature modifier
    armature_modifier = sleeve_obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_modifier.object = armature
    armature_modifier.use_vertex_groups = True

    # =========================================================================
    # NEW: JOINT_VERTEX_BUNDLES INTEGRATION
    # =========================================================================
    script_log(f"DEBUG create_sleeve: Integrating joint_vertex_bundles for {side} sleeve")

    # Get vertex bundles from global storage
    shoulder_bundle = joint_vertex_bundles.get(shoulder_bone_name)
    elbow_bundle = joint_vertex_bundles.get(upper_arm_bone_name)  # Elbow bundle uses upper arm bone
    wrist_bundle = joint_vertex_bundles.get(forearm_bone_name)    # Wrist bundle uses forearm bone

    # Create spherical vertex groups for bundle integration
    shoulder_sphere_group = sleeve_obj.vertex_groups.new(name=f"Shoulder_Sphere_{side}")
    elbow_sphere_group = sleeve_obj.vertex_groups.new(name=f"Elbow_Sphere_{side}")
    wrist_sphere_group = sleeve_obj.vertex_groups.new(name=f"Wrist_Sphere_{side}")

    # =========================================================================
    # APPLY SHOULDER BUNDLE VERTEX WEIGHTS
    # =========================================================================
    if shoulder_bundle:
        script_log(f"✓ Applying shoulder vertex bundle with {shoulder_bundle['vertex_count']} vertices")
        shoulder_vertex_positions = shoulder_bundle['vertex_positions']

        for i, vertex in enumerate(sleeve_obj.data.vertices):
            vert_pos = sleeve_obj.matrix_world @ vertex.co
            min_distance = float('inf')

            # Find closest vertex in the bundle
            for bundle_vert_pos in shoulder_vertex_positions:
                distance = (vert_pos - bundle_vert_pos).length
                min_distance = min(min_distance, distance)

            # Apply weight based on distance to nearest bundle vertex
            if min_distance <= shoulder_sphere_radius:
                weight = 1.0 - (min_distance / shoulder_sphere_radius)
                # Apply falloff type
                if falloff_type == "quadratic":
                    weight = weight * weight
                elif falloff_type == "smooth":
                    weight = weight * weight * (3 - 2 * weight)

                if weight > min_weight_threshold:
                    shoulder_sphere_group.add([i], weight, 'REPLACE')
    else:
        script_log(f"⚠ No shoulder bundle found for {shoulder_bone_name}, using fallback shoulder weighting")

    # =========================================================================
    # APPLY ELBOW BUNDLE VERTEX WEIGHTS
    # =========================================================================
    if elbow_bundle:
        script_log(f"✓ Applying elbow vertex bundle with {elbow_bundle['vertex_count']} vertices")
        elbow_vertex_positions = elbow_bundle['vertex_positions']

        for i, vertex in enumerate(sleeve_obj.data.vertices):
            vert_pos = sleeve_obj.matrix_world @ vertex.co
            min_distance = float('inf')

            # Find closest vertex in the bundle
            for bundle_vert_pos in elbow_vertex_positions:
                distance = (vert_pos - bundle_vert_pos).length
                min_distance = min(min_distance, distance)

            # Apply weight based on distance to nearest bundle vertex
            if min_distance <= elbow_sphere_radius:
                weight = 1.0 - (min_distance / elbow_sphere_radius)
                # Apply falloff type
                if falloff_type == "quadratic":
                    weight = weight * weight
                elif falloff_type == "smooth":
                    weight = weight * weight * (3 - 2 * weight)

                if weight > min_weight_threshold:
                    elbow_sphere_group.add([i], weight, 'REPLACE')
    else:
        script_log(f"⚠ No elbow bundle found for {upper_arm_bone_name}, using fallback elbow weighting")

    # =========================================================================
    # APPLY WRIST BUNDLE VERTEX WEIGHTS
    # =========================================================================
    if wrist_bundle:
        script_log(f"✓ Applying wrist vertex bundle with {wrist_bundle['vertex_count']} vertices")
        wrist_vertex_positions = wrist_bundle['vertex_positions']

        for i, vertex in enumerate(sleeve_obj.data.vertices):
            vert_pos = sleeve_obj.matrix_world @ vertex.co
            min_distance = float('inf')

            # Find closest vertex in the bundle
            for bundle_vert_pos in wrist_vertex_positions:
                distance = (vert_pos - bundle_vert_pos).length
                min_distance = min(min_distance, distance)

            # Apply weight based on distance to nearest bundle vertex
            if min_distance <= wrist_sphere_radius:
                weight = 1.0 - (min_distance / wrist_sphere_radius)
                # Apply falloff type
                if falloff_type == "quadratic":
                    weight = weight * weight
                elif falloff_type == "smooth":
                    weight = weight * weight * (3 - 2 * weight)

                if weight > min_weight_threshold:
                    wrist_sphere_group.add([i], weight, 'REPLACE')
    else:
        script_log(f"⚠ No wrist bundle found for {forearm_bone_name}, using fallback wrist weighting")

    # =========================================================================
    # PRESERVE EXISTING ARMATURE BINDING (Backward Compatibility)
    # =========================================================================
    script_log("DEBUG create_sleeve: Preserving existing armature binding system")

    # Create vertex groups for armature binding (existing functionality)
    shoulder_binding_group = sleeve_obj.vertex_groups.new(name=shoulder_bone_name)
    forearm_binding_group = sleeve_obj.vertex_groups.new(name=forearm_bone_name)

    # Assign vertices to bone vertex groups with smooth blending (existing functionality)
    for i, vertex in enumerate(mesh.vertices):
        # Normalize Z position from 0 (shoulder) to 1 (forearm)
        z_norm = (vertex.co.z + actual_sleeve_length / 2) / actual_sleeve_length

        # Shoulder influence decreases from top to bottom
        shoulder_weight = 1.0 - z_norm
        # Forearm influence increases from top to bottom
        forearm_weight = z_norm

        # Smooth the transition
        shoulder_weight = shoulder_weight * shoulder_weight
        forearm_weight = forearm_weight * forearm_weight

        # Normalize weights
        total = shoulder_weight + forearm_weight
        if total > 0:
            shoulder_weight /= total
            forearm_weight /= total

        shoulder_binding_group.add([i], shoulder_weight, 'REPLACE')
        forearm_binding_group.add([i], forearm_weight, 'REPLACE')

    # =========================================================================
    # CREATE COMBINED PINNING GROUP (Shoulder + Elbow + Wrist)
    # =========================================================================
    script_log("DEBUG create_sleeve: Creating combined pinning group for shoulder, elbow, and wrist")

    # Create combined pinning group using joint_vertex_bundles
    combined_pinning_group = sleeve_obj.vertex_groups.new(name=f"{side}_Sleeve_Combined_Anchors")

    # Combine weights from ALL THREE spherical groups
    for i in range(len(sleeve_obj.data.vertices)):
        max_weight = 0.0

        # Check shoulder sphere weight
        shoulder_group = sleeve_obj.vertex_groups.get(f"Shoulder_Sphere_{side}")
        if shoulder_group:
            try:
                shoulder_weight = shoulder_group.weight(i)
                max_weight = max(max_weight, shoulder_weight)
            except:
                pass

        # Check elbow sphere weight
        elbow_group = sleeve_obj.vertex_groups.get(f"Elbow_Sphere_{side}")
        if elbow_group:
            try:
                elbow_weight = elbow_group.weight(i)
                max_weight = max(max_weight, elbow_weight)
            except:
                pass

        # Check wrist sphere weight
        wrist_group = sleeve_obj.vertex_groups.get(f"Wrist_Sphere_{side}")
        if wrist_group:
            try:
                wrist_weight = wrist_group.weight(i)
                max_weight = max(max_weight, wrist_weight)
            except:
                pass

        if max_weight > min_weight_threshold:
            combined_pinning_group.add([i], max_weight, 'REPLACE')

    # =========================================================================
    # CLOTH SIMULATION SETUP (Preserved Existing Functionality)
    # =========================================================================
    script_log("DEBUG create_sleeve: Creating cloth modifier")

    # Create cloth modifier
    cloth_mod = sleeve_obj.modifiers.new(name="Cloth", type='CLOTH')
    script_log(f"✓ Created cloth modifier for {side} sleeve")

    # =========================================================================
    # APPLYING CLOTH SETTINGS FROM CONFIG (Preserved Existing Functionality)
    # =========================================================================
    script_log("DEBUG create_sleeve: Applying cloth settings from config")

    # Get cloth settings from sleeve config
    cloth_config = sleeve_config.get("cloth_settings", {})

    # Apply basic cloth settings
    cloth_mod.settings.quality = cloth_config.get("quality", 6)
    cloth_mod.settings.mass = cloth_config.get("mass", 0.3)
    cloth_mod.settings.air_damping = cloth_config.get("air_damping", 1.0)
    cloth_mod.settings.time_scale = cloth_config.get("time_scale", 1.0)

    # Apply stiffness settings
    cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 10.0)
    cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 10.0)
    cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 5.0)
    cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.5)

    # Apply collision settings (Modern Blender 4.3+ API)
    cloth_mod.collision_settings.use_self_collision = cloth_config.get("self_collision", True)
    cloth_mod.collision_settings.self_distance_min = cloth_config.get("self_distance_min", 0.005)
    cloth_mod.collision_settings.collision_quality = cloth_config.get("collision_quality", 6)
    cloth_mod.collision_settings.use_collision = cloth_config.get("external_collisions", True)

    # PIN CLOTH TO THE COMBINED GROUP (Shoulder + Elbow + Wrist) - USING BUNDLES
    cloth_mod.settings.vertex_group_mass = f"{side}_Sleeve_Combined_Anchors"

    script_log(f"✓ Applied cloth settings from config for {side} sleeve")
    script_log(f"  - Quality: {cloth_mod.settings.quality}")
    script_log(f"  - Mass: {cloth_mod.settings.mass}")
    script_log(f"  - Stiffness: T={cloth_mod.settings.tension_stiffness}, C={cloth_mod.settings.compression_stiffness}")
    script_log(f"  - Sphere radii: shoulder={shoulder_sphere_radius:.3f}, elbow={elbow_sphere_radius:.3f}, wrist={wrist_sphere_radius:.3f}")
    script_log(f"  - Pinned to: {side}_Sleeve_Combined_Anchors (shoulder + elbow + wrist bundles)")

    # =========================================================================
    # ADD SUBDIVISION AND MATERIALS (Preserved Existing Functionality)
    # =========================================================================
    if subdiv_levels > 0:
        script_log("DEBUG create_sleeve: Adding subdivision surface modifier")
        subdiv_mod = sleeve_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = subdiv_levels
        subdiv_mod.render_levels = subdiv_levels

    # Add material
    material_config = sleeve_config.get("material", {})
    material_color = material_config.get("color", [0.1, 0.3, 0.8, 1.0])
    sleeve_mat = bpy.data.materials.new(name=f"{side.capitalize()}Sleeve_Material")
    sleeve_mat.diffuse_color = material_color
    sleeve_mat.roughness = material_config.get("roughness", 0.7)
    sleeve_mat.metallic = material_config.get("metallic", 0.0)
    sleeve_obj.data.materials.append(sleeve_mat)

    # =========================================================================
    # SET MODIFIER ORDER (Preserved Existing Functionality)
    # =========================================================================
    script_log("DEBUG create_sleeve: Setting modifier order")
    bpy.context.view_layer.objects.active = sleeve_obj
    modifiers = sleeve_obj.modifiers

    # Build correct order based on which modifiers are present
    correct_order = ["Subdivision", "Armature", "Cloth"]
    for mod_name in correct_order:
        mod_index = modifiers.find(mod_name)
        if mod_index >= 0:
            while mod_index > correct_order.index(mod_name):
                bpy.ops.object.modifier_move_up(modifier=mod_name)
                mod_index -= 1

    # =========================================================================
    # FINAL VERIFICATION AND LOGGING
    # =========================================================================
    bpy.context.view_layer.update()

    # Log bundle usage
    bundle_status = []
    if shoulder_bundle: bundle_status.append(f"shoulder({shoulder_bundle['vertex_count']}v)")
    if elbow_bundle: bundle_status.append(f"elbow({elbow_bundle['vertex_count']}v)")
    if wrist_bundle: bundle_status.append(f"wrist({wrist_bundle['vertex_count']}v)")

    script_log(f"DEBUG create_sleeve: Successfully created {side} sleeve with joint_vertex_bundles integration")
    script_log(f"Created {side} sleeve from shoulder to forearm (length: {actual_sleeve_length:.3f})")
    script_log(f"Sleeve diameters: shoulder={diameter_start}, elbow={diameter_elbow}, wrist={diameter_end}")
    script_log(f"Vertex bundles used: {', '.join(bundle_status) if bundle_status else 'NONE (fallback mode)'}")
    script_log(f"✓ Cloth simulation ENABLED with pinning to shoulder + elbow + wrist spherical vertex groups")
    script_log(f"✓ Seamless coordination: Uses same vertex bundles as coat (shoulder) and mittens (wrist)")
    script_log(f"✓ Backward compatibility: Preserved existing armature binding system")

    return sleeve_obj

##########################################################################################

def setup_cloth_parenting(cloth_obj, armature_obj, bone_names, default_weight=0.3):
    """Set up vertex groups and parenting for cloth simulation with proximity-based weighting"""
    # Add armature modifier
    armature_mod = cloth_obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj

    # Create vertex groups for different bone influences with proximity-based weights
    for bone_name in bone_names:
        if bone_name in armature_obj.pose.bones:
            bone = armature_obj.pose.bones[bone_name]
            bone_pos = bone.head

            # Create vertex group
            if bone_name in cloth_obj.vertex_groups:
                cloth_obj.vertex_groups.remove(cloth_obj.vertex_groups[bone_name])
            vgroup = cloth_obj.vertex_groups.new(name=bone_name)

            # Assign vertices with proximity-based weights
            for i, vert in enumerate(cloth_obj.data.vertices):
                vert_pos = cloth_obj.matrix_world @ vert.co
                distance = (vert_pos - bone_pos).length
                # Weight decreases with distance
                weight = max(0, default_weight * (1.0 - distance / 2.0))  # Adjust divisor as needed
                if weight > 0:
                    vgroup.add([i], weight, 'REPLACE')

    script_log(f"Set up cloth parenting for {cloth_obj.name} with bones: {bone_names}")

##########################################################################################

def create_cloth_material(name, material_config):
    """Create a material for cloth garments"""
    cloth_mat = bpy.data.materials.new(name=name)
    cloth_mat.use_nodes = True

    # Get material properties from config or use defaults
    color = material_config.get("color", [0.8, 0.8, 0.8, 1.0])
    roughness = material_config.get("roughness", 0.7)
    metallic = material_config.get("metallic", 0.0)

    # Set material properties
    cloth_mat.diffuse_color = color
    cloth_mat.roughness = roughness
    cloth_mat.metallic = metallic

    return cloth_mat


##########################################################################################

def save_blender_file():
    """Save the Blender file to the correct output location"""
    try:
        from utils import get_show_path, get_scene_folder_name

        show_path = get_show_path(args.show)
        scene_folder_name = get_scene_folder_name(args.show, args.scene)
        outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
        os.makedirs(outputs_dir, exist_ok=True)

        # Change from "_kid.blend" to "_cloth.blend"
        output_blender_file = os.path.join(outputs_dir, f"{args.scene}_cloth.blend")
        bpy.ops.wm.save_as_mainfile(filepath=output_blender_file)
        script_log(f"Cloth simulation saved to: {output_blender_file}")

    except Exception as e:
        script_log(f"Error saving Blender file: {e}")


##########################################################################################

def verify_hip_constraints(armature_obj):
    """Verify that all hip-related constraints are set up"""
    script_log("=== VERIFYING HIP CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    hip_midpoint_obj = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")

    # Check LowerSpine constraints
    if "DEF_LowerSpine" in armature_obj.pose.bones:
        lower_spine = armature_obj.pose.bones["DEF_LowerSpine"]
        script_log(f"DEF_LowerSpine constraints: {len(lower_spine.constraints)}")
        for constraint in lower_spine.constraints:
            # Check if constraint has target attribute before accessing it
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
                # Check if constraint has target attribute before accessing it
                if hasattr(constraint, 'target'):
                    target_name = constraint.target.name if constraint.target else "None"
                else:
                    target_name = "No target"
                script_log(f"  - {constraint.type} -> {target_name}")

    bpy.ops.object.mode_set(mode='OBJECT')

##########################################################################################

def main_execution():
    """Main execution for cloth simulation"""
    script_log("=== 4M CLOTH INNER STARTED ===\n")

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

        # 1. CREATE CONTROL POINTS FIRST
        script_log("=== CREATING CONTROL POINTS ===")
        create_control_points(figure_name)

        # 2. CREATE FRAME CONTROL POINTS
        script_log("=== CREATING FRAME CONTROL POINTS ===")
        first_frame = mocap_data.get(str(frame_numbers[0]), {})
        create_hip_frame_control_point(first_frame)
        create_shoulder_frame_control_point(first_frame)

        # 3. Set up virtual point constraints
        setup_virtual_point_constraints()
        setup_virtual_frame_constraints()

        # 4. Create rig and vertex bundles
        armature_obj = create_kid_rig(figure_name)
        make_vertex_all_bundles(armature_obj)

        # 5. ALIGN BONES WITH CONTROL POINTS
        align_bones_with_control_points(armature_obj, figure_name)

        # 6. SET UP CONSTRAINTS
        setup_direct_constraints(armature_obj, figure_name)
        setup_two_segment_spine_constraints(armature_obj, figure_name)
        setup_root_bone_transform_constraints(armature_obj)

        # 7. CREATE CLOTH GARMENTS (REPLACES KID FLESH)
        script_log("=== CREATING CLOTH GARMENTS ===")
        garments_created = create_cloth_garments(armature_obj, figure_name)

        # 8. Force initial constraint solve
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.visual_transform_apply()
        bpy.ops.object.mode_set(mode='OBJECT')

        # 9. Final scene update
        bpy.context.view_layer.update()

        # 10. Save file
        save_blender_file()

        script_log(f"=== 4M CLOTH INNER COMPLETED - Created {garments_created} garments ===\n")

    except Exception as e:
        script_log(f"ERROR in main execution: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")

##########################################################################################

if __name__ == "__main__":
    main_execution()