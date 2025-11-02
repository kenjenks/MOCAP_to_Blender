# 4S_stick_inner.py (Version 6.0 - Reborn from the ashes)

import bpy
import sys
import os
import argparse
import json
from mathutils import Vector, Quaternion, Euler
from math import radians


# Parse command line arguments
def parse_arguments():
    """Parse command line arguments passed from 4S_stick.py"""
    parser = argparse.ArgumentParser(description='4S Stick Inner Script')
    parser.add_argument('--project-root', required=True, help='Path to project root')
    parser.add_argument('--show', required=True, help='Show name')
    parser.add_argument('--scene', required=True, help='Scene name')

    # Parse only the arguments after '--'
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
    from utils import script_log, get_scene_config, get_processing_step_paths, get_scene_paths
except ImportError as e:
    script_log(f"FAILED to import utils: {e}", force_log=True)
    sys.exit(1)

# Global variables to store the loaded data and config
mocap_data = {}
bone_definitions = {}
cylinder_settings = {}
stick_figure_settings = {}
visibility_settings = {}
ghost_settings = {}
file_settings = {}
frame_numbers = []
head_settings = {}
control_objects = {}  # Dictionary to store control objects by joint name


def load_config_and_data():
    """Load configuration and mocap data using project utilities"""
    global mocap_data, bone_definitions, cylinder_settings, stick_figure_settings, ghost_settings, file_settings, frame_numbers, head_settings

    try:
        # Get scene configuration
        scene_config = get_scene_config(args.show, args.scene)

        # Get processing step paths
        step_paths = get_processing_step_paths(args.show, args.scene, "stick_animation")

        # Input JSON is from the apply_physics step (same as 4D_magic)
        processing_steps = scene_config.get("processing_steps", {})
        apply_physics_step = processing_steps.get("apply_physics", {})
        input_json_relative = apply_physics_step.get("output_file", "step_4_input.json")

        # Build absolute path to input JSON
        scene_paths = get_scene_paths(args.show, args.scene)
        inputs_dir = os.path.dirname(scene_paths["output_pose_data"])
        INPUT_JSON_FILE = os.path.join(inputs_dir, input_json_relative)

        # Load stick figure specific config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        STICK_CONFIG_FILE = os.path.join(script_dir, "4S_stick_config.json")

        with open(STICK_CONFIG_FILE, 'r') as file:
            config = json.load(file)
            file_settings = config.get("file_settings", {})
            cylinder_settings = config.get("cylinder_settings", {})
            stick_figure_settings = config.get("stick_figure_settings", {})
            head_settings = config.get("head_settings", {})
            bone_definitions = config.get("bone_definitions", {})
            ghost_settings = config.get("ghost_settings", {})

        with open(INPUT_JSON_FILE, 'r') as file:
            mocap_data = json.load(file)

        script_log(f"Loaded mocap data from: {INPUT_JSON_FILE}", force_log=True)
        script_log(f"Loaded stick config from: {STICK_CONFIG_FILE}", force_log=True)

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


def add_collection(name, parent_collection=None):
    """Add a new collection if it doesn't exist and link it to the scene."""
    if name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[name]

    if parent_collection:
        # Move collection from root to parent
        if new_collection in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(new_collection)
        if new_collection not in parent_collection.children:
            parent_collection.children.link(new_collection)

    return new_collection


def create_transparent_material(hex_color, alpha=0.3):
    """Create a transparent material from a hex color string."""
    mat_name = f"Material_{hex_color.replace('#', '')}_Alpha_{int(alpha * 100)}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    # mat.shadow_method = 'NONE' if hasattr(mat, 'shadow_method') else None

    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        principled_bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        principled_bsdf.inputs['Alpha'].default_value = alpha

    return mat


def create_material(hex_color):
    """Create a new material from a hex color string."""
    mat_name = f"Material_{hex_color.replace('#', '')}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        principled_bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
    return mat


def create_stick_figure(figure_name, color, is_ghost=False):
    """
    Creates a stick figure using cylinders for bones and spheres for joints.
    Returns a dictionary of control points and a dictionary of bones.
    """
    script_log(f"Creating stick figure: {figure_name}", force_log=True)

    bpy.ops.object.select_all(action='DESELECT')

    # Create main collections
    main_collection = add_collection(f"{figure_name}_MainFigure")
    control_collection = add_collection(f"{figure_name}_ControlPoints")
    bone_collection = add_collection(f"{figure_name}_Bones")

    # Set collection for active object using the correct LayerCollection
    layer_collection = bpy.context.view_layer.layer_collection.children.get(main_collection.name)
    if layer_collection:
        bpy.context.view_layer.active_layer_collection = layer_collection

    control_points = {}
    bones = {}

    # --- FIRST PASS: CREATE ALL CONTROL POINTS (SPHERES) ---
    for bone_name, bone_data in bone_definitions.items():
        start_mp = bone_data["start_mp"]
        end_mp = bone_data["end_mp"]

        # Only create a sphere if it doesn't already exist
        for mp in [start_mp, end_mp]:
            if mp not in control_points:
                bpy.ops.mesh.primitive_uv_sphere_add(
                    radius=head_settings.get("radius", 0.03) * 0.5,
                    location=(0, 0, 0)
                )
                control_obj = bpy.context.active_object
                control_obj.name = f"{figure_name}_{mp}"
                control_obj.hide_set(False)
                control_points[mp] = control_obj
                script_log(f"DEBUG: Control Point: {control_obj.name}, radius={head_settings.get('radius', 0.03) * 0.5}")

                # Link to control collection and unlink from main scene collection
                if control_obj.name in bpy.context.scene.collection.objects:
                    bpy.context.scene.collection.objects.unlink(control_obj)
                control_collection.objects.link(control_obj)

    # --- SECOND PASS: CREATE ALL BONES (CYLINDERS) WITH PROPER GEOMETRY ---
    for bone_name, bone_data in bone_definitions.items():
        start_mp = bone_data["start_mp"]
        end_mp = bone_data["end_mp"]

        start_control = control_points.get(start_mp)
        end_control = control_points.get(end_mp)

        if not start_control or not end_control:
            script_log(f"Warning: Missing control points for bone '{bone_name}'. Skipping.", force_log=True)
            continue

        # Get positions from first frame to calculate initial bone length
        first_frame = mocap_data.get('1', {})
        start_pos = first_frame.get(start_mp, {'x': 0, 'y': 0, 'z': 0})
        end_pos = first_frame.get(end_mp, {'x': 0.1, 'y': 0, 'z': 0})  # Default length

        start_vec = Vector((start_pos["x"], start_pos["y"], start_pos["z"]))
        end_vec = Vector((end_pos["x"], end_pos["y"], end_pos["z"]))

        # Calculate bone length and direction
        bone_vector = end_vec - start_vec
        bone_length = bone_vector.length

        if bone_length < 0.001:
            script_log(f"Warning: Bone '{bone_name}' has zero length. Skipping.", force_log=True)
            continue

        # Create cylinder with proper geometry
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32,
            radius=cylinder_settings.get("radius", 0.005),
            depth=1.0,  # Unit cylinder
            location=(0, 0, 0),  # Temporary location
            rotation=(0, 0, 0)
        )
        bone_obj = bpy.context.active_object
        bone_obj.name = f"{figure_name}_{bone_name}"

        # Calculate the midpoint between start and end control points
        midpoint = (start_vec + end_vec) / 2

        # Position the cylinder at the midpoint
        bone_obj.location = midpoint

        # Scale the cylinder to the correct length
        bone_obj.scale = (1, 1, bone_length)

        # Orient the cylinder to point from start to end control point
        if bone_length > 0.001:
            # Calculate rotation to align cylinder with bone direction
            up_vector = Vector((0, 0, 1))  # Default cylinder orientation
            rotation_quat = up_vector.rotation_difference(bone_vector.normalized())
            bone_obj.rotation_mode = 'QUATERNION'
            bone_obj.rotation_quaternion = rotation_quat

        script_log(f"DEBUG: Bone: {bone_obj.name}, radius={cylinder_settings.get('radius', 0.005)}, depth={bone_length}")

        # Link to bone collection
        if bone_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(bone_obj)
        bone_collection.objects.link(bone_obj)

        # Apply color
        if is_ghost:
            mat = create_material(color)
            bone_obj.data.materials.clear()
            bone_obj.data.materials.append(mat)
        else:
            bone_color = bone_definitions.get(bone_name, {}).get("color", color)
            mat = create_transparent_material(bone_color, 1.0)
            bone_obj.data.materials.clear()
            bone_obj.data.materials.append(mat)
            script_log(f"DEBUG: Setting bone color for {bone_obj.name} to {bone_color}", force_log=True)

        bones[bone_name] = bone_obj

    return control_points, bones


def create_video_plane():
    """Create a plane and apply the video texture to it with proper synchronization"""
    # Get video settings from scene config
    scene_config = get_scene_config(args.show, args.scene)
    video_settings = scene_config.get("video_settings", {})

    show_video = video_settings.get("show_video", file_settings.get("show_video", False))
    if not show_video:
        script_log("Video plane creation disabled in settings", force_log=True)
        return

    video_file = video_settings.get("video_file_name", file_settings.get("video_file_name"))
    if not video_file:
        script_log("Warning: No video file name specified. Skipping video plane creation.", force_log=True)
        return

    # Look for video file in the scene inputs directory
    scene_paths = get_scene_paths(args.show, args.scene)
    inputs_dir = os.path.dirname(scene_paths["output_pose_data"])
    video_file_path = os.path.join(inputs_dir, video_file)

    if not os.path.exists(video_file_path):
        script_log(f"Warning: Video file '{video_file_path}' not found. Skipping video plane creation.", force_log=True)
        return

    # Create the plane
    video_plane_type = video_settings.get("video_plane", file_settings.get("video_plane", "XZ")).upper()
    plane_location = (0, 0, 0)
    rotation_axis = (0, 0, 0)
    if video_plane_type == "XY":
        script_log("DEBUG: Create the video plane in the XY plane.", force_log=True)
        plane_location = (0, 0, 0)
        rotation_axis = (0, 0, 0)
    elif video_plane_type == "XZ":
        script_log("DEBUG: Create the video plane in the XZ plane.", force_log=True)
        plane_location = (0, -0.3, 0)
        rotation_axis = (radians(90), 0, 0)
    elif video_plane_type == "YZ":
        script_log("DEBUG: Create the video plane in the YZ plane.", force_log=True)
        plane_location = (0, 0, 0)
        rotation_axis = (0, radians(90), 0)

    bpy.ops.mesh.primitive_plane_add(
        size=video_settings.get("video_size", file_settings.get("video_size", 1.0)),
        enter_editmode=False,
        align='WORLD',
        location=plane_location,
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

    # Set frame duration using image_user (the correct way for video synchronization)
    if frame_numbers:
        image_texture.image_user.frame_duration = len(frame_numbers)
        image_texture.image_user.use_auto_refresh = True

    # Add an Emission shader to the material
    emission_shader = mat.node_tree.nodes.new('ShaderNodeEmission')

    # Connect the nodes
    links = mat.node_tree.links
    links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
    links.new(image_texture.outputs['Color'], emission_shader.inputs['Color'])
    links.new(emission_shader.outputs['Emission'], mat.node_tree.nodes['Material Output'].inputs['Surface'])

    video_plane.location = (0, 0, 0)

    script_log("Video plane created and configured with proper frame synchronization.", force_log=True)


def animate_figures(ghost_figures, control_points, bones):
    """
    Animate all figures, updating control points and bones.
    """
    if not frame_numbers:
        script_log("No frame data to animate.", force_log=True)
        return

    script_log(f"Starting animation for {len(frame_numbers)} frames...", force_log=True)

    # Animate control points for the main figure
    for frame_number in frame_numbers:
        frame_data = mocap_data[str(frame_number)]
        bpy.context.scene.frame_set(frame_number)

        # Update control point positions and keyframe them
        for joint_name, control_obj in control_points.items():
            if joint_name in frame_data:
                joint_data = frame_data[joint_name]
                control_obj.location = Vector((joint_data["x"], joint_data["y"], joint_data["z"]))
                control_obj.keyframe_insert(data_path="location", frame=frame_number)

        # Update bone positions and orientations based on control points
        for bone_name, bone_data in bone_definitions.items():
            bone_obj = bones.get(bone_name)
            start_mp = bone_data["start_mp"]
            end_mp = bone_data["end_mp"]

            if not bone_obj or start_mp not in frame_data or end_mp not in frame_data:
                continue

            start_data = frame_data[start_mp]
            end_data = frame_data[end_mp]

            start_vec = Vector((start_data["x"], start_data["y"], start_data["z"]))
            end_vec = Vector((end_data["x"], end_data["y"], end_data["z"]))

            bone_vector = end_vec - start_vec
            bone_length = bone_vector.length

            if bone_length > 0.001:
                # Calculate the midpoint between start and end control points
                midpoint = (start_vec + end_vec) / 2

                # Update bone position to midpoint
                bone_obj.location = midpoint

                # Calculate rotation to align cylinder with bone direction
                up_vector = Vector((0, 0, 1))  # Default cylinder orientation
                rotation_quat = up_vector.rotation_difference(bone_vector.normalized())
                bone_obj.rotation_mode = 'QUATERNION'
                bone_obj.rotation_quaternion = rotation_quat  # ← THIS LINE WAS MISSING

                # Update scale for length
                bone_obj.scale = (1, 1, bone_length)

                # Keyframe bone transformations
                bone_obj.keyframe_insert(data_path="location", frame=frame_number)
                bone_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
                bone_obj.keyframe_insert(data_path="scale", frame=frame_number)

    # Set animation range
    bpy.context.scene.frame_start = frame_numbers[0]
    bpy.context.scene.frame_end = frame_numbers[-1]
    script_log(f"Bone animation range set from frame {frame_numbers[0]} to {frame_numbers[-1]}.", force_log=True)


def save_blender_file():
    """Save the Blender file to the correct output location"""
    try:
        from utils import get_show_path, get_scene_folder_name

        show_path = get_show_path(args.show)
        scene_folder_name = get_scene_folder_name(args.show, args.scene)
        outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
        os.makedirs(outputs_dir, exist_ok=True)

        output_blend_file = os.path.join(outputs_dir, f"{args.scene}_stick.blend")
        bpy.ops.wm.save_as_mainfile(filepath=output_blend_file)
        script_log(f"Stick figure animation saved to: {output_blend_file}", force_log=True)

    except Exception as e:
        script_log(f"Error saving Blender file: {e}", force_log=True)


# MAIN EXECUTION
script_log("=== 4S STICK INNER PHASE 2 STARTED ===", force_log=True)
script_log(f"Show: {args.show}, Scene: {args.scene}", force_log=True)

try:
    # Load configuration and data
    load_config_and_data()
    script_log("SUCCESS: Loaded config and data", force_log=True)

    # Clear existing objects if they exist
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    #########################################################################################################
    # Create main stick figure
    #########################################################################################################

    figure_color = stick_figure_settings.get("color", "#ffffff")
    control_points, main_bones = create_stick_figure("Main", figure_color)

    for bone_name, bone_obj in main_bones.items():
        bone_color = bone_definitions.get(bone_name, {}).get("color", figure_color)
        mat = create_transparent_material(bone_color, 1.0)  # Not transparent at all
        bone_obj.data.materials.clear()
        bone_obj.data.materials.append(mat)
        script_log(f"DEBUG: bone_name: {bone_name}, bone_color: {bone_color}", force_log=True)

    script_log(f"DEBUG: Number of bone objects created: {len(main_bones)}", force_log=True)
    for bone_name, bone_obj in main_bones.items():
        script_log(f"DEBUG: Bone '{bone_name}' - Location: {bone_obj.location}, Scale: {bone_obj.scale}", force_log=True)
        script_log(f"DEBUG: Bone '{bone_name}' - Visible: {not bone_obj.hide_get()}", force_log=True)
        script_log(f"DEBUG: Bone '{bone_name}' - Collection: {bone_obj.users_collection[0].name if bone_obj.users_collection else 'None'}", force_log=True)
        script_log(f"DEBUG: Bone '{bone_name}' - Material: {len(bone_obj.data.materials)} materials", force_log=True)

        # Force make visible
        bone_obj.hide_set(False)
        bone_obj.hide_viewport = False
        bone_obj.hide_render = False

    script_log("Created main stick figure", force_log=True)

    # Create ghost figures if enabled
    ghost_figures = {"previous": [], "future": []}
    if ghost_settings.get("show_ghosts", False):
        ghost_steps = ghost_settings.get("ghost_steps", 10)
        num_ghosts = ghost_settings.get("num_ghosts", 3)
        previous_colors = ghost_settings.get("previous_ghost_colors", [])
        future_colors = ghost_settings.get("future_ghost_colors", [])
        previous_trans = ghost_settings.get("previous_ghost_trans", 0.05)
        future_trans = ghost_settings.get("future_ghost_trans", 0.05)

        # Create "previous" ghosts - use colors from previous_ghost_colors
        for i in range(min(num_ghosts, len(previous_colors))):
            ghost_color = previous_colors[i]
            ghost_control_points, ghost_bones = create_stick_figure(f"GhostPrevious_{i + 1}", ghost_color, is_ghost=True)

            # Apply transparency to all ghost bones (including head)
            for bone_name, bone_obj in ghost_bones.items():
                # Create transparent material using the ghost color
                mat = create_transparent_material(ghost_color, previous_trans)
                bone_obj.data.materials.clear()
                bone_obj.data.materials.append(mat)

            ghost_figures["previous"].append((ghost_control_points, ghost_bones))

        # Create "future" ghosts - use colors from future_ghost_colors
        for i in range(min(num_ghosts, len(future_colors))):
            ghost_color = future_colors[i]
            ghost_control_points, ghost_bones = create_stick_figure(f"GhostFuture_{i + 1}", ghost_color, is_ghost=True)

            # Apply transparency to all ghost bones (including head)
            for bone_name, bone_obj in ghost_bones.items():
                # Create transparent material using the ghost color
                mat = create_transparent_material(ghost_color, future_trans)
                bone_obj.data.materials.clear()
                bone_obj.data.materials.append(mat)

            ghost_figures["future"].append((ghost_control_points, ghost_bones))

        script_log("Created ghost figures", force_log=True)

    # Create video plane if enabled
    create_video_plane()

    # Animate all figures
    animate_figures(ghost_figures, control_points, main_bones)

    # Save the Blender file
    save_blender_file()

    script_log("=== 4S STICK INNER PHASE 2 COMPLETED ===", force_log=True)

except Exception as e:
    script_log(f"ERROR in main execution: {e}", force_log=True)
    import traceback
    script_log(f"Traceback: {traceback.format_exc()}", force_log=True)

