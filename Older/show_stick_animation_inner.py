# show_stick_animation_inner.py (Version 2.15 with array rotation support)

import bpy
import os
import sys
from mathutils import Vector, Euler, Quaternion, Matrix
import json
from datetime import datetime
import math

CONFIG_FILE = "show_stick_animation_config.json"


def load_config():
    """Load configuration from JSON file"""
    with open(CONFIG_FILE, 'r') as config_file:
        config = json.load(config_file)
    cylinder_radius = config.get("cylinder_settings", {}).get("radius", 0.05)
    cylinder_overlap = config.get("cylinder_settings", {}).get("overlap_epsilon", 0.02)

    file_settings = config.get("file_settings", {})
    show_video = file_settings.get("show_video", False)
    video_file_name = file_settings.get("video_file_name", "")
    video_plane = file_settings.get("video_plane", "XY")
    video_size = file_settings.get("video_size", 1.6)
    video_rotation = file_settings.get("video_rotation", 0)

    # Handle both single number and array format for video_rotation
    if isinstance(video_rotation, list) and len(video_rotation) == 3:
        video_rotation_x, video_rotation_y, video_rotation_z = video_rotation
    else:
        # For backward compatibility with single number format
        video_rotation_x = 0
        video_rotation_y = 0
        video_rotation_z = video_rotation

    return (
        cylinder_radius,
        cylinder_overlap,
        config["bone_definitions"],
        show_video,
        video_file_name,
        video_plane,
        video_size,
        video_rotation_x,
        video_rotation_y,
        video_rotation_z
    )


def hex_to_rgb(hex_color):
    """Convert hex color string to RGBA tuple (0-1 range)"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)) + (1.0,)


def create_material(name, color):
    """Create a new material with given name and color"""
    material = bpy.data.materials.new(name=name)
    material.diffuse_color = color
    return material


def main():
    # Get the JSON file path from command line arguments
    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1:]  # get all args after "--"

    print("INNER: Start", flush=True)
    print("", flush=True)

    if not argv:
        print("INNER: Error: No JSON file path provided", flush=True)
        return

    json_file_path = argv[0]
    output_blend_file_path = bpy.data.filepath

    if not os.path.exists(json_file_path):
        print(f"INNER: Error: JSON file not found at '{json_file_path}'", flush=True)
        return

    try:
        with open(json_file_path, 'r') as json_file:
            all_frames_data = json.load(json_file)
    except Exception as error:
        print(f"INNER: Error loading JSON data: {error}", flush=True)
        return

    if not all_frames_data:
        print("INNER: No frame data found in the JSON file.", flush=True)
        return

    # Get frame numbers early
    frame_numbers = sorted([int(frame) for frame in all_frames_data.keys()])
    if not frame_numbers:
        print("INNER: No frames to animate.", flush=True)
        return

    # Load configuration
    cylinder_radius, cylinder_overlap, bone_definitions, show_video, video_file_name, video_plane, video_size, video_rotation_x, video_rotation_y, video_rotation_z = load_config()

    # Create armature and put it in Edit Mode
    bpy.ops.object.add(type='ARMATURE', enter_editmode=True, location=(0, 0, 0))
    armature_object = bpy.context.object
    armature_object.name = "StickFigure"
    armature_data = armature_object.data
    armature_data.name = "StickFigureData"

    # Create bones in Edit Mode
    # The armature is already in edit mode from its creation
    edit_bones = armature_data.edit_bones

    for bone_name, properties in bone_definitions.items():
        new_bone = edit_bones.new(bone_name)
        new_bone.head = (0, 0, 0)
        new_bone.tail = (0, 0.1, 0)  # Small default length
        if properties["parent"] and properties["parent"] in edit_bones:
            new_bone.parent = edit_bones[properties["parent"]]

    # Switch to Object Mode to prepare for cylinder creation and video plane
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create video plane if show_video is true and a video file is specified
    video_image = None
    image_texture_node = None
    if show_video and video_file_name:
        print(f"INNER: Creating video plane for '{video_file_name}' on {video_plane} plane.")
        try:
            # Create a plane
            bpy.ops.mesh.primitive_plane_add(size=video_size, enter_editmode=False, location=(0, 0, 0))
            video_plane_obj = bpy.context.object
            video_plane_obj.name = "VideoPlane"

            # Set plane orientation and rotation
            if video_plane == "XY":
                video_plane_obj.rotation_euler = (math.radians(video_rotation_x), math.radians(video_rotation_y),
                                                  math.radians(video_rotation_z))
            elif video_plane == "XZ":
                video_plane_obj.rotation_euler = (math.pi / 2 + math.radians(video_rotation_x),
                                                  math.radians(video_rotation_y), math.radians(video_rotation_z))
            elif video_plane == "YZ":
                video_plane_obj.rotation_euler = (math.radians(video_rotation_x),
                                                  math.pi / 2 + math.radians(video_rotation_y),
                                                  math.radians(video_rotation_z))
            else:
                print(f"INNER: Warning: Unknown video plane '{video_plane}'. Defaulting to XY.")
                video_plane_obj.rotation_euler = (math.radians(video_rotation_x), math.radians(video_rotation_y),
                                                  math.radians(video_rotation_z))

            # Create a material for the video texture
            video_material = bpy.data.materials.new(name="VideoMaterial")
            video_material.use_nodes = True

            # Clear default nodes
            video_material.node_tree.nodes.clear()

            # Create nodes
            image_texture_node = video_material.node_tree.nodes.new(type='ShaderNodeTexImage')
            mapping_node = video_material.node_tree.nodes.new(type='ShaderNodeMapping')
            emission_node = video_material.node_tree.nodes.new(type='ShaderNodeEmission')
            output_node = video_material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')

            # Position nodes
            image_texture_node.location = (-500, 0)
            mapping_node.location = (-300, 0)
            emission_node.location = (-100, 0)
            output_node.location = (100, 0)

            # Load video
            video_path = os.path.join(os.path.dirname(output_blend_file_path), video_file_name)
            video_image = bpy.data.images.load(video_path)
            video_image.source = 'MOVIE'
            image_texture_node.image = video_image

            # Set up animation
            image_texture_node.image_user.frame_duration = len(all_frames_data)
            image_texture_node.image_user.use_auto_refresh = True

            # Rotate the video texture using the config values
            mapping_node.inputs['Rotation'].default_value = (math.radians(video_rotation_x),
                                                             math.radians(video_rotation_y),
                                                             math.radians(video_rotation_z))

            # Link nodes
            video_material.node_tree.links.new(
                image_texture_node.outputs['Color'],
                mapping_node.inputs['Vector']
            )
            video_material.node_tree.links.new(
                mapping_node.outputs['Vector'],
                emission_node.inputs['Color']
            )
            video_material.node_tree.links.new(
                emission_node.outputs['Emission'],
                output_node.inputs['Surface']
            )

            # Make sure the material uses alpha blend
            video_material.blend_method = 'BLEND'

            # Assign the material to the plane
            if video_plane_obj.data.materials:
                video_plane_obj.data.materials[0] = video_material
            else:
                video_plane_obj.data.materials.append(video_material)

        except Exception as e:
            print(f"INNER: Error creating video plane: {e}", flush=True)

    # Re-select the armature and go back into Edit Mode
    # This is a critical step because the plane creation switched us to Object Mode
    bpy.context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode='EDIT')

    # Hide the armature (keep cylinders visible)
    armature_object.hide_set(True)
    armature_object.hide_render = True

    # Switch back to Object Mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Create materials
    white_material = create_material("WhiteStick", (1.0, 1.0, 1.0, 1.0))
    gray_text_material = create_material("GrayText", (0.5, 0.5, 0.5, 1.0))

    # Create cylinders for bones
    cylinder_objects = {}
    pose_bones = armature_object.pose.bones

    for bone_name, properties in bone_definitions.items():
        bone = pose_bones.get(bone_name)
        if not bone:
            continue

        # Get positions from first frame
        first_frame = all_frames_data[str(frame_numbers[0])]
        start_pos = first_frame.get(properties["start_mp"])
        end_pos = first_frame.get(properties["end_mp"])

        if not start_pos or not end_pos:
            continue

        start_vec = Vector((start_pos["x"], start_pos["y"], start_pos["z"]))
        end_vec = Vector((end_pos["x"], end_pos["y"], end_pos["z"]))

        # Calculate direction and length
        direction = end_vec - start_vec
        length = direction.length

        # Skip if points are too close
        if length < 0.001:
            continue

        # Create cylinder at origin first
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=8,
            radius=cylinder_radius,
            depth=1.0,  # Default length, we'll scale it
            enter_editmode=False,
            location=(0, 0, 0)  # Create at origin first
        )

        cylinder_object = bpy.context.object
        cylinder_object.name = f"Cylinder_{bone_name}"
        cylinder_object.rotation_mode = 'QUATERNION'

        # Calculate rotation to align with direction
        up = Vector((0, 0, 1))  # Default cylinder orientation in Blender is along Z-axis
        if direction.length > 0:
            rot_quat = up.rotation_difference(direction.normalized())
        else:
            rot_quat = Quaternion()  # identity rotation

        # Calculate midpoint
        midpoint = (start_vec + end_vec) / 2

        # Set transform
        cylinder_object.location = midpoint
        cylinder_object.rotation_quaternion = rot_quat
        cylinder_object.scale = (1, 1, length)  # Scale along Z-axis to get correct length

        # Apply bone color
        color_rgb = hex_to_rgb(properties["color"])
        material = create_material(f"Material_{bone_name}", color_rgb)
        cylinder_object.data.materials.append(material)

        # Parent to armature
        cylinder_object.parent = armature_object
        cylinder_objects[bone_name] = cylinder_object

    # Animate cylinders over all frames
    for frame_number in frame_numbers:
        frame_data = all_frames_data[str(frame_number)]
        bpy.context.scene.frame_set(frame_number)

        # Manually keyframe the video image's frame property
        if show_video and video_image:
            # Find the image texture node in the material
            for mat in bpy.data.materials:
                if mat.name == "VideoMaterial" and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'TEX_IMAGE' and node.image == video_image:
                            node.image_user.frame_current = frame_number
                            node.image_user.keyframe_insert(data_path="frame_current", frame=frame_number)
                            break

        print(f"INNER: Animating frame={frame_number}")
        for bone_name, properties in bone_definitions.items():
            cylinder = cylinder_objects.get(bone_name)
            if not cylinder:
                continue

            start_pos = frame_data.get(properties["start_mp"])
            end_pos = frame_data.get(properties["end_mp"])

            if not start_pos or not end_pos:
                continue

            start_vec = Vector((start_pos["x"], start_pos["y"], start_pos["z"]))
            end_vec = Vector((end_pos["x"], end_pos["y"], end_pos["z"]))
            direction = end_vec - start_vec
            length = direction.length

            if length < 0.001:
                continue

            midpoint = (start_vec + end_vec) / 2
            up = Vector((0, 0, 1))  # Default cylinder orientation
            rotation = up.rotation_difference(direction.normalized())

            # Apply animated transform
            cylinder.location = midpoint
            cylinder.rotation_mode = 'QUATERNION'
            cylinder.rotation_quaternion = rotation
            cylinder.scale = (1, 1, length)  # Only scale Z-axis for length

            # Insert keyframes
            cylinder.keyframe_insert(data_path="location", frame=frame_number)
            cylinder.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)
            cylinder.keyframe_insert(data_path="scale", frame=frame_number)

    # Set animation range
    bpy.context.scene.frame_start = frame_numbers[0]
    bpy.context.scene.frame_end = frame_numbers[-1]

    # Save the file
    bpy.ops.wm.save_mainfile()
    print(f"INNER: [{datetime.now().strftime('%H:%M:%S')}] Animation saved to '{output_blend_file_path}'.")


if __name__ == "__main__":
    main()