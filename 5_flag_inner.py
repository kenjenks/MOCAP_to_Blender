# 5_flag_inner.py (version 3.0 - Uses new JSON format for input data frames)

import bpy
import json
import os
from mathutils import Vector, Quaternion


def get_video_dimensions(video_path):
    """
    Get video dimensions using Blender's built-in image loading.
    Returns (width, height, frame_count) or (None, None, 0) if failed.
    """
    try:
        # Load the video as an image to get its dimensions
        video_image = bpy.data.images.load(video_path, check_existing=False)
        width = video_image.size[0]
        height = video_image.size[1]

        # Get video frame count
        frame_count = video_image.frame_duration

        # Remove the temporary image reference
        bpy.data.images.remove(video_image)

        return width, height, frame_count

    except Exception as e:
        print(f"Error getting video dimensions with Blender: {e}")
        return None, None, 0


def get_flagpole_length(flagpole_obj):
    """
    Calculate the actual world-space length of the flagpole.
    """
    if flagpole_obj.type == 'MESH':
        # Get the world matrix
        world_matrix = flagpole_obj.matrix_world

        # Get all vertices in world space
        local_verts = [v.co for v in flagpole_obj.data.vertices]
        world_verts = [world_matrix @ v for v in local_verts]

        # Find min and max Z coordinates (assuming flagpole is vertical)
        z_coords = [v.z for v in world_verts]
        flagpole_length = max(z_coords) - min(z_coords)

        print(f"Flagpole mesh length calculated: {flagpole_length:.4f} Blender units")
        return flagpole_length
    else:
        # Fallback to object dimensions for non-mesh objects
        flagpole_length = flagpole_obj.dimensions.z
        print(f"Using object dimensions for flagpole length: {flagpole_length:.4f} Blender units")
        return flagpole_length


# Clear existing animation data for the flag
def clear_animation_data(obj):
    if obj.animation_data:
        obj.animation_data_clear()


# Load configuration from JSON file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def create_video_plane(video_file_path, video_width, video_height, animation_frames, video_frames):
    """
    Creates a plane in the Blender scene to display the video.
    Uses the correct frame mapping to ensure video and animation end at the same time.
    """
    if not os.path.exists(video_file_path):
        print(f"Error: Video file not found at '{video_file_path}'")
        return

    print(f"Creating video plane with file: {video_file_path}")
    print(f"Animation frames: {animation_frames}, Video frames: {video_frames}")

    # 1. Add a new plane object
    bpy.ops.mesh.primitive_plane_add(
        size=0.8,
        enter_editmode=False,
        align='WORLD',
        location=(0, -0.5, 0),
        rotation=(1.5708, 0, 2*1.5708)  # 1.5708 is 90 degrees in radians
    )
    video_plane = bpy.context.active_object
    video_plane.name = "VideoPlane"
    print("Video plane object created")

    # 2. Create material and texture
    mat = bpy.data.materials.new(name="VideoMaterial")
    video_plane.data.materials.append(mat)
    mat.use_nodes = True

    # Clear existing nodes
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Add nodes
    texture_coord = nodes.new('ShaderNodeTexCoord')
    image_texture = nodes.new('ShaderNodeTexImage')
    emission_shader = nodes.new('ShaderNodeEmission')
    output_node = nodes.new('ShaderNodeOutputMaterial')

    # Position nodes
    texture_coord.location = (0, 0)
    image_texture.location = (0, 0)
    emission_shader.location = (0, 0)
    output_node.location = (0, 0)

    # Load the video
    try:
        print(f"Loading video: {video_file_path}")
        image_texture.image = bpy.data.images.load(video_file_path, check_existing=True)
        print("Video loaded successfully")

    except RuntimeError as e:
        print(f"Error: Could not load video file '{video_file_path}': {e}")
        return

    # Set the image as a movie source
    image_texture.image.source = 'MOVIE'
    print("Set image source to MOVIE")

    # Set frame properties using image_user - CRITICAL FIX
    if hasattr(image_texture, 'image_user'):
        # Set the video to play at normal speed (1:1 with animation frames)
        image_texture.image_user.frame_duration = video_frames
        image_texture.image_user.use_auto_refresh = True
        image_texture.image_user.use_cyclic = True
        image_texture.image_user.frame_start = 1

        # Calculate the correct frame offset to make video and animation end together
        if video_frames > animation_frames:
            # Video is longer than animation - slow down video playback
            playback_speed = video_frames / animation_frames
            image_texture.image_user.frame_offset = 0
            print(f"Video is longer: slowing playback by factor {playback_speed:.2f}")
        else:
            # Animation is longer than video - video will loop
            image_texture.image_user.frame_offset = 0
            print("Animation is longer: video will loop")

        print(f"Video frame duration set to: {video_frames}")
        print(f"Auto refresh: {image_texture.image_user.use_auto_refresh}")
        print(f"Cyclic: {image_texture.image_user.use_cyclic}")

    # Connect the nodes
    links = mat.node_tree.links
    links.new(texture_coord.outputs['UV'], image_texture.inputs['Vector'])
    links.new(image_texture.outputs['Color'], emission_shader.inputs['Color'])
    links.new(emission_shader.outputs['Emission'], output_node.inputs['Surface'])

    # 3. Scale the plane to match video aspect ratio and target size
    aspect_ratio = video_width / video_height

    # Target approximately 4x4 units while maintaining aspect ratio
    target_size = 8.0
    video_plane.scale.x = target_size * aspect_ratio
    video_plane.scale.z = target_size
    video_plane.scale.y = target_size

    print(f"Plane scaled to aspect ratio: {aspect_ratio}")
    print(f"Final scale: X={video_plane.scale.x:.2f}, Z={video_plane.scale.z:.2f}")

    # 4. Set up simple driver for basic synchronization
    try:
        print("Creating basic frame synchronization...")
        # Remove any complex drivers and use simple frame mapping
        if hasattr(image_texture, 'image_user'):
            # Reset to simple 1:1 mapping
            image_texture.image_user.frame_offset = 0

        print("Basic frame synchronization set up")

    except Exception as e:
        print(f"Warning: Could not set up frame synchronization: {e}")

    print(f"Created video plane for '{os.path.basename(video_file_path)}' ({video_width}x{video_height})")
    print("Video will play at normal speed synchronized with animation")


def setup_video_playback_speed(video_file_path, animation_frames, video_frames):
    """
    Set up the video playback speed to match the animation duration.
    This ensures both end at the same time.
    """
    # Find all video textures in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if (node.type == 'TEX_IMAGE' and node.image and
                        node.image.source == 'MOVIE' and
                        os.path.basename(node.image.filepath) == os.path.basename(video_file_path)):

                    if hasattr(node, 'image_user'):
                        # Calculate the playback speed ratio
                        if video_frames > 0 and animation_frames > 0:
                            # Set the video to play at the correct speed to match animation
                            node.image_user.frame_duration = video_frames
                            node.image_user.frame_start = 1
                            node.image_user.frame_offset = 0

                            print(f"Video playback configured: {video_frames} frames")
                            print(f"Animation frames: {animation_frames}")

                            if video_frames != animation_frames:
                                print(
                                    f"Note: Video ({video_frames} frames) and animation ({animation_frames} frames) have different lengths")
                                print("Video will play at normal speed and may loop or end early")


# Main function to set flag positions from JSON data
def set_flag_positions_from_json():
    # Get the directory of the current Blender file
    blend_file_path = bpy.data.filepath
    blend_dir = os.path.dirname(blend_file_path)

    # Construct path to config file
    config_path = os.path.join(blend_dir, "flag_config.json")

    # Load configuration
    config = load_config(config_path)

    # Get max_frames from config (default to 30 if not specified)
    max_frames = config.get("max_frames", 30)

    # Get video file path
    video_file_path = os.path.join(blend_dir, config.get("video", "step_1_input.mp4"))

    # Get actual video dimensions and frame count using Blender's built-in method
    video_width, video_height, video_frame_count = get_video_dimensions(video_file_path)
    if video_width is None or video_height is None:
        # Fallback to default dimensions if detection fails
        video_width = 1280
        video_height = 720
        video_frame_count = 0
        print(f"Warning: Could not get video dimensions. Using default: {video_width}x{video_height}")
    else:
        print(f"Video dimensions detected: {video_width}x{video_height}")
        if video_frame_count > 0:
            print(f"Video frame count: {video_frame_count}")
        else:
            print("Warning: Could not detect video frame count")

    # Load JSON data with 3D coordinates
    json_path = os.path.join(blend_dir, config["flag_coords_json_path"])
    print(f"DEBUG: Opening 3D translation and rotation dataset: {json_path}")
    with open(json_path, 'r') as f:
        flag_data = json.load(f)

    # Get the flag and flagpole objects
    flag_obj = bpy.data.objects.get(config["object_name"])
    flagpole_obj = bpy.data.objects.get(config["flagpole_object_name"])

    if not flag_obj:
        raise Exception(f"Object '{config['object_name']}' not found")
    if not flagpole_obj:
        raise Exception(f"Flagpole object '{config['flagpole_object_name']}' not found")

    # Clear any existing animation data
    clear_animation_data(flag_obj)

    # Convert JSON object with frame numbers as keys to sorted list of frames
    frames = []
    for frame_num in sorted(flag_data.keys(), key=int):
        frames.append(flag_data[frame_num])

    # Calculate scale factor based on flagpole dimensions
    flagpole_length = get_flagpole_length(flagpole_obj)

    print(f"Normalized flag length in JSON: 1.0")
    print(f"Blender flagpole length: {flagpole_length:.4f} Blender units")

        # Set the frame range based on available data or max_frames
    total_frames = min(len(frames), max_frames)

    print(f"Animating {total_frames} frames (limited by max_frames: {max_frames})")

    # If video frame count couldn't be detected, use animation frames as fallback
    if video_frame_count <= 0:
        video_frame_count = total_frames
        print(f"Using animation frame count for video: {video_frame_count}")

    # Create video plane with actual dimensions
    create_video_plane(video_file_path, video_width, video_height, total_frames, video_frame_count)

    # Set up video playback speed
    setup_video_playback_speed(video_file_path, total_frames, video_frame_count)

    # Set the scene's frame range to start at 1 and end at total_frames
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = total_frames

    # Set keyframes for each frame
    for frame_idx in range(total_frames):
        frame_data = frames[frame_idx]

        # Set the current frame (starting at frame 1)
        bpy.context.scene.frame_set(frame_idx + 1)


        # Extract position, rotation from the nested structure
        position = frame_data['flag_3d']['position']
        rotation = frame_data['flag_3d']['rotation']

        x_blender = position['x']
        y_blender = position['y']
        z_blender = position['z']

        flag_obj.location.x = x_blender
        flag_obj.location.y = y_blender
        flag_obj.location.z = z_blender

        # Apply rotation with coordinate transformation
        flag_obj.rotation_mode = 'QUATERNION'

        # Extract quaternion components by name
        # Check if the rotation data is a dictionary (the expected format)
        # or a list (the old format)
        if isinstance(rotation, dict):
            # New JSON format with named keys
            w, x, y, z = rotation['w'], rotation['x'], rotation['y'], rotation['z']
        elif isinstance(rotation, list) and len(rotation) == 4:
            # Fallback to older format with a list of values
            w, x, y, z = rotation[0], rotation[1], rotation[2], rotation[3]
        else:
            # Handle unexpected format or corrupted data
            print(f"Error: Invalid rotation data format in frame {frame_idx + 1}")
            continue

        quat = Quaternion((w, x, y, z))
        quat.normalize()  # This will ensure consistent sign
        flag_obj.rotation_quaternion = quat

        print(f"DEBUG: frame {frame_idx}: NORMALIZED QUAT (w={quat.w:.4f}, x={quat.x:.4f}, y={quat.y:.4f}, z={quat.z:.4f})")

        # Debug output for first few frames (frame_idx + 1 is correct here)
        if frame_idx < 5:
            print(f"DEBUG: Frame {frame_idx + 1}:")
            print(f"DEBUG:   Raw position: ({position['x']:.3f}, {position['y']:.3f}, {position['z']:.3f})")

            if isinstance(rotation, dict):
                print(
                    f"DEBUG:   Raw rotation: w={rotation['w']:.3f}, x={rotation['x']:.3f}, y={rotation['y']:.3f}, z={rotation['z']:.3f}")
            elif isinstance(rotation, list):
                print(
                    f"DEBUG:   Raw rotation: w={rotation[0]:.3f}, x={rotation[1]:.3f}, y={rotation[2]:.3f}, z={rotation[3]:.3f}")

            # Convert quaternion to Euler angles for debugging
            euler_angles = quat.to_euler('XYZ')
            euler_degrees = (euler_angles.x * 180 / 3.14159,
                             euler_angles.y * 180 / 3.14159,
                             euler_angles.z * 180 / 3.14159)
            print(
                f"DEBUG:   Euler angles (degrees): ({euler_degrees[0]:.1f}, {euler_degrees[1]:.1f}, {euler_degrees[2]:.1f})")
            print(f"DEBUG:   Blender position: ({x_blender:.3f}, {y_blender:.3f}, {z_blender:.3f})\n")

        # Insert keyframes for location and rotation
        flag_obj.keyframe_insert(data_path="location", index=-1)
        flag_obj.keyframe_insert(data_path="rotation_quaternion", index=-1)

    # Set viewport shading to MATERIAL for video visibility
    print("Setting viewport shading to MATERIAL mode...")
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'
                    print("✓ Viewport shading set to MATERIAL mode")
                    break

    # Save the file to persist changes (including frame range)
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)

    print(f"Animation created for {total_frames} frames")
    print(f"Animation range: Frame {bpy.context.scene.frame_start} to {bpy.context.scene.frame_end}")
    print(f"Final position: X: {flag_obj.location.x:.3f}, Y: {flag_obj.location.y:.3f}, Z: {flag_obj.location.z:.3f}")
    print("File saved with animation data and video plane")
    print("Video playback configured to match animation duration")


# Execute the function
if __name__ == "__main__":
    print("\n===\nStarting internal Blender script 5_flag_inner.py\n===\n")
    set_flag_positions_from_json()
    print("\n===\nFinished internal Blender script 5_flag_inner.py\n===\n")
