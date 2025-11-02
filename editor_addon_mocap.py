# editor_addon_mocap.py (version 1.7 - Add Copy button)

bl_info = {
    "name": "Mocap Editor",
    "author": "Frederic Shenan II",
    "version": (1, 7, 0),
    "blender": (4, 1, 0),
    "location": "3D Viewport > Sidebar > Mocap",
    "description": "Adds a Blender panel to edit and clean up mocap data. Features cubic interpolation with configurable frame window.",
    "warning": "Save .blend file first and ensure config/data files are in same directory",
    "doc_url": "",
    "category": "Animation",
}

import bpy
import json
import os
from mathutils import Vector

# Global variables to hold the paths and data
CONFIG_FILE_NAME = "edit_stick_animation_config.json"
INPUT_JSON_FILE = None
MOCAP_DATA = None
CONFIG_LOADED = False
CONFIG = None

# Global variables for ghost visibility state
PAST_GHOSTS_VISIBLE = True
FUTURE_GHOSTS_VISIBLE = True


def get_file_paths():
    """
    Determines the absolute paths for the config and input JSON files
    based on the current .blend file's location.
    Returns (config_path, input_json_path) or (None, None) on error
    """
    global CONFIG

    try:
        # Safely get the filepath with proper error handling
        blend_filepath = bpy.data.filepath
        if not blend_filepath:
            return None, None, "Please save your .blend file first to define the project directory."

        # Get the directory of the current .blend file
        blend_dir = os.path.dirname(blend_filepath)

        # Construct the path to the config file
        config_file_path = os.path.join(blend_dir, CONFIG_FILE_NAME)

        if not os.path.exists(config_file_path):
            return None, None, f"Config file not found: {config_file_path}"

        # Load the config file to find the input JSON file name
        try:
            with open(config_file_path, 'r') as file:
                CONFIG = json.load(file)
                input_json_name = CONFIG.get("file_settings", {}).get("input_json")
                if not input_json_name:
                    return None, None, "Missing 'input_json' in 'file_settings' of the config file."
        except Exception as e:
            return None, None, f"Error loading config file: {e}"

        # Construct the full path to the input JSON file
        input_json_path = os.path.join(blend_dir, input_json_name)

        if not os.path.exists(input_json_path):
            return None, None, f"Input JSON file not found: {input_json_path}"

        return config_file_path, input_json_path, None

    except AttributeError:
        # Handle the case where bpy.data.filepath is not accessible
        return None, None, "File path not accessible in current context. Please save .blend file."
    except Exception as e:
        return None, None, f"Error getting file paths: {e}"


def get_interpolation_frames():
    """Get the number of interpolation frames from config"""
    global CONFIG
    if CONFIG:
        return CONFIG.get("edit_add_on_settings", {}).get("num_interpolation_frames", 5)
    return 5  # Default value


def cubic_interpolate_xyz(points, target_frame):
    """
    Perform cubic interpolation on XYZ data using Catmull–Rom splines.
    points: list of tuples (frame, x, y, z), must have >= 4 points
    target_frame: frame to interpolate to
    """
    if len(points) < 4:
        # Fall back to linear interpolation if not enough points
        return linear_interpolate_xyz(points, target_frame)

    # Sort points by frame
    points = sorted(points, key=lambda p: p[0])
    frames = [p[0] for p in points]

    # Find segment containing target_frame
    if target_frame <= frames[0]:
        return points[0][1], points[0][2], points[0][3]
    if target_frame >= frames[-1]:
        return points[-1][1], points[-1][2], points[-1][3]

    # Find indices
    for i in range(1, len(frames) - 2):
        if frames[i] <= target_frame <= frames[i + 1]:
            break

    f0, f1, f2, f3 = frames[i - 1], frames[i], frames[i + 1], frames[i + 2]
    p0 = Vector(points[i - 1][1:])
    p1 = Vector(points[i][1:])
    p2 = Vector(points[i + 1][1:])
    p3 = Vector(points[i + 2][1:])

    # Normalize t between f1 and f2
    t = (target_frame - f1) / (f2 - f1)

    # Catmull-Rom basis
    def catmull_rom(p0, p1, p2, p3, t):
        return 0.5 * (
                (2 * p1) +
                (-p0 + p2) * t +
                (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t +
                (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t
        )

    result = catmull_rom(p0, p1, p2, p3, t)
    return result.x, result.y, result.z


def linear_interpolate_xyz(points, target_frame):
    """Simple linear interpolation between two nearest points"""
    points = sorted(points, key=lambda p: p[0])
    frames = [p[0] for p in points]

    if target_frame <= frames[0]:
        return points[0][1], points[0][2], points[0][3]
    if target_frame >= frames[-1]:
        return points[-1][1], points[-1][2], points[-1][3]

    for i in range(1, len(frames)):
        if frames[i - 1] <= target_frame <= frames[i]:
            f1, f2 = frames[i - 1], frames[i]
            p1, p2 = Vector(points[i - 1][1:]), Vector(points[i][1:])
            t = (target_frame - f1) / (f2 - f1)
            result = p1.lerp(p2, t)
            return result.x, result.y, result.z

    # If we get this far, just return the current frame XYZ
    return points[0][1], points[0][2], points[0][3]


def load_mocap_data():
    """
    Loads mocap data from the input JSON file.
    Returns (success, message)
    """
    global INPUT_JSON_FILE, MOCAP_DATA, CONFIG_LOADED

    # First try to get file paths if not already set
    if INPUT_JSON_FILE is None:
        config_path, input_json_path, error_msg = get_file_paths()
        if not input_json_path:
            return False, error_msg
        INPUT_JSON_FILE = input_json_path

    try:
        with open(INPUT_JSON_FILE, 'r') as json_file:
            MOCAP_DATA = json.load(json_file)
        CONFIG_LOADED = True
        return True, f"Successfully loaded mocap data from {os.path.basename(INPUT_JSON_FILE)}"
    except FileNotFoundError:
        return False, f"Mocap data file not found: {INPUT_JSON_FILE}"
    except json.JSONDecodeError:
        return False, f"Error decoding JSON from file: {INPUT_JSON_FILE}. Check file format."
    except Exception as e:
        return False, f"Error loading mocap data: {e}"


def save_mocap_data(data):
    """Saves the modified mocap data back to the input JSON file"""
    global INPUT_JSON_FILE

    if INPUT_JSON_FILE is None:
        # Try to get the path again
        _, input_json_path, error_msg = get_file_paths()
        if not input_json_path:
            return False, error_msg
        INPUT_JSON_FILE = input_json_path

    try:
        # Create backup before saving
        backup_path = INPUT_JSON_FILE + ".backup"
        if os.path.exists(INPUT_JSON_FILE):
            import shutil
            shutil.copy2(INPUT_JSON_FILE, backup_path)

        with open(INPUT_JSON_FILE, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        return True, f"Successfully saved mocap data to {os.path.basename(INPUT_JSON_FILE)}"
    except Exception as e:
        return False, f"Error saving mocap data: {e}"


def reload_mocap_data():
    """Reloads mocap data from file"""
    global MOCAP_DATA, CONFIG_OLDED
    MOCAP_DATA = None
    CONFIG_LOADED = False
    return load_mocap_data()


def is_frame_hidden(frame_data):
    """Check if a frame is hidden (confidence < 0.1)"""
    if not frame_data:
        return False

    for joint_data in frame_data.values():
        if "confidence" in joint_data and joint_data["confidence"] < 0.1:
            return True
    return False


def toggle_blender_objects_visibility(frame_number, hide):
    """Toggle visibility of Blender objects for the current frame"""
    try:
        # Look for objects that might represent the current frame
        # This assumes objects are named with frame numbers or have custom properties
        for obj in bpy.data.objects:
            # Check if object has frame number property or in name
            frame_num = obj.get("frame_number")
            if frame_num is None:
                # Try to extract from name
                try:
                    frame_num = int(obj.name.split("_")[-1])
                except:
                    continue

            if frame_num == frame_number:
                obj.hide_viewport = hide
                obj.hide_render = hide
    except Exception as e:
        print(f"Error toggling Blender object visibility: {e}")


def find_ghost_objects():
    """Find all ghost objects in the scene and separate them into past and future"""
    past_ghosts = []
    future_ghosts = []

    # Look for objects in the GhostFigures collection or with ghost names
    try:
        ghost_collection = bpy.data.collections.get("GhostFigures")
        if ghost_collection:
            for obj in ghost_collection.objects:
                if "Ghost_Past" in obj.name:
                    past_ghosts.append(obj)
                elif "Ghost_Future" in obj.name:
                    future_ghosts.append(obj)
    except AttributeError:
        # Data context is restricted, can't access collections
        return [], []

    # Also check all objects in the scene for ghost naming patterns
    for obj in bpy.data.objects:
        if obj.name not in past_ghosts and obj.name not in future_ghosts:
            if "Ghost_Past" in obj.name:
                past_ghosts.append(obj)
            elif "Ghost_Future" in obj.name:
                future_ghosts.append(obj)

    return past_ghosts, future_ghosts


def toggle_ghost_objects_visibility(ghost_objects, hide):
    """Toggle visibility of multiple ghost objects"""
    for obj in ghost_objects:
        obj.hide_viewport = hide
        obj.hide_render = hide


def update_all_ghost_visibility():
    """Update visibility of all ghosts based on global state"""
    global PAST_GHOSTS_VISIBLE, FUTURE_GHOSTS_VISIBLE

    try:
        past_ghosts, future_ghosts = find_ghost_objects()

        if past_ghosts:
            for obj in past_ghosts:
                # Remove animation data for viewport visibility
                if obj.animation_data and obj.animation_data.action:
                    for fcurve in obj.animation_data.action.fcurves:
                        if 'hide_viewport' in fcurve.data_path:
                            obj.animation_data.action.fcurves.remove(fcurve)
                obj.hide_viewport = not PAST_GHOSTS_VISIBLE

                # Remove animation data for render visibility
                if obj.animation_data and obj.animation_data.action:
                    for fcurve in obj.animation_data.action.fcurves:
                        if 'hide_render' in fcurve.data_path:
                            obj.animation_data.action.fcurves.remove(fcurve)
                obj.hide_render = not PAST_GHOSTS_VISIBLE

        if future_ghosts:
            for obj in future_ghosts:
                # Remove animation data for viewport visibility
                if obj.animation_data and obj.animation_data.action:
                    for fcurve in obj.animation_data.action.fcurves:
                        if 'hide_viewport' in fcurve.data_path:
                            obj.animation_data.action.fcurves.remove(fcurve)
                obj.hide_viewport = not FUTURE_GHOSTS_VISIBLE

                # Remove animation data for render visibility
                if obj.animation_data and obj.animation_data.action:
                    for fcurve in obj.animation_data.action.fcurves:
                        if 'hide_render' in fcurve.data_path:
                            obj.animation_data.action.fcurves.remove(fcurve)
                obj.hide_render = not FUTURE_GHOSTS_VISIBLE

    except AttributeError:
        # Data context is restricted, skip updating
        pass


@bpy.app.handlers.persistent
def update_ghosts_on_frame_change(scene):
    """Update ghost visibility when frame changes"""
    update_all_ghost_visibility()


# --- Blender Operators ---
class MOCAP_OT_reload_data(bpy.types.Operator):
    bl_idname = "mocap.reload_data"
    bl_label = "Reload Mocap Data"
    bl_description = "Reload mocap data from file"
    bl_options = {'REGISTER'}

    def execute(self, context):
        success, message = reload_mocap_data()
        if success:
            self.report({'INFO'}, message)
        else:
            self.report({'ERROR'}, message)
        return {'FINISHED'}


class MOCAP_OT_toggle_hide_frame(bpy.types.Operator):
    bl_idname = "mocap.toggle_hide_frame"
    bl_label = "Hide Current Frame"
    bl_description = "Hides or shows the current frame in both MediaPipe data and Blender objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global MOCAP_DATA

        # Ensure data is loaded
        if MOCAP_DATA is None:
            success, message = load_mocap_data()
            if not success:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}

        current_frame = context.scene.frame_current
        frame_str = str(current_frame)

        # Check if frame exists in data
        if frame_str not in MOCAP_DATA:
            self.report({'WARNING'}, f"No mocap data found for frame {current_frame}.")
            return {'CANCELLED'}

        # Determine if we're hiding or unhiding
        frame_hidden = is_frame_hidden(MOCAP_DATA[frame_str])
        new_confidence = 0.0 if not frame_hidden else 0.9

        # Update visibility for all joints in this frame
        for joint_name in MOCAP_DATA[frame_str]:
            MOCAP_DATA[frame_str][joint_name]["confidence"] = new_confidence

        # Toggle Blender object visibility
        toggle_blender_objects_visibility(current_frame, not frame_hidden)

        # Save changes
        success, message = save_mocap_data(MOCAP_DATA)
        if success:
            action = "hidden" if not frame_hidden else "un-hidden"
            self.report({'INFO'}, f"Frame {current_frame} {action}.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class MOCAP_OT_copy_previous_frame(bpy.types.Operator):
    bl_idname = "mocap.copy_previous_frame"
    bl_label = "Copy Previous Frame"
    bl_description = "Copies XYZ positions from previous frame and sets confidence to 0.90"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global MOCAP_DATA

        # Ensure data is loaded
        if MOCAP_DATA is None:
            success, message = load_mocap_data()
            if not success:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}

        current_frame = context.scene.frame_current
        previous_frame = current_frame - 1
        previous_frame_str = str(previous_frame)
        current_frame_str = str(current_frame)

        # Check if previous frame exists
        if previous_frame_str not in MOCAP_DATA:
            self.report({'WARNING'}, f"No mocap data found for previous frame {previous_frame}.")
            return {'CANCELLED'}

        # Copy data from previous frame
        MOCAP_DATA[current_frame_str] = {}
        for joint_name, joint_data in MOCAP_DATA[previous_frame_str].items():
            # Copy XYZ values and set confidence to 0.90
            MOCAP_DATA[current_frame_str][joint_name] = {
                "x": joint_data.get("x", 0),
                "y": joint_data.get("y", 0),
                "z": joint_data.get("z", 0),
                "confidence": 0.90
            }

        # Save changes
        success, message = save_mocap_data(MOCAP_DATA)
        if success:
            self.report({'INFO'}, f"Copied frame {previous_frame} to frame {current_frame} with confidence 0.90.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class MOCAP_OT_interpolate_from_past(bpy.types.Operator):
    bl_idname = "mocap.interpolate_from_past"
    bl_label = "Interpolate From Past"
    bl_description = "Interpolate current frame using past frames"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global MOCAP_DATA

        # Ensure data is loaded
        if MOCAP_DATA is None:
            success, message = load_mocap_data()
            if not success:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}

        current_frame = context.scene.frame_current
        num_frames = get_interpolation_frames()

        # Get frames for interpolation (current frame and previous frames)
        frames_to_interpolate = []
        for i in range(num_frames):
            frame_num = current_frame - i
            frame_str = str(frame_num)
            if frame_str in MOCAP_DATA and not is_frame_hidden(MOCAP_DATA[frame_str]):
                frames_to_interpolate.append(frame_num)

        if len(frames_to_interpolate) < 2:
            self.report({'WARNING'}, f"Need at least 2 visible frames to interpolate. Found {len(frames_to_interpolate)}.")
            return {'CANCELLED'}

        # For each joint, collect data points for interpolation
        joints_data = {}
        for frame_num in frames_to_interpolate:
            frame_str = str(frame_num)
            for joint_name, joint_data in MOCAP_DATA[frame_str].items():
                if joint_name not in joints_data:
                    joints_data[joint_name] = []
                joints_data[joint_name].append((
                    frame_num,
                    joint_data["x"],
                    joint_data["y"],
                    joint_data["z"]
                ))

        # Interpolate each joint
        current_frame_str = str(current_frame)
        if current_frame_str not in MOCAP_DATA:
            MOCAP_DATA[current_frame_str] = {}

        for joint_name, points in joints_data.items():
            if len(points) >= 2:  # Need at least 2 points for interpolation
                x, y, z = linear_interpolate_xyz(points, current_frame)
                MOCAP_DATA[current_frame_str][joint_name] = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": 0.9
                }

        # Save changes
        success, message = save_mocap_data(MOCAP_DATA)
        if success:
            self.report({'INFO'}, f"Interpolated frame {current_frame} using {len(frames_to_interpolate)} past frames.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class MOCAP_OT_interpolate_from_future(bpy.types.Operator):
    bl_idname = "mocap.interpolate_from_future"
    bl_label = "Interpolate From Future"
    bl_description = "Interpolate current frame using future frames"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global MOCAP_DATA

        # Ensure data is loaded
        if MOCAP_DATA is None:
            success, message = load_mocap_data()
            if not success:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}

        current_frame = context.scene.frame_current
        num_frames = get_interpolation_frames()

        # Get frames for interpolation (current frame and future frames)
        frames_to_interpolate = []
        for i in range(num_frames):
            frame_num = current_frame + i
            frame_str = str(frame_num)
            if frame_str in MOCAP_DATA and not is_frame_hidden(MOCAP_DATA[frame_str]):
                frames_to_interpolate.append(frame_num)

        if len(frames_to_interpolate) < 2:
            self.report({'WARNING'}, f"Need at least 2 visible frames to interpolate. Found {len(frames_to_interpolate)}.")
            return {'CANCELLED'}

        # For each joint, collect data points for interpolation
        joints_data = {}
        for frame_num in frames_to_interpolate:
            frame_str = str(frame_num)
            for joint_name, joint_data in MOCAP_DATA[frame_str].items():
                if joint_name not in joints_data:
                    joints_data[joint_name] = []
                joints_data[joint_name].append((
                    frame_num,
                    joint_data["x"],
                    joint_data["y"],
                    joint_data["z"]
                ))

        # Interpolate each joint
        current_frame_str = str(current_frame)
        if current_frame_str not in MOCAP_DATA:
            MOCAP_DATA[current_frame_str] = {}

        for joint_name, points in joints_data.items():
            if len(points) >= 2:  # Need at least 2 points for interpolation
                x, y, z = linear_interpolate_xyz(points, current_frame)
                MOCAP_DATA[current_frame_str][joint_name] = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": 0.9
                }

        # Save changes
        success, message = save_mocap_data(MOCAP_DATA)
        if success:
            self.report({'INFO'}, f"Interpolated frame {current_frame} using {len(frames_to_interpolate)} future frames.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class MOCAP_OT_interpolate_cubic(bpy.types.Operator):
    bl_idname = "mocap.interpolate_cubic"
    bl_label = "Cubic Interpolate"
    bl_description = "Interpolate current frame using cubic interpolation with surrounding frames"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        global MOCAP_DATA

        # Ensure data is loaded
        if MOCAP_DATA is None:
            success, message = load_mocap_data()
            if not success:
                self.report({'ERROR'}, message)
                return {'CANCELLED'}

        current_frame = context.scene.frame_current
        num_frames = get_interpolation_frames()

        # Get frames for interpolation (past and future frames around current)
        frames_to_interpolate = []
        for i in range(-num_frames, num_frames + 1):
            frame_num = current_frame + i
            frame_str = str(frame_num)
            if frame_str in MOCAP_DATA and not is_frame_hidden(MOCAP_DATA[frame_str]):
                frames_to_interpolate.append(frame_num)

        if len(frames_to_interpolate) < 4:
            self.report({'WARNING'}, f"Need at least 4 visible frames for cubic interpolation. Found {len(frames_to_interpolate)}.")
            return {'CANCELLED'}

        # For each joint, collect data points for interpolation
        joints_data = {}
        for frame_num in frames_to_interpolate:
            frame_str = str(frame_num)
            for joint_name, joint_data in MOCAP_DATA[frame_str].items():
                if joint_name not in joints_data:
                    joints_data[joint_name] = []
                joints_data[joint_name].append((
                    frame_num,
                    joint_data["x"],
                    joint_data["y"],
                    joint_data["z"]
                ))

        # Interpolate each joint
        current_frame_str = str(current_frame)
        if current_frame_str not in MOCAP_DATA:
            MOCAP_DATA[current_frame_str] = {}

        for joint_name, points in joints_data.items():
            if len(points) >= 4:  # Need at least 4 points for cubic interpolation
                x, y, z = cubic_interpolate_xyz(points, current_frame)
                MOCAP_DATA[current_frame_str][joint_name] = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": 0.9
                }
            elif len(points) >= 2:
                # Fall back to linear interpolation
                x, y, z = linear_interpolate_xyz(points, current_frame)
                MOCAP_DATA[current_frame_str][joint_name] = {
                    "x": x,
                    "y": y,
                    "z": z,
                    "confidence": 0.9
                }

        # Save changes
        success, message = save_mocap_data(MOCAP_DATA)
        if success:
            self.report({'INFO'}, f"Cubic interpolated frame {current_frame} using {len(frames_to_interpolate)} surrounding frames.")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, message)
            return {'CANCELLED'}


class MOCAP_OT_toggle_past_ghosts(bpy.types.Operator):
    bl_idname = "mocap.toggle_past_ghosts"
    bl_label = "Toggle Past Ghosts"
    bl_description = "Toggle visibility of past ghost figures"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global PAST_GHOSTS_VISIBLE
        PAST_GHOSTS_VISIBLE = not PAST_GHOSTS_VISIBLE
        update_all_ghost_visibility()

        status = "visible" if PAST_GHOSTS_VISIBLE else "hidden"
        self.report({'INFO'}, f"Past ghosts are now {status}")
        return {'FINISHED'}


class MOCAP_OT_toggle_future_ghosts(bpy.types.Operator):
    bl_idname = "mocap.toggle_future_ghosts"
    bl_label = "Toggle Future Ghosts"
    bl_description = "Toggle visibility of future ghost figures"
    bl_options = {'REGISTER'}

    def execute(self, context):
        global FUTURE_GHOSTS_VISIBLE
        FUTURE_GHOSTS_VISIBLE = not FUTURE_GHOSTS_VISIBLE
        update_all_ghost_visibility()

        status = "visible" if FUTURE_GHOSTS_VISIBLE else "hidden"
        self.report({'INFO'}, f"Future ghosts are now {status}")
        return {'FINISHED'}


# --- Blender Panel ---
class MOCAP_PT_editor_panel(bpy.types.Panel):
    bl_label = "Mocap Editor"
    bl_idname = "MOCAP_PT_editor_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Mocap"

    def draw(self, context):
        layout = self.layout

        # File Operations
        box = layout.box()
        box.label(text="File Operations")
        row = box.row()
        row.operator("mocap.reload_data", icon='FILE_REFRESH')

        # Frame Editing
        box = layout.box()
        box.label(text="Frame Editing")

        # Copy button added above Hide Current Frame
        row = box.row()
        row.operator("mocap.copy_previous_frame", icon='DUPLICATE')

        row = box.row()
        row.operator("mocap.toggle_hide_frame", icon='HIDE_ON' if not context.scene.get('frame_hidden', False) else 'HIDE_OFF')

        # Interpolation Options
        row = box.row()
        row.operator("mocap.interpolate_from_past", icon='TRIA_LEFT')
        row.operator("mocap.interpolate_from_future", icon='TRIA_RIGHT')

        row = box.row()
        row.operator("mocap.interpolate_cubic", icon='SMOOTHCURVE')

        # Ghost Controls
        box = layout.box()
        box.label(text="Ghost Controls")

        row = box.row()
        row.operator("mocap.toggle_past_ghosts", icon='GHOST_ENABLED')
        row.operator("mocap.toggle_future_ghosts", icon='GHOST_ENABLED')

        # Status Info
        box = layout.box()
        box.label(text="Status Info")

        if MOCAP_DATA is None:
            box.label(text="Mocap data: Not loaded", icon='ERROR')
        else:
            box.label(text=f"Mocap data: Loaded ({len(MOCAP_DATA)} frames)", icon='CHECKMARK')

        current_frame = context.scene.frame_current
        frame_str = str(current_frame)
        if MOCAP_DATA and frame_str in MOCAP_DATA:
            frame_hidden = is_frame_hidden(MOCAP_DATA[frame_str])
            status = "Hidden" if frame_hidden else "Visible"
            box.label(text=f"Frame {current_frame}: {status}", icon='HIDE_OFF' if frame_hidden else 'HIDE_ON')
        else:
            box.label(text=f"Frame {current_frame}: No data", icon='QUESTION')


# --- Registration ---
classes = [
    MOCAP_OT_reload_data,
    MOCAP_OT_toggle_hide_frame,
    MOCAP_OT_copy_previous_frame,
    MOCAP_OT_interpolate_from_past,
    MOCAP_OT_interpolate_from_future,
    MOCAP_OT_interpolate_cubic,
    MOCAP_OT_toggle_past_ghosts,
    MOCAP_OT_toggle_future_ghosts,
    MOCAP_PT_editor_panel,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Add frame change handler for ghost visibility
    if update_ghosts_on_frame_change not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(update_ghosts_on_frame_change)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Remove frame change handler
    if update_ghosts_on_frame_change in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(update_ghosts_on_frame_change)


if __name__ == "__main__":
    register()