# blender_measure_rig.py

import bpy  # Import bpy for Blender operations
import json
import sys
import os
from mathutils import Vector  # Import Vector for mathutils operations
from datetime import datetime  # Import datetime for timestamps

# Define the log file path
log_file_path = "blender_measure_rig_phase_0_debug.txt"

# Store original stdout
original_stdout = sys.stdout
log_file = None  # Initialize to None

# Define the output proportions path globally with a default value
# This ensures it's always defined, even if no argument is passed via sys.argv
output_proportions_path = "./step_4_input_measures_Rainv3_2.json"


# --- Helper function to print to log and console (if available) ---
def log_message(message):
    global log_file, original_stdout
    # Try to write to log file first
    if log_file:
        try:
            log_file.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
            log_file.flush()  # Ensure message is written immediately
        except ValueError:
            # If log_file is closed, fall back to original stdout
            if original_stdout and original_stdout != sys.stdout:
                original_stdout.write(
                    f"[{datetime.now().strftime('%H:%M:%S')}] LOG_ERROR (log_file closed): {message}\n")
                original_stdout.flush()
            return
    # Always try to print to original stdout if it's different from current sys.stdout
    if original_stdout and sys.stdout != original_stdout:
        original_stdout.write(f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
        original_stdout.flush()
    # If sys.stdout is already original_stdout (e.g., running directly), just print
    elif sys.stdout == original_stdout:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")


# --- Parse command-line arguments for output_proportions_path BEFORE the main try block ---
# This ensures output_proportions_path is set correctly when Blender calls the script.
if "--" in sys.argv:
    try:
        # Arguments after '--' are custom arguments
        index = sys.argv.index("--")
        if len(sys.argv) > index + 1:
            output_proportions_path = sys.argv[index + 1]
            # log_message(f"DEBUG: output_proportions_path updated from sys.argv: {output_proportions_path}")
        else:
            log_message("Warning: '--' found but no argument followed it. Using default output_proportions_path.")
    except ValueError:
        # This case should ideally not happen if "--" is checked first
        log_message("Error: Problem parsing sys.argv for output_proportions_path.")
else:
    log_message("INFO: '--' separator not found in sys.argv. Using default output_proportions_path.")


try:
    # Open the log file for writing, overwriting previous content
    log_file = open(log_file_path, 'w')
    sys.stdout = log_file  # Redirect stdout to the log file
    log_message(f"--- Starting Blender Rig Measurement PHASE 0 Debug Log (blender_measure_rig.py) ---")
    log_message(f"Log file opened at: {log_file_path}")
    log_message("DEBUG: This is a test message from PHASE 0 script.")
    log_message(f"DEBUG: Final output_proportions_path: {output_proportions_path}")


    proportions = {}

    # --- Step 1: Ensure we are in OBJECT mode and get the armature ---
    log_message("DEBUG: Step 1.1 - Entered measure_blender_rig and set to OBJECT mode.")
    bpy.ops.object.mode_set(mode='OBJECT')

    # Get the armature object (assuming it's the selected one or named 'RIG-rain')
    armature_obj = bpy.context.view_layer.objects.active
    if not armature_obj or armature_obj.type != 'ARMATURE':
        # Try to find by name if not active or not armature
        armature_obj = bpy.data.objects.get('RIG-rain')
        if not armature_obj or armature_obj.type != 'ARMATURE':
            log_message("Error: No active armature object found or 'RIG-rain' not found. Please select the armature.")
            sys.exit(1)
    log_message("DEBUG: Step 1.2 - Armature object retrieved.")

    # --- Step 2: Select the armature and set it active ---
    bpy.ops.object.select_all(action='DESELECT')
    armature_obj.select_set(True)
    bpy.context.view_layer.objects.active = armature_obj
    log_message("DEBUG: Step 2.1 - Armature type verified.")
    log_message("DEBUG: Step 2.2 - Armature selected and set active.")

    # --- Step 3: Enter EDIT mode to get bone head/tail positions ---
    bpy.ops.object.mode_set(mode='EDIT')
    edit_bones = armature_obj.data.edit_bones
    log_message("DEBUG: Step 3.1 - Entered EDIT mode and got edit_bones.")

    # --- Step 4: Get armature's world matrix and define helper for world positions ---
    armature_world_matrix = armature_obj.matrix_world
    log_message("DEBUG: Step 4.1 - Got armature_world_matrix.")


    def get_edit_bone_world_pos(bone_name, point_type='head'):
        bone = edit_bones.get(bone_name)
        if bone:
            if point_type == 'head':
                local_pos = bone.head
            elif point_type == 'tail':
                local_pos = bone.tail
            else:
                log_message(f"Error: Invalid point_type '{point_type}' for bone '{bone_name}'.")
                return None
            world_pos = armature_world_matrix @ local_pos
            log_message(
                f"  DEBUG: Bone '{bone_name}' {point_type} local: {local_pos.x:.4f}, {local_pos.y:.4f}, {local_pos.z:.4f}")
            log_message(
                f"  DEBUG: Bone '{bone_name}' {point_type} world: {world_pos.x:.4f}, {world_pos.y:.4f}, {world_pos.z:.4f}")
            return world_pos
        else:
            log_message(
                f"Warning: Bone '{bone_name}' not found for world position calculation (point_type: {point_type}).")
            return None


    log_message("DEBUG: Step 4.2 - Defined get_edit_bone_world_pos helper.")

    # --- Step 5: Reference Height (e.g., top of head to ground) ---
    # Using 'DEF-Head' tail for top of head, and 'IK-Foot.L/R' for ground reference
    head_top_pos = get_edit_bone_world_pos('DEF-Head', 'tail')  # Corrected case
    ik_foot_l_pos = get_edit_bone_world_pos('IK-Foot.L', 'head')
    ik_foot_r_pos = get_edit_bone_world_pos('IK-Foot.R', 'head')

    min_foot_z = float('inf')
    if ik_foot_l_pos:
        min_foot_z = min(min_foot_z, ik_foot_l_pos.z)
    if ik_foot_r_pos:
        min_foot_z = min(min_foot_z, ik_foot_r_pos.z)

    if head_top_pos and min_foot_z != float('inf'):
        proportions["blender_model_reference_height"] = head_top_pos.z - min_foot_z
    else:
        log_message("Warning: Could not determine Blender model reference height. Using default.")
        proportions["blender_model_reference_height"] = 1.6  # Default to 1.6 meters
    log_message("DEBUG: Step 5.1 - Calculated reference height.")

    # --- Step 6: Shoulder Width (distance between DEF-Upperarm1.L and DEF-Upperarm1.R heads) ---
    shoulder_l_pos = get_edit_bone_world_pos('DEF-Upperarm1.L', 'head')
    shoulder_r_pos = get_edit_bone_world_pos('DEF-Upperarm1.R', 'head')
    if shoulder_l_pos and shoulder_r_pos:
        proportions["blender_model_shoulder_width"] = (shoulder_l_pos - shoulder_r_pos).length
    else:
        log_message(
            "Warning: Could not determine Blender model shoulder width (using DEF-Upperarm1 heads). Using default.")
        proportions["blender_model_shoulder_width"] = 0.4  # Default
    log_message("DEBUG: Step 6.1 - Calculated shoulder width.")

    # --- Step 7: Hip Distance (REMOVED: This measurement is not straightforward with a single pelvis bone) ---
    # The 'blender_model_hip_distance' calculation has been removed as it was not accurately derived
    # from the 'DEF-Pelvis' bone and was falling back to a default.
    log_message(
        "DEBUG: Step 7.1 - Removed hip distance calculation as it was not directly measurable from available bones.")

    # --- Step 8: Upper Arm Length (Sum of DEF-Upperarm1.L/R and DEF-Upperarm2.L/R lengths) ---
    upper_arm_l_length = 0.0
    def_upper_arm1_l = edit_bones.get('DEF-Upperarm1.L')
    def_upper_arm2_l = edit_bones.get('DEF-Upperarm2.L')
    if def_upper_arm1_l:
        upper_arm_l_length += def_upper_arm1_l.length
        log_message(f"  DEBUG: Bone 'DEF-Upperarm1.L' length: {def_upper_arm1_l.length:.4f}")
    if def_upper_arm2_l:
        upper_arm_l_length += def_upper_arm2_l.length
        log_message(f"  DEBUG: Bone 'DEF-Upperarm2.L' length: {def_upper_arm2_l.length:.4f}")

    if upper_arm_l_length > 0:
        proportions["blender_model_upper_arm_left_length"] = upper_arm_l_length
    else:
        log_message("Warning: Could not determine Blender model left upper arm length. Using default.")
        proportions["blender_model_upper_arm_left_length"] = 0.3  # Default
    log_message("DEBUG: Step 8.1 - Calculated left upper arm length.")

    upper_arm_r_length = 0.0
    def_upper_arm1_r = edit_bones.get('DEF-Upperarm1.R')
    def_upper_arm2_r = edit_bones.get('DEF-Upperarm2.R')
    if def_upper_arm1_r:
        upper_arm_r_length += def_upper_arm1_r.length
        log_message(f"  DEBUG: Bone 'DEF-Upperarm1.R' length: {def_upper_arm1_r.length:.4f}")
    if def_upper_arm2_r:
        upper_arm_r_length += def_upper_arm2_r.length
        log_message(f"  DEBUG: Bone 'DEF-Upperarm2.R' length: {def_upper_arm2_r.length:.4f}")

    if upper_arm_r_length > 0:
        proportions["blender_model_upper_arm_right_length"] = upper_arm_r_length
    else:
        log_message("Warning: Could not determine Blender model right upper arm length. Using default.")
        proportions["blender_model_upper_arm_right_length"] = 0.3  # Default
    log_message("DEBUG: Step 8.2 - Calculated right upper arm length.")

    # --- Step 9: Forearm Length (Sum of DEF-Forearm1.L/R and DEF-Forearm2.L/R lengths) ---
    forearm_l_length = 0.0
    def_forearm1_l = edit_bones.get('DEF-Forearm1.L')
    def_forearm2_l = edit_bones.get('DEF-Forearm2.L')
    if def_forearm1_l:
        forearm_l_length += def_forearm1_l.length
        log_message(f"  DEBUG: Bone 'DEF-Forearm1.L' length: {def_forearm1_l.length:.4f}")
    if def_forearm2_l:
        forearm_l_length += def_forearm2_l.length
        log_message(f"  DEBUG: Bone 'DEF-Forearm2.L' length: {def_forearm2_l.length:.4f}")

    if forearm_l_length > 0:
        proportions["blender_model_forearm_left_length"] = forearm_l_length
    else:
        log_message("Warning: Could not determine Blender model left forearm length. Using default.")
        proportions["blender_model_forearm_left_length"] = 0.25  # Default
    log_message("DEBUG: Step 9.1 - Calculated left forearm length.")

    forearm_r_length = 0.0
    def_forearm1_r = edit_bones.get('DEF-Forearm1.R')
    def_forearm2_r = edit_bones.get('DEF-Forearm2.R')
    if def_forearm1_r:
        forearm_r_length += def_forearm1_r.length
        log_message(f"  DEBUG: Bone 'DEF-Forearm1.R' length: {def_forearm1_r.length:.4f}")
    if def_forearm2_r:
        forearm_r_length += def_forearm2_r.length
        log_message(f"  DEBUG: Bone 'DEF-Forearm2.R' length: {def_forearm2_r.length:.4f}")

    if forearm_r_length > 0:
        proportions["blender_model_forearm_right_length"] = forearm_r_length
    else:
        log_message("Warning: Could not determine Blender model right forearm length. Using default.")
        proportions["blender_model_forearm_right_length"] = 0.25  # Default
    log_message("DEBUG: Step 9.2 - Calculated right forearm length.")

    # --- Step 10: Torso Length (midpoint of shoulders to head of MSTR-Pelvis_Parent) ---
    # Using MSTR-Pelvis_Parent head for hip reference.
    mstr_pelvis_parent_pos = get_edit_bone_world_pos('MSTR-Pelvis_Parent', 'head') # Changed from DEF-Pelvis
    if shoulder_l_pos and shoulder_r_pos and mstr_pelvis_parent_pos:
        shoulder_midpoint = (shoulder_l_pos + shoulder_r_pos) / 2
        proportions["blender_model_torso_length"] = (shoulder_midpoint - mstr_pelvis_parent_pos).length
    else:
        log_message("Warning: Could not determine Blender model torso length. Using default.")
        proportions["blender_model_torso_length"] = 0.5  # Default
    log_message("DEBUG: Step 10.1 - Calculated torso length.")

    # --- Step 11: Hip Height (Z-coordinate of MSTR-Pelvis_Parent head from ground) ---
    if mstr_pelvis_parent_pos and min_foot_z != float('inf'):
        proportions["blender_model_hip_height"] = mstr_pelvis_parent_pos.z - min_foot_z
    else:
        log_message("Warning: Could not determine Blender model hip height. Using default.")
        proportions["blender_model_hip_height"] = 0.8  # Default
    log_message("DEBUG: Step 11.1 - Calculated hip height.")

    # --- NEW Step: Measure Head Length (robust approach) ---
    head_length = 0.0
    def_head_bone = edit_bones.get('DEF-Head')
    def_head_top_bone = edit_bones.get('DEF-Head_Top')
    def_neck_bone = edit_bones.get('DEF-Neck')
    def_spine3_bone = edit_bones.get('DEF-Spine3') # Alternative for neck base

    if def_head_bone and def_head_bone.length > 0.001: # Check for non-negligible length
        head_length = def_head_bone.length
        log_message(f"DEBUG: Used 'DEF-Head' bone length for head_length: {head_length:.4f}")
    elif def_head_top_bone and (def_neck_bone or def_spine3_bone):
        # Calculate distance from DEF-Head_Top to DEF-Neck or DEF-Spine3
        head_top_pos_world = get_edit_bone_world_pos('DEF-Head_Top', 'head') # Using head of Head_Top bone
        neck_base_pos_world = None

        if def_neck_bone:
            neck_base_pos_world = get_edit_bone_world_pos('DEF-Neck', 'head') # Head of neck bone
            log_message(f"DEBUG: Found 'DEF-Neck' bone for head length calculation.")
        elif def_spine3_bone:
            neck_base_pos_world = get_edit_bone_world_pos('DEF-Spine3', 'tail') # Tail of spine3 as neck base
            log_message(f"DEBUG: Found 'DEF-Spine3' bone (as alternative to neck) for head length calculation.")

        if head_top_pos_world and neck_base_pos_world:
            head_length = (head_top_pos_world - neck_base_pos_world).length
            log_message(f"DEBUG: Calculated head_length from DEF-Head_Top to neck bone: {head_length:.4f}")
        else:
            log_message("Warning: Could not determine head length from DEF-Head_Top and neck/spine bones. Using default.")
    else:
        log_message("Warning: Neither 'DEF-Head' nor 'DEF-Head_Top' with a suitable neck/spine bone found for head length. Using default.")

    if head_length > 0:
        proportions["blender_model_head_length"] = head_length
    else:
        proportions["blender_model_head_length"] = 0.2 # Default to 0.2 meters
        log_message(f"Warning: Final head length is 0 or not determined. Defaulting to {proportions['blender_model_head_length']:.4f} meters.")

    log_message("DEBUG: Step 12.1 - Calculated head length.")


    # --- Start of Step 13: Save to JSON ---
    try:
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_proportions_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_message(f"Created output directory: {output_dir}")

        with open(output_proportions_path, 'w') as f:
            json.dump(proportions, f, indent=4)
        log_message(f"Blender model proportions saved to: {output_proportions_path}")
    except Exception as e:
        log_message(f"Error saving Blender model proportions: {e}")
        sys.exit(1)  # Exit with error code on save failure
    log_message("DEBUG: Step 13.1 - Saved proportions to JSON.")

except Exception as e:
    log_message(f"An unexpected error occurred: {e}")
    sys.exit(1)
finally:
    # IMPORTANT: Restore original stdout and close the log file
    if log_file:
        log_message("--- Ending Blender Rig Measurement PHASE 0 Debug Log ---")
        sys.stdout = original_stdout  # Restore original stdout BEFORE closing the file
        log_file.close()

# The __main__ block is now primarily for direct script execution testing,
# as the argument parsing is moved outside for Blender's -P execution.
if __name__ == "__main__":
    # If run directly, and no arguments were passed, it will use the default path.
    # If arguments were passed, output_proportions_path would have been updated above.
    log_message("Script executed directly via __main__.")
    sys.exit(0)  # Final exit
