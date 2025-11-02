# 4K_kid.py (Version 1.0 - First Kid)

import os
import shutil
import subprocess
import json
import sys

########################################################################
#                                                                      #
#    ######   #      #  #      #       #######  #      #  #   #####    #
#    #     R  #      #  ##     #          #     #      #  #  #         #
#    #     R  #      #  #  #   #          #     #      #  #  #         #
#    ###RRR   #      #  #  #   #          #     ########  #   ####     #
#    #   #    #      #  #   #  #          #     #      #  #       #    #
#    #    #   #      #  #     ##          #     #      #  #       #    #
#    #     #   ######   #      #          #     #      #  #  #####     #
#                                                                      #
########################################################################

# --- Import Project Utilities ---
# Add the project root (HMP_SW) to the system path so we can import utils.py
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # Normalize path for better compatibility (e.g., C:\...)
    path_to_check = os.path.normpath(current_script_dir)
    project_root = None

    # Robustly traverse upwards until 'utils.py' is found
    while True:
        # Check if 'utils.py' exists in the current directory being checked
        if os.path.exists(os.path.join(path_to_check, "utils.py")):
            project_root = path_to_check
            break

        parent_path = os.path.dirname(path_to_check)

        # If we reached the filesystem root (path doesn't change), stop
        if parent_path == path_to_check:
            raise FileNotFoundError("Project root (containing 'utils.py') not found in parent directories.")

        path_to_check = parent_path

    # Add the found project root to the system path
    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

    # Now import the utilities file from the dynamically found project root
    from utils import (
        script_log,
        clear_log,
        load_global_config,
        get_project_root,
        get_scene_paths,
        get_current_show_name,
        get_current_scene_name,
        set_current_scene_name,
        load_tsv_list,
        get_show_path,
        get_blender_path,
        get_processing_step_paths,
        get_scene_folder_name,
        get_log_contents,
        get_log_path
    )

except ImportError as e:
    # This specifically catches the import failure
    print("--------------------------------------------------------------------------------------------------")
    print(f"FATAL ERROR: Could not import HMP_SW/utils.py.")
    print(f"Error: {e}")
    # Print effective sys.path to aid debugging shadowing issues
    print("\nCURRENT EFFECTIVE SYS.PATH (Check this list for other 'utils' module locations):")
    for i, p in enumerate(sys.path):
        print(f"  {i:02d}: {p}")
    print(
        "\nACTION REQUIRED: Please confirm that 'C:\\Users\\ken\\Documents\\Ken\\Fiction\\HMP_SW\\utils.py' exists and defines the function 'script_log'.")
    print("--------------------------------------------------------------------------------------------------")
    sys.exit(1)
except FileNotFoundError as e:
    # This catches the robust path search failure
    print(f"FATAL ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during utility setup: {e}")
    sys.exit(1)

# Local config file name (in the same folder as this script)
CONFIG_FILE = "4K_kid_config.json"
BASE_BLEND_FILE = "4K_kid_base.blend"
EDIT_SCRIPT = "4K_kid_inner.py"


def load_local_config():
    """Load local configuration from JSON file in script directory"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, CONFIG_FILE)

        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        script_log(f"Error: Local config file '{CONFIG_FILE}' not found.", force_log=True)
        sys.exit(1)
    except json.JSONDecodeError:
        script_log(f"Error: Could not decode JSON from '{CONFIG_FILE}'. Check file format.", force_log=True)
        sys.exit(1)


def launch_blender_with_file(blender_path, blend_file_path):
    """Launch Blender GUI with the specified .blend file"""
    if not os.path.exists(blender_path):
        script_log(f"Error: Blender executable not found at '{blender_path}'")
        return False

    if not os.path.exists(blend_file_path):
        script_log(f"Error: Blend file not found at '{blend_file_path}'")
        return False

    try:
        command = [blender_path, blend_file_path]
        script_log(f"Launching Blender GUI with file: {blend_file_path}")

        # Use Popen to launch without blocking
        process = subprocess.Popen(command)
        script_log(f"Blender launched successfully with PID: {process.pid}")
        return True

    except Exception as e:
        script_log(f"Error launching Blender: {e}")
        return False

def main():
    clear_log()
    script_log("=== 4K KID STARTED ===")

    # Get Blender executable path from global config
    blender_path = get_blender_path()
    script_log(f"Blender: {blender_path}")

    # Get path to the inner Blender script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inner_script_path = os.path.join(script_dir, EDIT_SCRIPT)

    if not os.path.exists(inner_script_path):
        script_log(f"Error: Inner Blender script not found at '{inner_script_path}'")
        return

    # Get project root (already found during import)
    project_root = get_project_root()

    # Get show and scene information
    show_name = get_current_show_name()
    scene_name = get_current_scene_name(show_name)

    script_log(f"Show: {show_name}")
    script_log(f"Scene: {scene_name}")

    # Load local config for kid-specific settings
    local_config = load_local_config()

    # Get file paths from scene-config.json using processing steps
    step_paths = get_processing_step_paths(show_name, scene_name, "blender_animation")
    input_json_file = step_paths['input_file']  # Use output from apply_physics step

    # Get base blend file path (located in script directory)
    base_blend_path = os.path.join(script_dir, BASE_BLEND_FILE)

    script_log(f"Input JSON: {input_json_file}")
    script_log(f"Base blend file: {base_blend_path}")

    # Build the correct output path for the final kid figure animation
    scene_folder_name = get_scene_folder_name(show_name, scene_name)
    show_path = get_show_path(show_name)
    outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
    output_blend_file = os.path.join(outputs_dir, f"{scene_name}_kid.blend")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_blend_file), exist_ok=True)

    # Copy the base file from script directory to output location
    if os.path.exists(base_blend_path):
        shutil.copy2(base_blend_path, output_blend_file)
        script_log(f"Copied base file from {base_blend_path} to {output_blend_file}")
    else:
        script_log(f"ERROR: Base file not found at {base_blend_path}")
        sys.exit(1)

    # Validate required files
    if not os.path.exists(blender_path):
        script_log(f"Error: Blender executable not found at '{blender_path}'")
        sys.exit(1)

    if not os.path.exists(input_json_file):
        script_log(f"Error: Mocap data file '{input_json_file}' not found.")
        script_log(f"Expected file from apply_physics step: {input_json_file}")
        sys.exit(1)

    # Run Blender with our script. Use the output blend file
    command = [
        blender_path,
        "--background",
        output_blend_file,  # Use the copied output file
        "--python",
        inner_script_path,
        "--",  # Separator for Blender's args
        "--project-root", project_root,
        "--show", show_name,
        "--scene", scene_name
    ]

    script_log("Launching Blender with 4K_kid_inner.py...")
    script_log(f"Command: {' '.join(command)}")
    script_log(f"Using output blend file: {output_blend_file}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        def has_error(result):
            return "error" in result.lower()

        if result.returncode == 0 and not has_error(get_log_contents()):
            script_log("Blender execution completed successfully")
            script_log(f"Kid figure animation saved to: {output_blend_file}")
            script_log("=== LAUNCHING BLENDER GUI ===")
            launch_blender_with_file(blender_path, output_blend_file)
        else:
            script_log(f"Blender execution failed with return code: {result.returncode}")
            script_log(f"Blender stderr: {result.stderr}")

            notepad_path = r"C:\Windows\System32\notepad.exe"
            log_file_path = get_log_path()
            command = [notepad_path, log_file_path]
            subprocess.Popen(command)
    except subprocess.CalledProcessError as e:
        script_log(f"Error running Blender: {e}")
        script_log(f"Blender stderr: {e.stderr}")
    except Exception as e:
        script_log(f"Unexpected error: {e}")

    script_log("=== 4K KID COMPLETE ===")



if __name__ == "__main__":
    main()