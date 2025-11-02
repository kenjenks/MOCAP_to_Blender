# 4K_kid_anim.py (version 1.0 - Animation Pipeline Controller)

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
"""
Key Features:

* 4K_kid_anim.py - The controller that:
*   * Finds the kid figure blend file created by 4K_kid.py
*   * Copies it to a new *_anim.blend file
*   * Launches Blender with the animation script

* 4K_kid_anim_inner.py - The animation engine that:
*   * Loads the existing armature and control points from the scene
*   * Applies MediaPipe landmark data to animate the rig
*   * Preserves any customizations (UV maps, costumes, shapes) made to the figure
*   * Saves the animated result

Workflow:

* Run 4K_kid.py to create the base figure
* Modelers can customize the *_kid.blend file (add UV maps, change shapes, add costumes)
* Run 4K_kid_anim.py to animate the customized figure
* Get the final *_anim.blend file with both customizations and animation

This separation allows artists to work on the visual aspects while the animation pipeline 
handles the technical motion application.
"""

# --- Import Project Utilities ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_check = os.path.normpath(current_script_dir)
    project_root = None

    while True:
        if os.path.exists(os.path.join(path_to_check, "utils.py")):
            project_root = path_to_check
            break

        parent_path = os.path.dirname(path_to_check)
        if parent_path == path_to_check:
            raise FileNotFoundError("Project root (containing 'utils.py') not found in parent directories.")
        path_to_check = parent_path

    if project_root and project_root not in sys.path:
        sys.path.append(project_root)

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
        get_scene_folder_name
    )

except ImportError as e:
    print("--------------------------------------------------------------------------------------------------")
    print(f"FATAL ERROR: Could not import HMP_SW/utils.py.")
    print(f"Error: {e}")
    print("\nCURRENT EFFECTIVE SYS.PATH:")
    for i, p in enumerate(sys.path):
        print(f"  {i:02d}: {p}")
    print(
        "\nACTION REQUIRED: Please confirm that 'C:\\Users\\ken\\Documents\\Ken\\Fiction\\HMP_SW\\utils.py' exists and defines the function 'script_log'.")
    print("--------------------------------------------------------------------------------------------------")
    sys.exit(1)
except FileNotFoundError as e:
    print(f"FATAL ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during utility setup: {e}")
    sys.exit(1)

# Local config file name (in the same folder as this script)
CONFIG_FILE = "4K_kid_config.json"
ANIM_SCRIPT = "4K_kid_anim_inner.py"


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


def find_kid_blend_file(show_name, scene_name):
    """Find the kid figure blend file created by 4K_kid.py"""
    show_path = get_show_path(show_name)
    scene_folder_name = get_scene_folder_name(show_name, scene_name)
    outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)

    # Look for the kid figure file
    kid_blend_file = os.path.join(outputs_dir, f"{scene_name}_kid.blend")
    anim_blend_file = os.path.join(outputs_dir, f"{scene_name}_anim.blend")

    if not os.path.exists(kid_blend_file):
        script_log(f"Error: Kid figure file not found at '{kid_blend_file}'", force_log=True)
        script_log("Please run 4K_kid.py first to create the figure.", force_log=True)
        return None, None

    return kid_blend_file, anim_blend_file


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
    script_log("=== 4K KID ANIMATION PIPELINE STARTED ===")

    # Get Blender executable path from global config
    blender_path = get_blender_path()
    script_log(f"Blender: {blender_path}")

    # Get path to the inner animation script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inner_script_path = os.path.join(script_dir, ANIM_SCRIPT)

    if not os.path.exists(inner_script_path):
        script_log(f"Error: Animation script not found at '{inner_script_path}'")
        return

    # Get project root
    project_root = get_project_root()

    # Get show and scene information
    show_name = get_current_show_name()
    scene_name = get_current_scene_name(show_name)

    script_log(f"Show: {show_name}")
    script_log(f"Scene: {scene_name}")

    # Load local config for animation settings
    local_config = load_local_config()

    # Find the kid figure blend file and set up animation file
    kid_blend_file, anim_blend_file = find_kid_blend_file(show_name, scene_name)
    if not kid_blend_file:
        sys.exit(1)

    # Copy the kid figure file to animation file
    shutil.copy2(kid_blend_file, anim_blend_file)
    script_log(f"Copied kid figure from {kid_blend_file} to {anim_blend_file}")

    # Validate required files
    if not os.path.exists(blender_path):
        script_log(f"Error: Blender executable not found at '{blender_path}'")
        sys.exit(1)

    # Run Blender with our animation script
    command = [
        blender_path,
        "--background",
        anim_blend_file,  # Use the animation file
        "--python",
        inner_script_path,
        "--",  # Separator for Blender's args
        "--project-root", project_root,
        "--show", show_name,
        "--scene", scene_name
    ]

    script_log("Launching Blender with 4K_kid_anim_inner.py...")
    script_log(f"Command: {' '.join(command)}")
    script_log(f"Using animation blend file: {anim_blend_file}")

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        if result.returncode == 0:
            script_log("Blender animation execution completed successfully")
            script_log(f"Animated figure saved to: {anim_blend_file}")
        else:
            script_log(f"Blender execution failed with return code: {result.returncode}")
            script_log(f"Blender stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        script_log(f"Error running Blender: {e}")
        script_log(f"Blender stderr: {e.stderr}")
    except Exception as e:
        script_log(f"Unexpected error: {e}")

    script_log("=== 4K KID ANIMATION PIPELINE COMPLETE ===")
    script_log("=== LAUNCHING BLENDER GUI ===")
    launch_blender_with_file(blender_path, anim_blend_file)

if __name__ == "__main__":
    main()