# 4D_magic.py (Version 2.0 - Using config files)

import os
import sys
import subprocess
import shutil

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
        load_global_config,
        get_scene_blender_input_path,
        get_project_root,
        get_scene_paths,
        get_current_show_name,
        get_current_scene_name,
        set_current_scene_name,
        load_tsv_list,
        get_show_path,
        get_blender_path
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
    print("\nACTION REQUIRED: Please confirm that 'C:\\Users\\ken\\Documents\\Ken\\Fiction\\HMP_SW\\utils.py' exists and defines the function 'script_log'.")
    print("--------------------------------------------------------------------------------------------------")
    sys.exit(1)
except FileNotFoundError as e:
    # This catches the robust path search failure
    print(f"FATAL ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FATAL ERROR during utility setup: {e}")
    sys.exit(1)


def main():
    script_log("=== 4D MAGIC STARTED ===")

    # Get Blender executable path
    blender_path = get_blender_path()
    script_log(f"Blender: {blender_path}")

    # Get path to the inner Blender script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inner_script_path = os.path.join(script_dir, "4D_magic_inner.py")

    if not os.path.exists(inner_script_path):
        script_log(f"Error: Inner Blender script not found at '{inner_script_path}'")
        return

    # Get project root (already found during import)
    project_root = get_project_root()

    show_name = get_current_show_name()
    scene_name = get_current_scene_name(show_name)

    # Get the input Blender rig file path from scene-config.json
    input_blender_rig_file = get_scene_blender_input_path(show_name, scene_name)

    # FIXED: Use get_scene_paths to get the correct output directory structure
    scene_paths = get_scene_paths(show_name, scene_name)

    # Get the scene folder name for proper path construction
    from utils import get_scene_folder_name
    scene_folder_name = get_scene_folder_name(show_name, scene_name)

    # Build the correct output path using show_path
    show_path = get_show_path(show_name)
    outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
    output_blend_file = os.path.join(outputs_dir, f"{scene_name}.blend")

    script_log(f"Input rig file: {input_blender_rig_file}")
    script_log(f"Output blend file: {output_blend_file}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_blend_file), exist_ok=True)

    # Copy the rig file to output location
    if os.path.exists(input_blender_rig_file):
        shutil.copy2(input_blender_rig_file, output_blend_file)
        script_log(f"Copied rig file from {input_blender_rig_file} to {output_blend_file}")
    else:
        script_log(f"ERROR: Rig file not found at {input_blender_rig_file}")

    # Rest of the function remains the same...
    # Pass ONLY project_root to Blender - let inner script figure out the rest
    blender_args = [
        blender_path,
        "--background",
        "--python", inner_script_path,
        "--",  # Separator: everything after this goes to the script, not Blender
        "--project-root", project_root
    ]

    script_log("Launching Blender with 4D_magic_inner.py...")
    script_log(f"Command: {' '.join(blender_args)}")
    script_log(f"Project Root: {project_root}")

    try:
        result = subprocess.run(blender_args, capture_output=True, text=True, check=True)
        if result.returncode == 0:
            script_log("Blender execution completed successfully")
        else:
            script_log(f"Blender execution failed with return code: {result.returncode}")
            script_log(f"Blender stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        script_log(f"Error running Blender: {e}")
        script_log(f"Blender stderr: {e.stderr}")
    except Exception as e:
        script_log(f"Unexpected error: {e}")

    script_log("=== 4D MAGIC COMPLETE ===")


if __name__ == "__main__":
    main()