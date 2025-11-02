# 4L_reskin.py (Version 2.0 - Reskin Pipeline Controller - Added Log Error Checking)

import os
import shutil
import subprocess
import sys
import time

############################################################
#                                                          #
#    ####   #    #  #    #      #####  #    #  #   ####    #
#    #   #  #    #  ##   #        #    #    #  #  #        #
#    ####   #    #  # #  #        #    ######  #   ###     #
#    # #    #    #  #  # #        #    #    #  #      #    #
#    #   #   ####   #   ##        #    #    #  #  ####     #
#                                                          #
############################################################


########################################################################
# --- Import Project Utilities ---
# Add the project root (HMP_SW) to the system path so we can import utils.py
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_check = os.path.normpath(current_script_dir)
    project_root = None

    # Robustly traverse upwards until 'utils.py' is found
    for i in range(5):
        parent_dir = os.path.dirname(path_to_check)
        if os.path.exists(os.path.join(parent_dir, 'utils.py')):
            project_root = parent_dir
            break
        path_to_check = parent_dir

    if project_root is None:
        print("ERROR: Could not find project root (directory containing utils.py).")
        sys.exit(1)

    if project_root not in sys.path:
        sys.path.append(project_root)

    # Imports necessary for the pipeline control. Added log checking utilities.
    from utils import script_log, comment, get_processing_step_paths, \
        get_current_show_name, get_current_scene_name, get_blender_path, \
        clear_log, get_log_contents, get_log_file_path  # <-- ADDED LOG UTILITIES

except ImportError as e:
    print(f"FAILED to import utilities: {e}")
    sys.exit(1)

# --- Configuration and Paths (UPDATED for 4L_reskin) ---
INNER_SCRIPT_NAME = "4L_reskin_inner.py"
INPUT_STEP_NAME = "4K_kid"  # Output of 4K_kid.py is the input
OUTPUT_STEP_NAME = "4L_reskin"  # Output of this script
CONFIG_FILE_NAME = "4L_reskin_config.json"  # Configuration file name
LOG_FILE_NAME = 'general_log.txt'  # Defined the standard log file name


def launch_blender_with_file(blender_path, blend_file_path):
    """Launch Blender GUI with the specified .blend file."""
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


def check_log_for_errors_and_act(blender_path, output_blend_file):
    """
    Checks the project log file for the word 'error' using utils.get_log_contents().
    If found, it displays and attempts to open the log file. Otherwise, it launches the Blender GUI.
    """
    script_log("\n--- POST-PROCESSING ACTION ---")

    try:
        # Give the system a moment to finish writing the log file
        time.sleep(0.5)

        # 1. Get Log Content (flow control is based on this)
        log_content = get_log_contents(LOG_FILE_NAME)
        log_file_path = get_log_file_path(LOG_FILE_NAME)  # Get path for output/I/O

        if not log_content:
            script_log(
                f"Warning: Log file not found at '{log_file_path}' or is empty. Assuming success and launching Blender.")
            launch_blender_with_file(blender_path, output_blend_file)
            return

        if "error" in log_content.lower():

            script_log("!!! ERROR DETECTED IN LOG FILE !!!")
            script_log(f"Blender process reported an error. Displaying log file: {log_file_path}")

            # Attempt to open the log file with the default system viewer
            try:
                # Use os.startfile for Windows (e.g., opens in Notepad)
                if sys.platform == "win32":
                    os.startfile(log_file_path)
                elif sys.platform == "darwin":
                    subprocess.Popen(['open', log_file_path])
                else:
                    script_log("Could not automatically open log file. Please check the path manually.")
            except Exception as e:
                script_log(f"Error opening log file with system tool: {e}")

        else:
            script_log("No errors detected in the log. Launching Blender GUI for visual inspection.")
            launch_blender_with_file(blender_path, output_blend_file)

    except Exception as e:
        script_log(f"FATAL ERROR during log check/action: {e}. Launching Blender GUI as a fallback.")
        launch_blender_with_file(blender_path, output_blend_file)

    script_log("\n--- ACTION COMPLETE ---")


def normalize_path_result(path_result, step_name):
    """Helper function to normalize path utility output to a directory string."""
    # This helper is included for robustness in path handling
    if isinstance(path_result, str):
        return path_result
    elif isinstance(path_result, dict):
        if 'dir' in path_result:
            return path_result['dir']
        elif 'input_file' in path_result and 'output_file' in path_result:
            dir_path = os.path.dirname(path_result['output_file'])
            return dir_path
        else:
            script_log(f"FATAL ERROR: Path utility failed for step ('{step_name}'). Dictionary missing required keys.")
            script_log(f"Returned value: {path_result}")
            sys.exit(1)
    else:
        script_log(f"FATAL ERROR: Path utility failed for step ('{step_name}'). Invalid return type.")
        script_log(f"Returned value: {path_result}")
        sys.exit(1)


def main():
    """Main function to control the reskin process."""

    # 0. Clear Log and Start
    try:
        clear_log(LOG_FILE_NAME)
    except Exception as e:
        script_log(f"WARNING: Failed to clear log file '{LOG_FILE_NAME}': {e}. Continuing execution.")

    script_log("=== 4L RESKIN CONTROLLER STARTING ===")

    try:
        # 1. Read show and scene names from the project environment using utils
        show_name = get_current_show_name()
        scene_name = get_current_scene_name(show_name)
    except Exception as e:
        script_log(f"ERROR: Could not determine show or scene name from utils: {e}")
        script_log("Please ensure the script is run from within the correct project context.")
        sys.exit(1)

    script_log(f"RESKIN STARTING (Show: {show_name}, Scene: {scene_name})")

    # Get the required paths using the project utilities
    input_paths = get_processing_step_paths(show_name, scene_name, INPUT_STEP_NAME)
    output_paths = get_processing_step_paths(show_name, scene_name, OUTPUT_STEP_NAME)

    # Resolve actual directories
    input_dir = normalize_path_result(input_paths, INPUT_STEP_NAME)
    output_dir = normalize_path_result(output_paths, OUTPUT_STEP_NAME)

    # Construct the blend file paths based on naming convention
    # Input Blend File: The output of the 4K_kid step, used as input here.
    input_blend_file_name = f"{scene_name}-{INPUT_STEP_NAME}.blend"
    input_blend_file = os.path.join(input_dir, input_blend_file_name)

    # Output Blend File: The file we will save the reskinned result to.
    output_blend_file_name = f"{scene_name}-{OUTPUT_STEP_NAME}.blend"
    output_blend_file = os.path.join(output_dir, output_blend_file_name)

    inner_script_path = os.path.join(current_script_dir, INNER_SCRIPT_NAME)

    # Define and check for the config file path
    config_file_path = os.path.join(current_script_dir, CONFIG_FILE_NAME)
    if not os.path.exists(config_file_path):
        script_log(f"ERROR: Config file not found: {config_file_path}")
        script_log(f"Please create {CONFIG_FILE_NAME} with the required parameters.")
        sys.exit(1)

    # 2. Check for input file
    if not os.path.exists(input_blend_file):
        script_log(f"ERROR: Input blend file not found: {input_blend_file}")
        script_log(f"Please run {INPUT_STEP_NAME}.py first to generate the base rig.")
        sys.exit(1)

    # 3. Copy the file for non-destructive processing
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(output_blend_file):
        script_log(f"Existing output file found: {output_blend_file}. Removing and replacing.")
        os.remove(output_blend_file)

    shutil.copy2(input_blend_file, output_blend_file)
    script_log(f"Copied input file to new reskin file: {output_blend_file}")

    # 4. Find Blender executable
    try:
        blender_path = get_blender_path()
    except Exception as e:
        script_log(f"ERROR: Failed to retrieve Blender path from utilities. Check configuration. Details: {e}")
        sys.exit(1)

    if not os.path.exists(blender_path):
        script_log(f"ERROR: Blender executable not found at configured path: {blender_path}")
        sys.exit(1)

    # 5. Run Blender with the inner script, passing the config file path
    command = [
        blender_path,
        "--background",
        output_blend_file,  # Work directly on the copied file
        "--python",
        inner_script_path,
        "--",  # Separator for Blender's args
        "--project-root", project_root,
        "--show", show_name,
        "--scene", scene_name,
        "--config-file-path", config_file_path  # Pass the configuration file path
    ]

    script_log("Launching Blender with 4L_reskin_inner.py in background mode...")
    script_log(f"Command: {' '.join(command)}")
    script_log(f"Using reskin blend file: {output_blend_file}")

    try:
        # NOTE: check=False is used so the script can continue to check the log file
        # regardless of Blender's exit code, allowing log checking for internal errors.
        subprocess.run(command, capture_output=True, text=True, check=False)

        script_log("Blender background execution finished. Proceeding to log check.")

    except Exception as e:
        script_log(f"An execution error occurred while running Blender (before log check): {e}")

    # 6. Check log for errors and launch Blender GUI or display log
    check_log_for_errors_and_act(blender_path, output_blend_file)

    script_log("=== 4L RESKIN CONTROLLER EXITING ===")


if __name__ == "__main__":
    main()
