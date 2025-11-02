# 5_flag.py (version 1.5 - Calls 5_flag_inner.py in a headless Blender instance)

import os
import subprocess
import shutil
import sys
import warnings
import json

# --- Configuration ---
# Define base directory relative to the script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Import centralized script_log function from the dedicated common utils file
try:
    sys.path.append(os.path.join(SCRIPT_DIR, '.'))
    from _4D_magic_utils import script_log, clear_log
except ImportError as e:
    warnings.warn(f"Warning: Could not import utility functions from _4D_magic_utils.py. "
                  f"Error: {e}. Logging will be printed to console only.", ImportWarning)


    # Define dummy functions to prevent a crash if the import fails
    def script_log(message, file_path=None):
        print(message)


    def clear_log():
        pass

# File paths for Blender automation
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, "flag_config.json")


def load_config():
    """Loads configuration data from a JSON file."""
    if not os.path.exists(CONFIG_FILE_PATH):
        script_log(f"Error: Configuration file not found at '{CONFIG_FILE_PATH}'")
        return None
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        script_log(f"Error loading configuration: {e}")
        return None


# --- Main Script Logic ---
def run_blender_automation():
    """
    Copies the base Blender file, checks 3D data, and
    executes the inner script within Blender's context.
    """
    script_log("--- Starting Blender Automation ---")
    clear_log()

    # 1. Load the configuration
    config = load_config()
    if not config:
        script_log("Could not load configuration. Exiting.")
        return

    # 2. Get other config values
    base_blend_file = os.path.join(SCRIPT_DIR, config.get("flag_input_blend_path"))
    destination_blend_file = os.path.join(SCRIPT_DIR, config.get("flag_ouput_blend_path"))
    inner_script_path = os.path.join(SCRIPT_DIR, "5_flag_inner.py")

    # 3. Check if the base blend file exists
    if not os.path.exists(base_blend_file):
        script_log(f"Error: Base Blender file not found at '{base_blend_file}'")
        return

    # 4. Check if the inner script exists
    if not os.path.exists(inner_script_path):
        script_log(f"Error: Inner script not found at '{inner_script_path}'")
        return

    # 5. Copy the base file to the destination
    script_log(f"Copying '{base_blend_file}' to '{destination_blend_file}'")
    shutil.copyfile(base_blend_file, destination_blend_file)

    # 6. Construct the command to run Blender
    blender_exe = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"

    command = [
        blender_exe,
        "--background",
        destination_blend_file,
        "--python",
        inner_script_path
    ]

    # 7. Execute the command
    script_log("\nExecuting Blender command:")
    script_log(" ".join(command))

    try:
        subprocess.check_call(command)
        script_log("\n--- Blender automation complete! ---")
        script_log(f"Final animated file saved at: '{destination_blend_file}'")
    except FileNotFoundError:
        script_log("Error: 'blender' command not found.")
        script_log("Please ensure Blender is installed and added to your system's PATH.")
    except subprocess.CalledProcessError as e:
        script_log(f"Error: Blender process failed with exit code {e.returncode}")
        script_log(f"stdout: {e.stdout.decode('utf-8') if e.stdout else 'None'}")
        script_log(f"stderr: {e.stderr.decode('utf-8') if e.stderr else 'None'}")
    except Exception as e:
        script_log(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    run_blender_automation()