# 0_RunMocapAnimPipeline.py (Version 1.1)

import subprocess
import os
import sys

def run_script(script_name, *args):
    """
    Runs a Python script using subprocess.

    Args:
        script_name (str): The name of the script to run.
        *args: Any additional arguments to pass to the script.
    """
    script_path = os.path.abspath(script_name)
    if not os.path.exists(script_path):
        print(f"Error: Script not found at '{script_path}'")
        sys.exit(1)

    command = [sys.executable, script_path] + list(args)
    print(f"\n--- Running: {' '.join(command)} ---")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"Errors from {script_name}:\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while running {script_name}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Define common paths (adjust these as necessary for your environment)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    blender_exe_path = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"
    cloudrig_base_blend_path = os.path.join(current_dir, "rain_v3.2 - Copy.blend") # Example, adjust if needed
    bone_mapping_file_path = os.path.join(current_dir, "blender_bone_mapping.json")

    # Step 1: Run 1_ExtractPoseData.py
    print("Starting Step 1: Extracting Pose Data...")
    run_script("1_ExtractPoseData.py")
    print("Step 1 Completed.")

    # Step 2: Run 2_FilterData.py
    print("\nStarting Step 2: Filtering Data...")
    run_script("2_FilterData.py")
    print("Step 2 Completed.")

    # Step 3: Run 3_ApplyPhysics.py
    print("\nStarting Step 3: Applying Physics (Data Copy)...")
    run_script("3_ApplyPhysics.py")
    print("Step 3 Completed.")

    # Step 4: Run 4D_magic.py
    print("\nStarting Step 4: Animating CloudRig in Blender...")
    # These file names are hardcoded inside 4D_magic.py's __main__ block.
    # If 4D_magic.py were designed to accept these as arguments, they'd be passed here.
    # For now, we assume 4D_magic.py will use its internal paths.
    # If you need to pass these, 4D_magic.py's main function would need modification.
    # Example if 4D_magic.py accepted args:
    # run_script("4D_magic.py",
    #            "--json", "step_4_input.json",
    #            "--output", "step_4_output.blend",
    #            "--blender_exe", blender_exe_path,
    #            "--cloudrig_blend", cloudrig_base_blend_path,
    #            "--mapping", bone_mapping_file_path)
    run_script("4D_magic.py")
    print("Step 4 Completed. Animation pipeline finished.")
