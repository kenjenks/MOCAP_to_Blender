# show_stick_animation.py (Version 2.0 with config file)

import os
import shutil
import subprocess
import json
import sys

CONFIG_FILE = "show_stick_animation_config.json"
BASE_BLEND_FILE = "show_stick_animation_base.blend"

def load_config():
    """Load configuration from JSON file"""
    with open(CONFIG_FILE, 'r') as file:
        config = json.load(file)
    cylinder_radius = 0.05  # Default cylinder radius
    cylinder_overlap = 0.02  # Default overlap epsilon
    return (
        config.get("file_settings", {}).get("input_json", "step_4_input.json"),
        config.get("file_settings", {}).get("output_blend", "step_4_stick_animation.blend"),
        cylinder_radius,
        cylinder_overlap,
        config["bone_definitions"]
    )

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    blender_executable = r"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe"

    if '--background' not in sys.argv:
        input_json, output_blend, _, _, _ = load_config()
        json_file_path = os.path.join(current_directory, input_json)
        blend_file_path = os.path.join(current_directory, output_blend)
        base_blend_path = os.path.join(current_directory, BASE_BLEND_FILE)

        if not os.path.exists(blender_executable):
            print(f"Error: Blender executable not found at '{blender_executable}'")
            sys.exit(1)

        if not os.path.exists(base_blend_path):
            print(f"Error: Base blend file not found at '{base_blend_path}'")
            sys.exit(1)

        print(f"Creating animation for {os.path.basename(json_file_path)}")

        # Copy the base blend file to our output location
        shutil.copyfile(base_blend_path, blend_file_path)

        # Run Blender with our script
        command = [
            blender_executable,
            "--background",
            blend_file_path,
            "--python",
            os.path.join(current_directory, "show_stick_animation_inner.py"),
            "--",
            json_file_path
        ]

        try:
            process = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            print(f"Error running Blender: {error}")
