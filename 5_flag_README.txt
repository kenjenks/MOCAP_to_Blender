README: 5_flag.py and 5_flag_inner.py
=================================================

Purpose

The 5_flag.py script is the final step in a motion capture pipeline. Its primary purpose is to automate the animation of a 3D flag model in Blender. It reads 2D coordinate data (XY) and rotation data from the flag_coords.json file, and uses this information to animate a pre-existing 3D flag object. The final output is a new Blender file (step_5_flag.blend) with the animated flag.

===

How it Works

The process is divided into two parts: an outer Python script (5_flag.py) and an inner Blender-specific Python script (5_flag_inner.py).

---

5_flag.py (The Orchestrator):

This script acts as a wrapper that prepares the environment for Blender.

It first copies the base 3D flag model (Flag.blend) to a new file named step_5_flag.blend to avoid modifying the original.

It then launches Blender in the background using a command-line interface.

The command tells Blender to open the newly copied file and execute 5_flag_inner.py as a Python script.

---

5_flag_inner.py (The Blender Script):

This script runs directly within the Blender environment.

It reads the flag_coords.json file to access the frame-by-frame coordinate and rotation data.

For each frame, it uses the Blender Python API (bpy) to:

* Find the flag object(s) within the scene.

* Set the object's 3D location and rotation based on the data from the JSON file.

* Insert a keyframe for the object's location and rotation.

* Add a plane containing a video clip, synched to the animation.

After processing all frames, the script saves the animated Blender file.

===

Prerequisites

Blender must be installed on your system.

The path to the Blender executable must be correctly specified in the 5_flag.py script.

A base flag model file, Flag.blend, must be present in the expected location.

The data from the motion capture pipeline must be saved in a JSON file named flag_coords.json in the specified directory.

===

Usage

To run the animation script, execute 5_flag.py in PyCharm or from your terminal:

python 5_flag.py

The script will print progress to the console. Once complete, you can open step_5_flag.blend in Blender to view the animated flag.
