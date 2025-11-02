Mocap to Kid Figure Animation With Editing (4K_kid)

Introduction

This program, 4K_kid, is part of a larger project, a pipeline designed to convert motion capture (mocap) data, specifically from a MediaPipe Pose estimation output, into a fully animated and editable rigged characters within a Blender .blend file.
The pipeline automates the process of generating a visually representative animation that can then be manually refined by an animator. The project is currently implementing a **"Fleshy Exoskeleton"** strategy to convert the initial stick figures into **organic, child-like characters** using multiple low-poly spheres along the limbs and torso.
In order to visualize and edit the MediaPipe Pose estimation output, 4K_kid creates a kid figure in Blender and provides a user interface, editor_addon_mocap.py, as a Blender add-on to allow some minor editing functions.
The core functionality revolves around a set of Python scripts and a configuration file that work together to:

Read JSON data containing 3D landmark coordinates from a MediaPipe video analysis.
Generate a 3D kid figure using geometric primitives (cylinders for bones, and now multiple tapered shapes along the limbs for a fleshy contour).
Keyframe the figure's position and rotation for every frame, creating a smooth animation.
Save the final animation as a .blend file, ready for editing in Blender.

**SUCCESSFUL IMPLEMENTATION UPDATE**: The 4K_kid_inner.py script has been successfully updated to create a complete rig system where the entire character moves in response to MediaPipe landmark data. The implementation now properly handles:
- Virtual spine and neck bones calculated from landmark midpoints
- Multi-sphere fleshy exoskeleton that deforms organically with limb movements
- Complete hierarchical bone structure that maintains proper proportions
- Smooth animation driven entirely by the original 33 MediaPipe landmarks

File Descriptions

=== 4K_kid.py ===

This is the main entry point for the program.
It is a standalone Python script that you run from the command line.
Its primary responsibilities are:

* Reading the configuration from 4K_kid_config.json.
* Setting up the necessary file paths using utils.py.
* Copying a base Blender file to the output location.
* Executing Blender in a non-graphical "background" mode, passing 4K_kid_inner.py as a Python script to be run inside the Blender environment.

=== 4K_kid_inner.py ===

This is the heart of the project's logic.
This script is designed to be executed by Blender's Python interpreter. It cannot be run on its own.
Its functions include:

* Parsing the command-line arguments passed from 4K_kid.py to locate the data and config files.
* Loading the motion capture data from the MediaPipe JSON output file.
The landmark data is based on the MediaPipe Pose API, which provides 33 3D landmarks for the human body (see official documentation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).
* Clearing the scene in the Blender file to ensure a clean slate.
* Creating all the 3D objects that make up the main kid figure and the ghost figures, based on the bone definitions in the config file. This includes creating the **virtual spine and neck bones** from landmark midpoints, and generating the **multi-sphere fleshy exoskeleton** on the limbs, head, and torso.
* Assigning materials and colors to the objects. Note that a recent fix was implemented to ensure the ghost figures receive their correct colors and transparency as defined in the config file.
* Creating a video plane in the background for visual reference.
* Animating the main kid figure and its ghosts by iterating through the JSON data and applying transformations (location, rotation, scale) for each frame. All motion is driven by the original 33 MediaPipe landmarks.
* **IMPLEMENTATION SUCCESS**: The entire rig system now moves cohesively, with the fleshy exoskeleton spheres properly following their respective bone movements, creating a natural, organic child-like character animation.
* Saving the modified .blend file upon completion.

=== 4K_kid_config.json ===

This JSON file is a single point of truth for customizing the animation.
It contains various settings that control the look and behavior of the animation, including:

* shapes: This setting allows users to control the contour of the figure. 
* bone_definitions: A critical section that maps the bones of the kid figure to the specific landmarks from the MediaPipe data (e.g., LEFT_SHOULDER, RIGHT_ELBOW) and defines the virtual bones (Spine, Neck).

=== Other Dependencies ===
* **4K_kid_config.json**: Defines the body shapes, sphere configurations (e.g., 4 spheres for forearms, 5 for thighs), and material settings for the fleshy exoskeleton.
* **utils.py**: Project utilities for file paths and logging, including the `script_log(msg)` function required for debugging output inside the Blender environment.

PROGRAM STATUS: FULLY IMPLEMENTED AND OPERATIONAL
The 4K_kid pipeline is now successfully converting MediaPipe motion capture data into fully animated, organic child-like characters with a working fleshy exoskeleton system. The entire rig moves in perfect synchronization with the input landmark data, creating smooth, natural animations ready for further refinement in Blender.
