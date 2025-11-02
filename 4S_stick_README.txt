Mocap to Stick Figure Animation With Editing (4S_stick)

Introduction

This program, 4S_stick, is part of a larger project, a pipeline designed to convert motion capture (mocap) data, specifically from a MediaPipe Pose estimation output, into a fully animated and editable rigged characters within a Blender .blend file. The pipeline automates the process of generating a visually representative animation that can then be manually refined by an animator.

In order to visualize and edit the MediaPipe Pose estimation output, 4S_stick creates a stick figure in Blender and provides a user interface, editor_addon_mocap.py, as a Blender add-on to allow some minor editing functions.

The core functionality revolves around a set of Python scripts and a configuration file that work together to:

Read JSON data containing 3D landmark coordinates from a MediaPipe video analysis.

Generate a 3D stick figure using geometric primitives (cylinders for bones, spheres for the head).

Create "ghost" figures to visualize the motion trail of the character's past and future movements.

Keyframe the figure's position and rotation for every frame, creating a smooth animation.

Save the final animation as a .blend file, ready for editing in Blender.

File Descriptions

=== edit_stick_animation.py ===

This is the main entry point for the program. It is a standalone Python script that you run from the command line. Its primary responsibilities are:

* Reading the configuration from 4S_stick_config.json.

* Setting up the necessary file paths using utils.py.

* Copying a base Blender file to the output location.

* Executing Blender in a non-graphical "background" mode, passing 4S_stick_inner.py as a Python script to be run inside the Blender environment.

=== 4S_stick_inner.py ===

This is the heart of the project's logic. This script is designed to be executed by Blender's Python interpreter. It cannot be run on its own. Its functions include:

* Parsing the command-line arguments passed from 4S_stick.py to locate the data and config files.

* Loading the motion capture data from the MediaPipe JSON output file. The landmark data is based on the MediaPipe Pose API, which provides 33 3D landmarks for the human body (see official documentation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker).

* Clearing the scene in the Blender file to ensure a clean slate.

* Creating all the 3D objects that make up the main stick figure and the ghost figures, based on the bone definitions in the config file.

* Assigning materials and colors to the objects. Note that a recent fix was implemented to ensure the ghost figures receive their correct colors and transparency as defined in the config file.

* Creating a video plane in the background for visual reference.

* Animating the main stick figure and its ghosts by iterating through the JSON data and applying transformations (location, rotation, scale) for each frame.

* Saving the modified .blend file upon completion.

=== 4S_stick_config.json ===

This JSON file is a single point of truth for customizing the animation. It contains various settings that control the look and behavior of the animation, including:

* cylinder_settings: Adjusts the radius of the bone cylinders.

* stick_figure_settings: Sets the color and other properties of the main figure.

* head_settings: Controls the head sphere's visibility, size, and color.

* bone_definitions: A critical section that maps the bones of the stick figure to the specific landmarks from the MediaPipe data (e.g., LEFT_SHOULDER, RIGHT_ELBOW).

* ghost_settings: Controls the visibility, number, and colors of the "ghost" figures that represent the motion trail.

=== editor_addon_mocap.py ===

This script is a Blender add-on intended for an animator's use after the initial animation has been generated. It adds a "Mocap" panel to Blender's user interface, providing tools to manually edit and clean up the imported motion capture data. Its purpose is to save time for an animator by allowing them to quickly delete a bad frame or interpolate a missing one without having to manipulate the keyframes manually.
