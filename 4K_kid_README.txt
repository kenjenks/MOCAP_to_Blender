Mocap to Kid Figure Animation (4K_kid and 4K_kid_anim)

Introduction

This program, 4K_kid, is part of a larger project pipeline designed to convert motion capture (mocap) data, specifically from a MediaPipe Pose estimation output, into fully animated and editable rigged characters within Blender .blend files. The pipeline automates the process of generating visually representative animations that can then be manually refined by animators.

The system implements a sophisticated vertex cloud flesh system that creates organic, child-like characters using complete mesh generation with proper vertex group assignment and armature deformation. The system generates 1172+ vertices with systematic bridging between body segments for natural deformation.

Technical Architecture

The 4K Kid system uses a sophisticated coordinate frame system for natural character movement:

- **Coordinate Frame System**: HipRoot and ShoulderRoot bones use COPY_TRANSFORMS constraints to follow VIRTUAL_HIP_FRAME and VIRTUAL_SHOULDER_FRAME, providing both position AND rotation from landmark data
- **Two-Segment Spine**: LowerSpine and UpperSpine bones with proper hierarchy-driven constraints
- **Vertex Cloud Flesh**: Complete mesh generation with 1172+ vertices, systematic spine-hip bridging, and proper vertex group assignment
- **Constraint-Based Animation**: STRETCH_TO constraints for limbs, COPY_TRANSFORMS for root bones, and DAMPED_TRACK for head rotation
- **Modular Pipeline**: Separate character creation and animation workflows for maximum flexibility

Core functionality includes:

- Reading JSON data containing 3D landmark coordinates from MediaPipe video analysis
- Generating 3D kid figures using armature-based rigging with vertex cloud flesh
- Creating complete mesh systems with proper weight painting and deformation
- Keyframing character movement for smooth animation
- Saving characters as .blend files ready for editing or animation in Blender

Once the figure is created and saved as a .blend file, the 4K_kid_anim program animates the figure using new mocap data from MediaPipe streams, transformed to Blender's +Z=up coordinate system.

IMPLEMENTATION

The 4K_kid_inner.py script creates a complete rig system where the entire character responds to MediaPipe landmark data:
- Coordinate frame system for natural hip and shoulder rotation
- Two-segment spine with hierarchy-driven constraints
- Complete vertex cloud flesh with systematic bridging
- Programmatic root bones (HipRoot and ShoulderRoot) for stable deformation

The 4K_kid_anim_inner.py script animates existing character rigs with new motion data:
- Clears existing animation before applying new motion
- Animates both direct control points (CTRL_*) and virtual control points (VIRTUAL_*)
- Applies squish factors from configuration
- Forces constraint solving across all frames
- Verifies animation by testing movement of key control points

Workflow

1. **Character Creation** (4K_kid.py):
   - Creates armature with coordinate frame system
   - Generates vertex cloud flesh mesh with 1172+ vertices
   - Sets up all constraints and vertex groups
   - Saves as Scene-XXXX_kid.blend

2. **Character Modification**:
   - Artists can modify mesh, materials, or proportions in Blender
   - Armature structure and constraints remain intact
   - Vertex groups preserve deformation relationships

3. **Animation** (4K_kid_anim.py):
   - Loads existing character rig
   - Applies new mocap data to control points
   - Maintains all artistic modifications
   - Forces constraint solving for proper deformation
   - Saves as Scene-XXXX_anim.blend

Key Benefits:
- Same character can be animated with different mocap sequences
- Supports iterative refinement workflow
- Preserves artist modifications between animations
- Complete vertex group assignment ensures proper mesh deformation

File Descriptions

=== 4K_kid.py ===

This is the main entry point for character creation.
It is a standalone Python script that runs from the command line.

Primary responsibilities:
* Reading configuration from 4K_kid_config.json
* Setting up necessary file paths using utils.py
* Copying a base Blender file to the output location
* Executing Blender in non-graphical "background" mode, passing 4K_kid_inner.py as a Python script to be run inside Blender

=== 4K_kid_inner.py ===

This is the heart of the character creation logic, executed by Blender's Python interpreter.

Key functions include:
* Parsing command-line arguments passed from 4K_kid.py
* Loading motion capture data from MediaPipe JSON output (33 landmarks)
* Creating complete armature system with coordinate frame architecture
* Generating vertex cloud flesh mesh with 1172+ vertices
* Implementing systematic spine-hip bridging with vertex subdivision
* Setting up bone constraints (STRETCH_TO, COPY_TRANSFORMS, DAMPED_TRACK)
* Assigning vertex groups for proper mesh deformation
* Creating virtual control points for spine and head positioning
* Saving the complete character as a .blend file

=== 4K_kid_anim.py ===

This is the entry point for the animation program.
It is a standalone Python script that runs from the command line.

Primary responsibilities:
* Reading configuration from 4K_kid_config.json
* Setting up necessary file paths using utils.py
* Loading existing character .blend files
* Executing Blender in background mode with 4K_kid_anim_inner.py
* Applying new mocap data to pre-built character rigs

=== 4K_kid_anim_inner.py ===

This script handles animation logic inside Blender's Python environment.

Key functions include:
* Parsing command-line arguments from 4K_kid_anim.py
* Loading new motion capture data from MediaPipe JSON files
* Clearing existing animation data from the scene
* Finding existing armatures and control points
* Animating both direct and virtual control points
* Applying squish factors to landmark positions
* Forcing constraint solving for proper spine behavior
* Verifying animation by testing control point movement
* Saving the animated scene as a new .blend file

=== 4K_kid_config.json ===

This JSON file is the single point of truth for customizing character appearance and behavior.

Contains settings for:
* bone_definitions: Maps character bones to MediaPipe landmarks and defines virtual bones
* kid_figure_settings: Controls squish factors (x_squish_fraction, y_squish_fraction, z_squish_fraction)
* shapes: Defines vertex ring configurations for each bone (position, radius, flatten_scale)

=== Other Dependencies ===
* **utils.py**: Project utilities for file paths and logging, including the `script_log(msg)` function required for debugging output inside Blender

Technical Notes

- **Vertex System**: Generates ~1172 vertices with complete weight painting and systematic bridging
- **Spine-Hip Bridge**: Uses LCM-based subdivision for perfect vertex alignment between segments
- **Coordinate Frames**: VIRTUAL_HIP_FRAME and VIRTUAL_SHOULDER_FRAME calculate proper rotation from landmark data
- **Constraint Architecture**: STRETCH_TO for limbs, COPY_TRANSFORMS for roots, DAMPED_TRACK for head rotation
- **Modular Output**: Creates both _kid.blend (character rig) and _anim.blend (animated scene) files
- **Deformation Quality**: Complete vertex group assignment ensures all vertices deform properly with armature

PROGRAM STATUS: PRODUCTION READY

The 4K Kid pipeline successfully converts MediaPipe motion capture data into fully animated characters with a complete vertex cloud system. The modular architecture supports both character creation and animation workflows, providing artists with flexible tools for character animation and refinement.

The system handles complete mesh generation, proper constraint setup, and reliable animation application, making it suitable for production animation pipelines.
