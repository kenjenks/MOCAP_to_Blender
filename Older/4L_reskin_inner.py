# 4L_reskin_inner.py (Version 6.4 - Anatomical Ordering for Better Bridging)

import bpy
import bmesh
import sys
import os
import argparse
import json
from mathutils import Vector, Matrix
import math


# Parse arguments first so we can import utils
def parse_arguments():
    """Parse command line arguments passed from the controller script"""
    parser = argparse.ArgumentParser(description='4L Reskin Inner Script')
    parser.add_argument('--project-root', required=True, help='Path to project root')
    parser.add_argument('--show', required=True, help='Show name')
    parser.add_argument('--scene', required=True, help='Scene name')
    parser.add_argument('--config-file-path', required=True, help='Path to reskin config file')

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    return parser.parse_args(argv)


# Parse arguments early so we can set up the path
g_args = parse_arguments()

# Add project_root to sys.path so we can import utils
if g_args.project_root not in sys.path:
    sys.path.append(g_args.project_root)

# Import project utilities
try:
    from utils import script_log, comment, get_scene_config, get_processing_step_paths, get_scene_paths
except ImportError as e:
    print(f"FAILED to import utils: {e}")
    sys.exit(1)

# Globals for configuration
g_config = None


##########################################################################################

def load_config_and_data(config_file_path):
    """Loads configuration data from the specified JSON file."""
    global g_config
    try:
        # Load the entire config file, not just reskin_parameters
        with open(config_file_path, 'r') as f:
            full_config = json.load(f)

        # Get the reskin_parameters section
        g_config = full_config.get("reskin_parameters", {})

        # DEBUG: Log what we found in the config
        script_log(f"Loaded reskin config from: {config_file_path}")
        script_log(f"Config keys found: {list(g_config.keys())}")

        if "deformation_bones" in g_config:
            script_log(f"Found deformation_bones: {len(g_config['deformation_bones'])} bones")
        else:
            script_log("WARNING: deformation_bones not found in config")

        if "scene_objects" in g_config:
            script_log(f"Scene objects: {g_config['scene_objects']}")

    except Exception as e:
        script_log(f"ERROR: Failed to load config file {config_file_path}: {e}")
        sys.exit(1)


def get_object_by_name(name):
    """Safely retrieves an object by name."""
    obj = bpy.data.objects.get(name)
    if not obj:
        script_log(f"ERROR: Required object '{name}' not found in scene.")
    return obj


def save_blender_file():
    """Saves the current blend file."""
    try:
        bpy.ops.wm.save_mainfile()
        script_log("✓ Saved updated blend file.")
    except Exception as e:
        script_log(f"Error saving Blender file: {e}")


def select_object(obj):
    """Selects a single object and makes it active."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def clean_up_old_mesh(new_mesh_name):
    """Delete any existing final new mesh to ensure a clean start."""
    if new_mesh_name in bpy.data.objects:
        new_mesh = bpy.data.objects[new_mesh_name]
        if new_mesh.users_collection:
            for collection in new_mesh.users_collection:
                collection.objects.unlink(new_mesh)
        bpy.data.objects.remove(new_mesh, do_unlink=True)
        script_log(f"Removed existing final mesh: {new_mesh_name}")


def cleanup_intermediate_objects():
    """Remove intermediate component objects"""
    objects_to_remove = [obj for obj in bpy.data.objects if obj.name.startswith("Component_")]
    for obj in objects_to_remove:
        if obj.users_collection:
            for collection in obj.users_collection:
                collection.objects.unlink(obj)
        bpy.data.objects.remove(obj, do_unlink=True)
    if objects_to_remove:
        script_log(f"✓ Cleaned up {len(objects_to_remove)} intermediate objects")


def validate_old_mesh_for_weights(old_mesh):
    """Validate that the old mesh has the required DEF_ vertex groups for weight transfer."""
    script_log(f"=== VALIDATING VERTEX GROUPS ON {old_mesh.name} ===")

    def_groups = []
    all_groups = []

    # Check if mesh has any vertex groups at all
    if not old_mesh.vertex_groups:
        script_log("❌ CRITICAL: Mesh has NO vertex groups at all!")
        return False

    # Analyze all vertex groups
    for vg in old_mesh.vertex_groups:
        all_groups.append(vg.name)
        if vg.name.startswith('DEF_'):
            def_groups.append(vg.name)

    # Log detailed findings
    script_log(f"Total vertex groups found: {len(all_groups)}")
    script_log(f"DEF_ vertex groups found: {len(def_groups)}")

    if def_groups:
        script_log("✓ DEF_ groups present (first 10 shown):")
        for i, group in enumerate(sorted(def_groups)[:10]):
            script_log(f"  {i + 1:2d}. {group}")
    else:
        script_log("❌ CRITICAL: No DEF_ vertex groups found!")

    validation_passed = len(def_groups) > 0
    if validation_passed:
        script_log("✓ Vertex group validation PASSED")
    else:
        script_log("❌ Vertex group validation FAILED")

    return validation_passed


def get_anatomical_bone_order(bone_names):
    """Organize bones in logical anatomical order for better bridging"""
    # Define the processing order - limbs first, then core
    anatomical_order = [
        # Left leg (bottom to top)
        'DEF_LeftFoot', 'DEF_LeftShin', 'DEF_LeftThigh', 'DEF_LeftHip',
        # Right leg (bottom to top)
        'DEF_RightFoot', 'DEF_RightShin', 'DEF_RightThigh', 'DEF_RightHip',
        # Left arm (hand to shoulder)
        'DEF_LeftHand', 'DEF_LeftForearm', 'DEF_LeftUpperArm', 'DEF_LeftShoulder',
        # Right arm (hand to shoulder)
        'DEF_RightHand', 'DEF_RightForearm', 'DEF_RightUpperArm', 'DEF_RightShoulder',
        # Spine (head down to hips)
        'DEF_Head', 'DEF_Neck', 'DEF_UpperSpine', 'DEF_LowerSpine', 'DEF_HipRoot'
    ]

    # Filter to only include bones that actually exist in our config
    ordered_bones = [bone for bone in anatomical_order if bone in bone_names]

    # Add any missing bones to the end (in case we have extras)
    missing_bones = [bone for bone in bone_names if bone not in ordered_bones]
    ordered_bones.extend(missing_bones)

    return ordered_bones


def isolate_component(old_mesh, bone_name):
    """
    Duplicates the mesh, isolates geometry based on the bone's vertex group,
    keeping ALL vertices that have ANY weight for this bone.
    """
    # Check if the bone_name exists on the original mesh first
    if not old_mesh.vertex_groups.get(bone_name):
        script_log(f"WARNING: Vertex group '{bone_name}' not found on old mesh. Skipping component.")
        return None

    # 1. Duplicate Mesh
    component_mesh = old_mesh.copy()
    component_mesh.data = old_mesh.data.copy()
    component_mesh.name = f"Component_{bone_name}"
    bpy.context.collection.objects.link(component_mesh)

    script_log(f"--- Isolating {bone_name} geometry...")

    # 2. Isolate geometry using the vertex group - KEEP OVERLAP
    select_object(component_mesh)

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='VERT')

        # Get the vertex group from the component mesh
        component_vg = component_mesh.vertex_groups.get(bone_name)
        if not component_vg:
            script_log(f"ERROR: Vertex group '{bone_name}' not found on component mesh")
            raise Exception(f"Vertex group '{bone_name}' missing on component")

        # METHOD 1: Select vertices with ANY weight (not just primary)
        bpy.ops.mesh.select_all(action='DESELECT')

        # Select by vertex group with threshold 0.0 to get ALL influenced vertices
        bpy.ops.object.vertex_group_set_active(group=component_vg.name)
        bpy.ops.object.vertex_group_select()

        # Now invert and delete vertices that have ZERO weight for this bone
        bpy.ops.mesh.select_all(action='INVERT')
        bpy.ops.mesh.delete(type='VERT')

    except Exception as e:
        script_log(f"ERROR: Failed during isolation of {bone_name}: {e}")
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.data.objects.remove(component_mesh, do_unlink=True)
        return None
    finally:
        bpy.ops.object.mode_set(mode='OBJECT')

    script_log(f"    ✓ Isolation complete for {bone_name} (kept overlapping vertices)")
    return component_mesh


def remesh_component(component, settings, bone_name):
    """Applies QuadriFlow remesh to a component mesh with optimized face counts."""

    select_object(component)

    # Smart face count based on component type for better performance
    if any(x in bone_name for x in ['Hip', 'Spine', 'Thigh', 'Torso']):
        target_faces = 4000  # Large components
    elif any(x in bone_name for x in ['UpperArm', 'Head', 'Shin']):
        target_faces = 2000  # Medium components
    elif any(x in bone_name for x in ['Forearm', 'Hand', 'Foot', 'Shoulder']):
        target_faces = 800  # Small components
    else:
        target_faces = 500  # Default for small parts

    script_log(f"    - Applying QuadriFlow Remesh (Target: {target_faces} faces for {bone_name})...")

    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        # USE ONLY THE WORKING PARAMETER - target_faces
        bpy.ops.object.quadriflow_remesh(target_faces=target_faces)

    except Exception as e:
        script_log(f"    ERROR: QuadriFlow remesh failed on {component.name}: {e}. Component failed.")
        bpy.data.objects.remove(component, do_unlink=True)
        return None

    script_log(f"    ✓ Remesh complete for {component.name}")
    return component


def create_component_based_reskin(config):
    """
    Main function for the component-based reskinning workflow with anatomical ordering.
    """
    scene_objects = config.get("scene_objects", {})
    old_mesh_name = scene_objects.get("old_mesh_name")
    new_mesh_name = scene_objects.get("new_mesh_name")
    rig_name = scene_objects.get("rig_name")

    bone_names = config.get("deformation_bones", [])
    settings = config.get("retopology_settings", {})
    smoothing_settings = config.get("smoothing_settings", {})

    script_log(f"DEBUG: Found {len(bone_names)} deformation bones")

    if not bone_names:
        script_log("CRITICAL ERROR: 'deformation_bones' list is empty or missing from config.")
        return None

    old_mesh = get_object_by_name(old_mesh_name)
    if not old_mesh:
        script_log(f"CRITICAL ERROR: Old mesh '{old_mesh_name}' not found.")
        return None

    # Validate old mesh has required vertex groups
    if not validate_old_mesh_for_weights(old_mesh):
        script_log("❌ ABORTING: Old mesh missing required DEF_ vertex groups for weight transfer")
        return None

    # Clean up previous final mesh and intermediate objects
    cleanup_intermediate_objects()
    clean_up_old_mesh(new_mesh_name)

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # STEP 1: Get bones in anatomical order
    ordered_bones = get_anatomical_bone_order(bone_names)
    script_log(f"Processing bones in anatomical order: {ordered_bones}")

    # STEP 2: Process and join components sequentially
    script_log("--- PROCESSING COMPONENTS IN ANATOMICAL ORDER ---")

    final_mesh = None
    processed_components = 0

    for i, bone_name in enumerate(ordered_bones):
        script_log(f"--- Processing {bone_name} ({i + 1}/{len(ordered_bones)}) ---")

        # Quick pre-check
        if not old_mesh.vertex_groups.get(bone_name):
            script_log(f"WARNING: Vertex group '{bone_name}' not found. Skipping.")
            continue

        # Isolate component
        component = isolate_component(old_mesh, bone_name)
        if not component:
            continue

        # Remesh component
        remeshed_component = remesh_component(component, settings, bone_name)
        if not remeshed_component:
            continue

        # Join with previous components
        if final_mesh is None:
            # First component becomes the base
            final_mesh = remeshed_component
            final_mesh.name = new_mesh_name
            script_log(f"    Started with {bone_name} as base")
        else:
            # Join new component with existing mesh
            select_object(final_mesh)
            remeshed_component.select_set(True)
            bpy.context.view_layer.objects.active = final_mesh

            try:
                bpy.ops.object.join()
                script_log(f"    Joined {bone_name} with existing mesh")

                # Quick merge after each join to stitch seams
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles(threshold=0.002)  # Moderate merging
                bpy.ops.object.mode_set(mode='OBJECT')

            except Exception as e:
                script_log(f"ERROR: Failed to join {bone_name}: {e}")
                continue

        processed_components += 1

    if not final_mesh:
        script_log("CRITICAL ERROR: No valid components were generated.")
        return None

    script_log(f"Successfully processed {processed_components}/{len(ordered_bones)} components in anatomical order")

    # STEP 3: Final seam stitching
    script_log("--- FINAL SEAM STITCHING ---")
    select_object(final_mesh)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # More aggressive final merge
    bpy.ops.mesh.remove_doubles(threshold=0.005)

    # Try to fill any remaining holes
    try:
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bridge_edge_loops()
        script_log("    - Applied final edge bridging")
    except:
        script_log("    - Final edge bridging not available")

    bpy.ops.object.mode_set(mode='OBJECT')

    # STEP 4: Weight Transfer
    script_log("--- 4. Transferring Weights ---")

    try:
        select_object(final_mesh)
        old_mesh.select_set(True)

        # Use correct enum value for Blender 4.4
        bpy.ops.object.data_transfer(data_type='VGROUP_WEIGHTS')
        script_log("✓ Weight transfer completed")

    except Exception as e:
        script_log(f"⚠ Weight transfer failed: {e}")
        script_log("⚠ Mesh created but may need manual weight painting")
    finally:
        old_mesh.select_set(False)

    # STEP 5: Setup final mesh
    script_log("--- 5. Setting up final mesh ---")

    # Add armature modifier
    rig_obj = bpy.data.objects.get(rig_name)
    if rig_obj:
        armature_mod = final_mesh.modifiers.new(name='Armature', type='ARMATURE')
        armature_mod.object = rig_obj
        armature_mod.use_vertex_groups = True
        script_log(f"    - Added Armature modifier ({rig_name}).")
    else:
        script_log(f"WARNING: Rig '{rig_name}' not found.")

    # Add subdivision
    subdiv_mod = final_mesh.modifiers.new(name='Subdivision', type='SUBSURF')
    subdiv_mod.levels = smoothing_settings.get("subdiv_viewport_levels", 2)
    subdiv_mod.render_levels = smoothing_settings.get("subdiv_render_levels", 3)
    script_log("    - Added Subdivision Surface modifier.")

    # Shade smooth
    bpy.ops.object.shade_smooth()
    script_log("    - Applied Shade Smooth.")

    script_log("✓ Anatomical component-based reskinning completed successfully")
    return final_mesh


def main_execution():
    """Main execution for the reskin pipeline."""
    global g_config

    script_log("=== 4L RESKIN INNER SCRIPT STARTED ===")

    # Load configuration
    load_config_and_data(g_args.config_file_path)

    if not g_config:
        script_log("ERROR: Configuration failed to load. Aborting.")
        return

    # Create New_Skin_Auto via component-based reskinning
    script_log("=== STARTING COMPONENT-BASED RESKINNING ===")

    try:
        new_mesh = create_component_based_reskin(g_config)
    except Exception as e:
        script_log(f"FATAL ERROR during component reskinning: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        cleanup_intermediate_objects()
        return

    if new_mesh:
        script_log(f"✓ SUCCESS: Created {new_mesh.name} via component-based reskinning")

        # Final scene update and save
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.context.view_layer.update()
        save_blender_file()

        script_log("=== 4L RESKIN INNER SCRIPT COMPLETED SUCCESSFULLY ===")
    else:
        script_log("=== 4L RESKIN INNER SCRIPT COMPLETED WITH ERRORS ===")


try:
    # Run the main execution
    main_execution()
except Exception as e:
    script_log(f"FATAL ERROR in main execution: {e}")
    import traceback

    script_log(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)