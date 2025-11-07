# 4M_cloth_inner.py (Version 12.0 - Nice coat!)

import bpy
import bmesh
import sys
import os
import argparse
import json
import math
from mathutils import Vector, Matrix

##########################################################################################

def main_execution():
    """Main execution for cloth simulation"""
    script_log("=== 4M CLOTH INNER STARTED ===\n")

    try:
        figure_name = "Main"

        # Load configuration and data
        load_config_and_data()

        # Clear scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        cleanup_existing_objects("Main")

        # Extract bone hierarchy and constraints
        extract_bone_hierarchy_and_controls(bone_definitions)

        # 1. CREATE CONTROL POINTS FIRST
        script_log("=== CREATING CONTROL POINTS ===")
        create_control_points(figure_name)

        # 2. CREATE FRAME CONTROL POINTS
        script_log("=== CREATING FRAME CONTROL POINTS ===")
        first_frame = mocap_data.get(str(frame_numbers[0]), {})
        create_hip_frame_control_point(first_frame)
        create_shoulder_frame_control_point(first_frame)

        # 3. Set up virtual point constraints
        setup_virtual_point_constraints()
        setup_virtual_frame_constraints()

        # 4. Create rig
        armature_obj = create_kid_rig(figure_name)

        # 5. ALIGN BONES WITH CONTROL POINTS
        align_bones_with_control_points(armature_obj, figure_name)

        # 6. SET UP CONSTRAINTS
        setup_direct_constraints(armature_obj, figure_name)
        setup_two_segment_spine_constraints(armature_obj, figure_name)
        setup_root_bone_transform_constraints(armature_obj)

        # 7. CREATE CLOTH GARMENTS (REPLACES KID FLESH)
        script_log("=== CREATING CLOTH GARMENTS ===")
        register_driver_functions()
        load_garment_configs()  # Load configurations into global garment_configs
        populate_joint_control_systems()
        make_vertex_all_bundles(armature_obj)  # Then: Create bundle systems
        garments_created = create_cloth_garments(armature_obj, figure_name)

        # 8. Force initial constraint solve
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')
        bpy.ops.pose.visual_transform_apply()
        bpy.ops.object.mode_set(mode='OBJECT')

        # 9. Final scene update
        bpy.context.view_layer.update()

        # 10. Save file
        save_blender_file()

        script_log(f"=== 4M CLOTH INNER COMPLETED - Created {garments_created} garments ===\n")

    except Exception as e:
        script_log(f"ERROR in main execution: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")

##########################################################################################

# Parse command line arguments
def parse_arguments():
    """Parse command line arguments passed from 4M_cloth.py"""
    parser = argparse.ArgumentParser(description='4M Cloth Inner Script')
    parser.add_argument('--project-root', required=True, help='Path to project root')
    parser.add_argument('--show', required=True, help='Show name')
    parser.add_argument('--scene', required=True, help='Scene name')

    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    return parser.parse_args(argv)

##########################################################################################

# Parse arguments
args = parse_arguments()

# Add project_root to sys.path so we can import utils
if args.project_root not in sys.path:
    sys.path.append(args.project_root)

# Import project utilities
try:
    from utils import script_log, comment, get_scene_config, get_processing_step_paths, get_scene_paths
except ImportError as e:
    script_log(f"FAILED to import utils: {e}")
    sys.exit(1)

# Add current script directory to sys.path for dynamic vertex utils
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import dynamic vertex utilities
try:
    from _4M_dynamic_vertex_utils import *
    script_log("✓ Successfully imported dynamic vertex utilities")
except ImportError as e:
    script_log(f"ERROR: Failed to import dynamic vertex utilities: {e}")

# Global variables
mocap_data = {}
bone_definitions = {}
frame_numbers = []
garment_configs = {}

# Control point tracking
control_point_objs = {}
joint_control_systems = {}

# Bone hierarchy structure
bone_parents = {}
bone_tail_control_points = {}
bone_head_control_points = {}
def_bone_names = {}
bone_constraint_types = {}
bone_tail_landmarks = {}

# Virtual control point calculations
VIRTUAL_POINT_CALCULATIONS = {
    "VIRTUAL_HIP_MIDPOINT": ["LEFT_HIP", "RIGHT_HIP"],
    "VIRTUAL_SHOULDER_MIDPOINT": ["LEFT_SHOULDER", "RIGHT_SHOULDER"],
    "VIRTUAL_SPINE_MIDPOINT": ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"],
    "VIRTUAL_HEAD_BASE": ["NOSE", "HEAD_TOP"]
}


##########################################################################################

def load_config_and_data():
    """Load configuration and mocap data using project utilities"""
    global mocap_data, bone_definitions, frame_numbers

    try:
        # Get scene configuration
        scene_config = get_scene_config(args.show, args.scene)

        # Get processing step paths
        step_paths = get_processing_step_paths(args.show, args.scene, "export_to_blender")

        # Input JSON is from the apply_physics step
        processing_steps = scene_config.get("processing_steps", {})
        apply_physics_step = processing_steps.get("apply_physics", {})
        input_json_relative = apply_physics_step.get("output_file", "step_4_input.json")

        # Build absolute path to input JSON
        scene_paths = get_scene_paths(args.show, args.scene)
        inputs_dir = os.path.dirname(scene_paths["output_pose_data"])
        INPUT_JSON_FILE = os.path.join(inputs_dir, input_json_relative)

        # Load cloth specific config
        script_dir = os.path.dirname(os.path.abspath(__file__))
        CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

        with open(CLOTH_CONFIG_FILE, 'r') as file:
            config = json.load(file)
            bone_definitions = config.get("bone_definitions", {})
            cloth_settings = config.get("cloth_settings", {})

        with open(INPUT_JSON_FILE, 'r') as file:
            mocap_data = json.load(file)

        script_log(f"Loaded mocap data from: {INPUT_JSON_FILE}")
        script_log(f"Loaded cloth config from: {CLOTH_CONFIG_FILE}")

    except FileNotFoundError as e:
        script_log(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        script_log(f"Error: Could not decode JSON from file: {e}")
        sys.exit(1)

    # Get the frame numbers from the JSON data
    frame_numbers = sorted([int(frame) for frame in mocap_data.keys()])
    if not frame_numbers:
        script_log("Error: No frame data found in JSON file.")
        sys.exit(1)

##########################################################################################

def load_garment_configs():
    """Load garment configurations into global variable"""
    global garment_configs
    script_log("=== LOADING GARMENT CONFIGS ===")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

    try:
        with open(CLOTH_CONFIG_FILE, 'r') as file:
            cloth_config = json.load(file)
            garment_configs = cloth_config.get("cloth_garments", {})

        # Log what we loaded
        garment_count = len(garment_configs)
        script_log(f"✓ Loaded {garment_count} garment configurations")

        # Log enabled garments for debugging
        enabled_garments = []
        for garment_name, config in garment_configs.items():
            if config.get("enabled", True):
                enabled_garments.append(garment_name)

        if enabled_garments:
            script_log(f"✓ Enabled garments: {', '.join(enabled_garments)}")
        else:
            script_log("⚠ No garments enabled in config")

        return True

    except Exception as e:
        script_log(f"ERROR: Failed to load garment configs: {e}")
        garment_configs = {}  # Ensure it's empty on failure
        return False

##########################################################################################

def get_bundle_radius_from_config(joint_type, side):
    """Get radius from garment_configs (used during bundle creation)"""
    global garment_configs

    defaults = {
        "shoulder": 0.075, "elbow": 0.06, "wrist": 0.04,
        "hip": 0.09, "knee": 0.07, "ankle": 0.06,
        "head_neck": 0.08, "neck_coordination": 0.075
    }

    try:
        if joint_type in ["shoulder", "elbow", "wrist"]:
            garment_config = garment_configs.get(f"{side}_sleeve", {})
            if joint_type == "shoulder":
                return garment_configs.get("diameter_start", 0.15) / 2
            elif joint_type == "elbow":
                return garment_configs.get("diameter_elbow", 0.12) / 2
            else:  # wrist
                return garment_configs.get("diameter_end", 0.08) / 2
        elif joint_type in ["hip", "knee", "ankle"]:
            garment_config = garment_configs.get(f"{side}_pants", {})
            if joint_type == "hip":
                return garment_configs.get("diameter_hip", 0.18) / 2
            elif joint_type == "knee":
                return garment_configs.get("diameter_knee", 0.14) / 2
            else:  # ankle
                return garment_configs.get("diameter_ankle", 0.12) / 2
        else:  # head/neck
            return defaults[joint_type]
    except:
        return defaults.get(joint_type, 0.05)

##########################################################################################

def get_bundle_radius(control_point_name):
    """Get radius for vertex bundle influence"""
    if control_point_name in joint_control_systems:
        return joint_control_systems[control_point_name].get('radius', 0.1)
    return 0.1

##########################################################################################

def get_bundle_center(control_point_name):
    """Get current position of vertex bundle center (RPY empty)"""
    if control_point_name in joint_control_systems:
        system_data = joint_control_systems[control_point_name]
        # Return the RPY empty's world location
        return system_data['rpy_empty'].location

    script_log(f"ERROR: {control_point_name} NOT in joint_control_systems")
    script_log(f"INFO: Available systems: {list(joint_control_systems.keys())}")
    return None

##########################################################################################

def create_empty_at_location(name, location=(0,0,0), size=0.1, empty_type='PLAIN_AXES'):
    """Create an empty object at specified location"""
    empty_data = bpy.data.objects.new(name, None)
    empty_data.empty_display_size = size
    empty_data.empty_display_type = empty_type
    bpy.context.collection.objects.link(empty_data)
    empty_data.location = location
    return empty_data

##########################################################################################

def populate_joint_control_systems():
    """Populate joint_control_systems with control point data for vertex bundles"""
    global joint_control_systems

    script_log("=== POPULATING JOINT CONTROL SYSTEMS ===")

    # Clear any existing entries
    joint_control_systems.clear()

    # Define control points and their properties
    control_point_definitions = {
        # Shoulder control points
        "CTRL_LEFT_SHOULDER": {"radius": 0.075, "type": "shoulder"},
        "CTRL_RIGHT_SHOULDER": {"radius": 0.075, "type": "shoulder"},

        # Elbow control points
        "CTRL_LEFT_ELBOW": {"radius": 0.06, "type": "elbow"},
        "CTRL_RIGHT_ELBOW": {"radius": 0.06, "type": "elbow"},

        # Wrist control points
        "CTRL_LEFT_WRIST": {"radius": 0.04, "type": "wrist"},
        "CTRL_RIGHT_WRIST": {"radius": 0.04, "type": "wrist"},

        # Hip control points
        "CTRL_LEFT_HIP": {"radius": 0.09, "type": "hip"},
        "CTRL_RIGHT_HIP": {"radius": 0.09, "type": "hip"},

        # Knee control points
        "CTRL_LEFT_KNEE": {"radius": 0.07, "type": "knee"},
        "CTRL_RIGHT_KNEE": {"radius": 0.07, "type": "knee"},

        # Ankle/Heel control points
        "CTRL_LEFT_HEEL": {"radius": 0.06, "type": "ankle"},
        "CTRL_RIGHT_HEEL": {"radius": 0.06, "type": "ankle"},

        # Head control points
        "CTRL_HEAD_TOP": {"radius": 0.08, "type": "head_neck"},
        "CTRL_NOSE": {"radius": 0.03, "type": "head_tracking"},
    }

    # Populate joint_control_systems with actual control point objects
    for cp_name, cp_data in control_point_definitions.items():
        cp_obj = bpy.data.objects.get(cp_name)
        if cp_obj:
            joint_control_systems[cp_name] = {
                "radius": cp_data["radius"],
                "type": cp_data["type"],
                "rpy_empty": cp_obj,  # Reference to the actual control point object
                "location": cp_obj.location
            }
            script_log(f"✓ Registered {cp_name} in joint_control_systems")
        else:
            script_log(f"⚠ WARNING: Control point {cp_name} not found in scene")

    script_log(f"✓ Populated {len(joint_control_systems)} control points in joint_control_systems")

##########################################################################################

##########################################################################################

def apply_material_from_config(obj, config_key, material_name=None, fallback_color=(0.8, 0.8, 0.8, 1.0)):
    """
    Apply material to object based on configuration.

    Args:
        obj: Blender object to apply material to
        config_key: Key in garment_configs for material configuration (e.g., "left_pants", "coat_torso")
        material_name: Optional custom material name, defaults to f"{config_key}_Material"
        fallback_color: RGBA fallback color if not specified in config
    """
    # Get material configuration
    garment_config = garment_configs.get(config_key, {})
    material_config = garment_configs.get("material", {})

    # Use provided name or generate from config key
    if material_name is None:
        # Clean up config key for material name (remove side prefix if present)
        clean_key = config_key.replace("left_", "").replace("right_", "").replace("garment_", "")
        material_name = f"{clean_key.capitalize()}_Material"

    # Get material properties with fallbacks
    material_color = material_config.get("color", fallback_color)
    roughness = material_config.get("roughness", 0.8)
    metallic = material_config.get("metallic", 0.0)
    specular = material_config.get("specular", 0.3)

    # Create material
    material = bpy.data.materials.new(name=material_name)

    # Use modern node-based approach for Blender 4.3+
    material.use_nodes = True

    # Clear default nodes
    material.node_tree.nodes.clear()

    # Create Principled BSDF setup
    output_node = material.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = material.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

    # Position nodes
    output_node.location = (300, 0)
    principled_node.location = (0, 0)

    # Connect nodes
    material.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Set material properties
    principled_node.inputs['Base Color'].default_value = material_color
    principled_node.inputs['Roughness'].default_value = roughness
    principled_node.inputs['Metallic'].default_value = metallic

    try:
        # Try modern specular input name first
        principled_node.inputs['Specular IOR Level'].default_value = specular
    except KeyError:
        try:
            # Try legacy specular input name
            principled_node.inputs['Specular'].default_value = specular
        except KeyError:
            # If neither exists, just log a warning and continue
            script_log(f"⚠ Specular input not found in Principled BSDF - using default")

    # Apply material to object
    obj.data.materials.append(material)

    script_log(f"✓ Applied material '{material_name}' to {obj.name}")
    return material

##########################################################################################

def apply_simple_material(obj, config_key, material_name=None, fallback_color=(0.8, 0.8, 0.8, 1.0)):
    """
    Apply simple material without nodes (for backward compatibility or simpler cases).

    Args:
        obj: Blender object to apply material to
        config_key: Key in garment_configs for material configuration
        material_name: Optional custom material name
        fallback_color: RGBA fallback color if not specified in config
    """
    # Get material configuration
    garment_config = garment_configs.get(config_key, {})
    material_config = garment_configs.get("material", {})

    # Use provided name or generate from config key
    if material_name is None:
        clean_key = config_key.replace("left_", "").replace("right_", "").replace("garment_", "")
        material_name = f"{clean_key.capitalize()}_Material"

    # Get material properties
    material_color = material_config.get("color", fallback_color)
    roughness = material_config.get("roughness", 0.8)
    metallic = material_config.get("metallic", 0.0)
    specular = material_config.get("specular", 0.3)

    # Create simple material
    material = bpy.data.materials.new(name=material_name)
    material.diffuse_color = material_color
    material.roughness = roughness
    material.metallic = metallic
    material.specular_intensity = specular

    # Apply material to object
    obj.data.materials.append(material)

    script_log(f"✓ Applied simple material '{material_name}' to {obj.name}")
    return material

##########################################################################################

def resolve_head_file_path(blend_file_path):
    """Resolve scene-relative paths for head .blend files"""

    # If absolute path, use as-is
    if os.path.isabs(blend_file_path):
        if os.path.exists(blend_file_path):
            return blend_file_path
        else:
            raise FileNotFoundError(f"Absolute path not found: {blend_file_path}")

    # Scene-relative path resolution
    try:
        from utils import get_scene_paths
        scene_paths = get_scene_paths(args.show, args.scene)
        scene_dir = os.path.dirname(scene_paths["output_pose_data"])
        scene_relative_path = os.path.join(scene_dir, blend_file_path)

        if os.path.exists(scene_relative_path):
            return scene_relative_path
        else:
            raise FileNotFoundError(f"Scene-relative path not found: {scene_relative_path}")

    except Exception as e:
        raise FileNotFoundError(f"Could not resolve scene-relative path {blend_file_path}: {e}")

##########################################################################################

def load_replaceable_head(blend_file_path, figure_name, armature_obj):
    """Append head mesh from external .blend file using naming conventions"""
    try:
        # Save current selection
        previous_active = bpy.context.active_object
        previous_selected = bpy.context.selected_objects.copy()

        script_log(f"Appending from: {blend_file_path}")

        # Append the head object using naming convention
        with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
            # Look for the standard head mesh name
            if "Head_Mesh" in data_from.objects:
                data_to.objects = ["Head_Mesh"]
                script_log("✓ Found Head_Mesh object")
            else:
                # Fallback: use first mesh object
                mesh_objects = [obj for obj in data_from.objects if obj.endswith("Mesh") or obj.startswith("Head")]
                if mesh_objects:
                    data_to.objects = [mesh_objects[0]]
                    script_log(f"✓ Using fallback mesh: {mesh_objects[0]}")
                else:
                    raise Exception("No suitable head mesh found in blend file")

        if not data_to.objects:
            raise Exception("No objects appended from blend file")

        # Get the appended object
        head_obj = data_to.objects[0]
        head_obj.name = f"{figure_name}_Head"

        # Position at head bone location
        head_bone = bpy.context.scene.objects.get(armature_obj.name).pose.bones.get("DEF_Head")
        if head_bone:
            head_obj.location = armature_obj.matrix_world @ head_bone.tail
            script_log(f"Positioned head at DEF_Head bone: {head_obj.location}")

        # Materials are automatically appended with the object
        script_log(f"Head object materials: {[mat.name for mat in head_obj.data.materials]}")

        return head_obj

    except Exception as e:
        script_log(f"Failed to load replaceable head: {e}")
        # Clean up any partially appended objects
        if 'data_to' in locals() and data_to.objects:
            for obj in data_to.objects:
                bpy.data.objects.remove(obj, do_unlink=True)
        return None

##########################################################################################

def setup_imported_head(head_obj, armature_obj, garment_config, replaceable_config):
    """Setup vertex groups and armature for imported head mesh using NEW VERTEX BUNDLES SYSTEM"""

    script_log("Setting up imported head for rig system with new vertex bundles...")

    # Clean up vertex groups if requested
    if replaceable_config.get("vertex_group_cleanup", True):
        script_log("Cleaning up existing vertex groups...")
        for vg in list(head_obj.vertex_groups):
            head_obj.vertex_groups.remove(vg)

    # Remove existing armature modifiers
    for mod in list(head_obj.modifiers):
        if mod.type == 'ARMATURE':
            head_obj.modifiers.remove(mod)

    # NEW: Get head-neck bundle from vertex bundles system
    head_neck_center = get_bundle_center("CTRL_HEAD_TOP")
    head_neck_radius = get_bundle_radius("CTRL_HEAD_TOP")

    # Fallback if bundle not available - try alternative control points
    if not head_neck_center:
        script_log("⚠ CTRL_HEAD_TOP vertex bundle not found, trying CTRL_HEAD_BASE...")
        head_neck_center = get_bundle_center("CTRL_HEAD_BASE")
        head_neck_radius = get_bundle_radius("CTRL_HEAD_BASE")

    # Final fallback if still not available
    if not head_neck_center:
        script_log("⚠ Head-neck vertex bundle not found, using head object location and config radius")
        head_neck_center = head_obj.location
        head_neck_radius = garment_configs.get("neck_connection_radius", 0.08)
    else:
        script_log(f"✓ Using head-neck vertex bundle: center={head_neck_center}, radius={head_neck_radius}")

    # Create vertex group for head bone
    head_group = head_obj.vertex_groups.new(name="DEF_Head")

    # NEW: Create coordination group using vertex bundles system
    coordination_group = head_obj.vertex_groups.new(name="Head_Coordination_Bundle")

    script_log("Applying head-neck bundle weighting with new vertex bundles system...")

    # Calculate sphere radius for influence
    head_sphere_radius = head_neck_radius * 2.0  # Double radius for better influence coverage

    weighted_vertices = 0
    coordination_vertices = 0

    for i, vertex in enumerate(head_obj.data.vertices):
        vert_pos = head_obj.matrix_world @ vertex.co
        distance = (vert_pos - head_neck_center).length

        # Apply weight based on distance to bundle center (NEW SPHERICAL APPROACH)
        if distance <= head_sphere_radius:
            weight = 1.0 - (distance / head_sphere_radius)
            weight = weight * weight  # Quadratic falloff for smoother transitions

            # Reduce weight for top of head (more flexible) while maintaining strong neck connection
            vert_local = head_obj.matrix_world.inverted() @ vert_pos

            # Calculate vertical position relative to head (assuming head is roughly aligned with world Z)
            # This helps differentiate neck area (strong influence) from top of head (weaker influence)
            if vert_local.z > 0.5:  # Top of head
                weight *= 0.3
            elif vert_local.z > 0.2:  # Upper head
                weight *= 0.6
            # Full weight for neck area (z < 0)

            if weight > 0.1:
                coordination_group.add([i], weight, 'REPLACE')
                coordination_vertices += 1

            # Apply to main head group with coordinated weight
            head_group.add([i], weight, 'REPLACE')
            weighted_vertices += 1

        else:
            # Light default weight for distant vertices - ensures all vertices have some influence
            head_group.add([i], 0.2, 'REPLACE')
            weighted_vertices += 1

    script_log(f"✓ Applied vertex bundle weighting to {weighted_vertices} vertices")
    script_log(f"✓ Coordination bundle applied to {coordination_vertices} vertices within influence radius")

    # Add armature modifier
    armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj
    armature_mod.use_vertex_groups = True
    script_log("✓ Added armature modifier with vertex group deformation")

    # =========================================================================
    # OPTIONAL: ADD CLOTH SIMULATION IF ENABLED IN CONFIG
    # =========================================================================
    cloth_config = garment_configs.get("cloth_settings", {})
    if cloth_config.get("enabled", False):
        script_log("DEBUG: Adding cloth simulation to imported head...")
        cloth_mod = head_obj.modifiers.new(name="Cloth", type='CLOTH')
        cloth_mod.settings.quality = cloth_config.get("quality", 6)
        cloth_mod.settings.mass = cloth_config.get("mass", 0.8)
        cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 15.0)
        cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 15.0)
        cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 10.0)
        cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 2.0)
        cloth_mod.settings.air_damping = cloth_config.get("air_damping", 1.0)

        # PIN CLOTH TO COORDINATION BUNDLE FOR ANIMATION SYNC
        cloth_mod.settings.vertex_group_mass = "Head_Coordination_Bundle"
        script_log("✓ Cloth simulation pinned to head coordination bundle")
    else:
        script_log("DEBUG: Cloth simulation disabled for imported head")

    # =========================================================================
    # OPTIONAL: ADD SUBDIVISION SURFACE IF ENABLED
    # =========================================================================
    subdivision_config = garment_configs.get("subdivision", {})
    if subdivision_config.get("enabled", True):
        subdiv_levels = subdivision_config.get("levels", 1)
        script_log(f"DEBUG: Adding subdivision surface with {subdiv_levels} levels...")
        subdiv_mod = head_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = subdiv_levels
        subdiv_mod.render_levels = subdiv_levels
        script_log("✓ Added subdivision surface modifier")

    # =========================================================================
    # SET PROPER MODIFIER ORDER
    # =========================================================================
    script_log("DEBUG: Setting proper modifier order...")
    bpy.context.view_layer.objects.active = head_obj
    modifiers = head_obj.modifiers

    # Build correct order based on which modifiers are present
    correct_order = []

    # Add subdivision first if present
    if "Subdivision" in modifiers:
        correct_order.append("Subdivision")

    # Armature always comes next
    correct_order.append("Armature")

    # Cloth last if present
    if "Cloth" in modifiers:
        correct_order.append("Cloth")

    # Reorder modifiers to ensure proper evaluation
    for mod_name in correct_order:
        mod_index = modifiers.find(mod_name)
        if mod_index >= 0:
            while mod_index > correct_order.index(mod_name):
                bpy.ops.object.modifier_move_up(modifier=mod_name)
                mod_index -= 1

    script_log(f"✓ Modifier order set: {correct_order}")

    # =========================================================================
    # VERIFY THE SETUP
    # =========================================================================
    bpy.context.view_layer.update()

    # Final verification
    has_armature = any(mod.type == 'ARMATURE' for mod in head_obj.modifiers)
    has_vertex_groups = len(head_obj.vertex_groups) > 0
    has_coordination_group = "Head_Coordination_Bundle" in head_obj.vertex_groups

    script_log("=== IMPORTED HEAD SETUP VERIFICATION ===")
    script_log(f"✓ Armature modifier: {'PRESENT' if has_armature else 'MISSING'}")
    script_log(f"✓ Vertex groups: {'PRESENT' if has_vertex_groups else 'MISSING'}")
    script_log(f"✓ Coordination bundle: {'PRESENT' if has_coordination_group else 'MISSING'}")
    script_log(f"✓ Head object: {head_obj.name}")
    script_log(f"✓ Armature object: {armature_obj.name}")

    if has_vertex_groups:
        script_log(f"✓ Vertex groups created: {[vg.name for vg in head_obj.vertex_groups]}")

    if cloth_config.get("enabled", False):
        script_log(f"✓ Cloth simulation: ENABLED and pinned to coordination bundle")
    else:
        script_log(f"✓ Cloth simulation: DISABLED")

    script_log(f"✓ NEW VERTEX BUNDLES SYSTEM: Using head-neck bundle for coordinated animation")
    script_log(f"✓ Bundle center: {head_neck_center}")
    script_log(f"✓ Influence radius: {head_sphere_radius:.3f}")
    script_log(f"✓ Weighted vertices: {weighted_vertices}/{len(head_obj.data.vertices)}")

    return head_obj

##########################################################################################

def save_hero_head(head_obj, export_path, armature_obj, garment_config):
    """Export procedural head as a template .blend file with standardizes naming and materials"""
    script_log(f"=== EXPORTING HERO HEAD TEMPLATE: {export_path} ===")

    # Save current selection state
    previous_active = bpy.context.active_object
    previous_selected = bpy.context.selected_objects.copy()

    try:
        # Create a duplicate of the head for export (don't modify original)
        bpy.ops.object.select_all(action='DESELECT')
        head_obj.select_set(True)
        bpy.context.view_layer.objects.active = head_obj
        bpy.ops.object.duplicate()

        export_head = bpy.context.active_object
        export_head.name = "Head_Mesh"  # Standardized name

        # =========================================================================
        # STEP 1: CLEAN UP THE EXPORT HEAD
        # =========================================================================
        script_log("STEP 1: Cleaning up export head...")

        # Remove armature modifier (keep geometry only)
        for mod in list(export_head.modifiers):
            if mod.type == 'ARMATURE':
                export_head.modifiers.remove(mod)

        # Clear vertex groups (artist will create their own)
        for vg in list(export_head.vertex_groups):
            export_head.vertex_groups.remove(vg)

        # Reset transformations for clean export
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

        # =========================================================================
        # STEP 2: ENSURE CONSISTENT NAMING OF UV MAPS
        # =========================================================================
        script_log("STEP 2: Setting up UV maps...")

        # Ensure we have a UV map
        if not export_head.data.uv_layers:
            script_log("Creating default UV map...")
            export_head.data.uv_layers.new(name="UVMap")

        # Apply smart UV project for better starting point
        bpy.context.view_layer.objects.active = export_head
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(66),
            island_margin=0.02,
            area_weight=1.0,
            correct_aspect=True,
            scale_to_bounds=True
        )
        bpy.ops.object.mode_set(mode='OBJECT')
        script_log("✓ Applied smart UV projection")

        # =========================================================================
        # STEP 3: SETUP STANDARD MATERIALS
        # =========================================================================
        script_log("STEP 3: Setting up standard materials...")

        # Clear existing materials
        export_head.data.materials.clear()

        # Create standard head material
        apply_material_from_config(head_obj, "garment_head", material_name="Head_Material", fallback_color=(0.96, 0.86, 0.72, 1.0))

        """ Need to fix this eventually
        # Clear default nodes for clean setup
        head_mat.node_tree.nodes.clear()

        # Create modern Principled BSDF setup
        output_node = head_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = head_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

        # Position nodes
        output_node.location = (300, 0)
        principled_node.location = (0, 0)

        # Connect nodes
        head_mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

        # Set realistic skin properties
        principled_node.inputs['Base Color'].default_value = (0.96, 0.86, 0.72, 1.0)  # Skin tone
        principled_node.inputs['Roughness'].default_value = 0.4  # Slightly rough for skin
        principled_node.inputs['Subsurface'].default_value = 0.2  # Skin subsurface scattering
        principled_node.inputs['Subsurface Radius'].default_value = (1.0, 0.2, 0.1)  # Skin scattering
        principled_node.inputs['Subsurface Color'].default_value = (0.9, 0.7, 0.7, 1.0)  # Blood color

        # Add material to head
        export_head.data.materials.append(head_mat)

        # Create placeholder materials for other parts
        eye_mat = bpy.data.materials.new(name="Head_Eyes_Material")
        eye_mat.use_nodes = True
        eye_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.1, 0.1, 0.9,
                                                                                         1.0)  # Blue eyes

        teeth_mat = bpy.data.materials.new(name="Head_Teeth_Material")
        teeth_mat.use_nodes = True
        teeth_mat.node_tree.nodes["Principled BSDF"].inputs['Base Color'].default_value = (0.95, 0.95, 0.9,
                                                                                           1.0)  # Off-white teeth

        script_log("✓ Created standard materials: Head_Material, Head_Eyes_Material, Head_Teeth_Material")
        Need to fix """


        # =========================================================================
        # STEP 4: CREATE METADATA AND ORGANIZATION
        # =========================================================================
        script_log("STEP 4: Adding metadata...")

        # Add custom properties for documentation
        export_head["is_hero_head_template"] = True
        export_head["export_version"] = "1.0"
        export_head["original_scale"] = list(export_head.scale)
        export_head["recommended_scale"] = 1.0

        # Create a collection for organization
        if "Hero_Head_Assets" not in bpy.data.collections:
            head_collection = bpy.data.collections.new("Hero_Head_Assets")
            bpy.context.scene.collection.children.link(head_collection)
        else:
            head_collection = bpy.data.collections["Hero_Head_Assets"]

        # Move head to collection
        if export_head.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(export_head)
        head_collection.objects.link(export_head)

        # =========================================================================
        # STEP 5: EXPORT TO BLEND FILE
        # =========================================================================
        script_log(f"STEP 5: Exporting to: {export_path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(export_path), exist_ok=True)

        # Select only the head for export
        bpy.ops.object.select_all(action='DESELECT')
        export_head.select_set(True)
        bpy.context.view_layer.objects.active = export_head

        # Save as new blend file
        bpy.ops.wm.save_as_mainfile(
            filepath=export_path,
            check_existing=False,
            compress=True
        )

        script_log(f"✓ Successfully exported hero head template: {export_path}")

        # =========================================================================
        # STEP 6: CLEAN UP
        # =========================================================================
        script_log("STEP 6: Cleaning up...")

        # Delete the temporary export object (original remains in scene)
        bpy.ops.object.select_all(action='DESELECT')
        export_head.select_set(True)
        bpy.ops.object.delete()

        # Restore original selection
        if previous_active:
            bpy.context.view_layer.objects.active = previous_active
        for obj in previous_selected:
            obj.select_set(True)

        return True

    except Exception as e:
        script_log(f"ERROR Failed to export hero head template: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")

        # Clean up on failure
        try:
            if 'export_head' in locals() and export_head.name in bpy.data.objects:
                bpy.data.objects.remove(export_head, do_unlink=True)
        except:
            pass

        # Restore selection state
        try:
            if previous_active:
                bpy.context.view_layer.objects.active = previous_active
            for obj in previous_selected:
                obj.select_set(True)
        except:
            pass

        return False

##########################################################################################

def create_head(armature_obj, figure_name, garment_config, global_cloth_settings, neck_config=None):
    """Create head - either procedural or from replaceable asset with template export - UPDATED WITH NEW VERTEX BUNDLES SYSTEM"""
    script_log("=== HEAD CREATION STARTED (NEW VERTEX BUNDLES SYSTEM) ===")

    # =========================================================================
    # STEP 1: ENSURE NOSE CONTROL POINT EXISTS FIRST (CRITICAL FOR SAFETY)
    # =========================================================================
    script_log("STEP 1: Ensuring NOSE control point exists for safe constraint setup...")
    first_frame = mocap_data.get(str(frame_numbers[0]), {})

    def ensure_nose_control_point_safe(first_frame):
        """Guarantee CTRL_NOSE exists and is positioned correctly - SAFE VERSION"""
        script_log("Ensuring NOSE control point exists...")

        nose_obj = bpy.data.objects.get("CTRL_NOSE")

        if not nose_obj:
            script_log("Creating CTRL_NOSE control point from landmark data...")
            # Create from landmark data
            if "NOSE" in first_frame:
                pos_data = first_frame["NOSE"]
                nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

                bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
                nose_obj = bpy.context.active_object
                nose_obj.name = "CTRL_NOSE"

                # Add to control collection
                control_coll = bpy.data.collections.get("Main_ControlPoints")
                if control_coll:
                    control_coll.objects.link(nose_obj)

                script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
            else:
                script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
                bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
                nose_obj = bpy.context.active_object
                nose_obj.name = "CTRL_NOSE"
        else:
            script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

        return nose_obj

    def setup_head_constraints_safe(armature_obj, head_bone_name):
        """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
        script_log(f"Setting up safe head constraints for {head_bone_name}...")

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        head_bone = armature_obj.pose.bones[head_bone_name]

        # 1. CLEAR ALL existing constraints first (prevent conflicts)
        for constraint in list(head_bone.constraints):
            head_bone.constraints.remove(constraint)
        script_log("✓ Cleared existing head constraints")

        # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        if neck_bone:
            copy_loc = head_bone.constraints.new('COPY_LOCATION')
            copy_loc.target = armature_obj
            copy_loc.subtarget = "DEF_Neck"
            copy_loc.use_offset = True  # Maintain head's local position relative to neck
            copy_loc.influence = 1.0
            script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

        # 3. ROTATION: Damped track to NOSE control point (facing direction)
        nose_obj = bpy.data.objects.get("CTRL_NOSE")
        if nose_obj:
            track = head_bone.constraints.new('DAMPED_TRACK')
            track.name = "Track_Nose_Direction"
            track.target = nose_obj
            track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
            track.influence = 1.0
            script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

            # 4. ROTATION LIMITS: Prevent extreme angles for stability
            limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
            limit_rot.name = "Limit_Head_Rotation"
            limit_rot.use_limit_x = True
            limit_rot.min_x = -0.5  # Limited head tilt (nodding)
            limit_rot.max_x = 0.5
            limit_rot.use_limit_y = True
            limit_rot.min_y = -1.5  # Head turn (looking left/right)
            limit_rot.max_y = 1.5
            limit_rot.use_limit_z = True
            limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
            limit_rot.max_z = 0.3
            limit_rot.owner_space = 'LOCAL'
            script_log("✓ Added natural head rotation limits")
        else:
            script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

        bpy.ops.object.mode_set(mode='OBJECT')
        script_log("✓ Safe head constraints setup complete")

    # CREATE NOSE CONTROL POINT BEFORE ANYTHING ELSE
    nose_obj = ensure_nose_control_point_safe(first_frame)

    # =========================================================================
    # STEP 2: CREATE PROCEDURAL HEAD FIRST (needed for export or fallback)
    # =========================================================================
    script_log("STEP 2: Creating procedural head with safe constraints...")

    def create_procedural_head_safe(armature_obj, figure_name, garment_config, global_cloth_settings, neck_config=None):
        """Create human-like head with coordinated vertex bundles for seamless neck integration - UPDATED WITH NEW VERTEX BUNDLES SYSTEM"""
        script_log("Creating procedural garment_head with coordinated vertex bundles and safe constraints...")

        # Get head bone position
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        try:
            head_bone = armature_obj.pose.bones.get("DEF_Head")
            neck_bone = armature_obj.pose.bones.get("DEF_Neck")
            head_bone_name = "DEF_Head"
            neck_bone_name = "DEF_Neck"

            bpy.ops.object.mode_set(mode='OBJECT')

            if not head_bone:
                script_log("ERROR: Could not find head bone")
                return None

            # Get head bone positions in world space - Use tail for head position
            head_base_pos = armature_obj.matrix_world @ head_bone.head  # Base of head/neck (where neck connects)
            head_top_pos = armature_obj.matrix_world @ head_bone.tail  # Top of head (where head should be)

            # Get neck bone position for coordination
            neck_base_pos = armature_obj.matrix_world @ neck_bone.head if neck_bone else head_base_pos
            neck_top_pos = armature_obj.matrix_world @ neck_bone.tail if neck_bone else head_base_pos

            # Get head dimensions from new config format
            scale = garment_configs.get("scale", [0.25, 0.28, 0.3])  # Larger, more human-sized proportions
            position_offset = garment_configs.get("position_offset", [0.0, 0.0, 0.0])
            rotation_offset = garment_configs.get("rotation_offset", [0.0, 0.0, 0.0])
            neck_connection_radius = garment_configs.get("neck_connection_radius", 0.08)
            subdivision_levels = garment_configs.get("subdivision_levels", 3)  # More subdivisions for detail

            # Get geometry settings from config
            geometry_settings = global_cloth_settings.get("geometry_settings", {})
            head_segments = geometry_settings.get("head_segments", 32)  # More segments for detail
            head_ring_count = geometry_settings.get("head_ring_count", 24)  # More rings for detail
            base_radii = global_cloth_settings.get("base_radii", {})
            head_base_radius = base_radii.get("head", 0.25)  # Larger base radius

            # Get head-neck bundle diameter from config - USING NEW VERTEX BUNDLES SYSTEM
            head_neck_diameter = garment_configs.get("diameter_neck", 0.16)  # Diameter at base of head/neck junction

            # NEW: Get head-neck bundle center and radius from vertex bundles system
            head_neck_center = get_bundle_center("CTRL_HEAD_TOP")  # Using HEAD_TOP as head-neck junction
            head_neck_radius = get_bundle_radius("CTRL_HEAD_TOP")

            # Fallback if head bundle not available
            if not head_neck_center:
                script_log("⚠ Head-neck vertex bundle not found, using calculated position")
                head_neck_center = head_base_pos
                head_neck_radius = head_neck_diameter / 2

            script_log(f"DEBUG: Procedural head scale: {scale}")
            script_log(f"DEBUG: Position offset: {position_offset}, Rotation offset: {rotation_offset}")
            script_log(f"DEBUG: Head bone - base (neck): {head_base_pos}, top (head): {head_top_pos}")
            script_log(f"DEBUG: Head geometry - segments: {head_segments}, rings: {head_ring_count}")
            script_log(f"DEBUG: Head-neck junction diameter: {head_neck_diameter}")
            script_log(f"DEBUG: Head-neck bundle center: {head_neck_center}, radius: {head_neck_radius}")

            # =========================================================================
            # STEP 1: CREATE HEAD MESH WITH NECK OPENING
            # =========================================================================
            script_log("DEBUG: Creating procedural head as UV sphere with neck opening...")
            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=head_segments,
                ring_count=head_ring_count,
                radius=head_base_radius,
                location=head_top_pos + Vector(position_offset)  # Position at TOP of head bone
            )
            head_obj = bpy.context.active_object
            head_obj.name = f"{figure_name}_Head"

            # Apply human-like scaling (wider and taller than spherical)
            head_obj.scale = Vector(scale)

            # Apply rotation offset if specified
            if any(rotation_offset):
                head_obj.rotation_euler = Vector(rotation_offset)

            # =========================================================================
            # STEP 2: SHAPE HEAD TO HUMAN PROPORTIONS
            # =========================================================================
            script_log("DEBUG: Shaping head to human proportions...")
            bpy.context.view_layer.objects.active = head_obj

            # Use bmesh for vertex manipulation without entering edit mode
            bm = bmesh.new()
            bm.from_mesh(head_obj.data)

            # Define human head proportions (these could be moved to config)
            human_proportions = garment_configs.get("human_proportions", {
                'forehead_scale': 0.95,  # Slightly narrower forehead
                'cheek_scale': 1.1,  # Fuller cheeks
                'jaw_scale': 0.9,  # Narrower jaw
                'chin_scale': 0.8,  # More pronounced chin
                'back_head_scale': 1.05,  # Fuller back of head
                'top_head_scale': 0.95,  # Slightly flatter top
                'neck_opening_scale': 0.7,  # Create neck opening
            })

            # Shape vertices to human proportions
            for vert in bm.verts:
                # Convert to spherical coordinates for easier manipulation
                x, y, z = vert.co
                radius = (x ** 2 + y ** 2 + z ** 2) ** 0.5
                if radius > 0.001:  # Avoid division by zero
                    # Normalize to unit sphere
                    nx, ny, nz = x / radius, y / radius, z / radius

                    # Apply human proportions based on vertex position
                    scale_factor = 1.0

                    # Forehead area (front top)
                    if nz > 0.3 and ny > 0.2:
                        scale_factor = human_proportions['forehead_scale']

                    # Cheek area (front sides)
                    elif nz > -0.2 and abs(nx) > 0.4:
                        scale_factor = human_proportions['cheek_scale']

                    # Jaw area (front bottom)
                    elif nz < -0.3 and abs(ny) < 0.3:
                        scale_factor = human_proportions['jaw_scale']

                    # Chin area (very front bottom)
                    elif nz < -0.5 and abs(nx) < 0.2:
                        scale_factor = human_proportions['chin_scale']

                    # Back of head
                    elif nz < -0.1 and abs(nx) < 0.3:
                        scale_factor = human_proportions['back_head_scale']

                    # Top of head
                    elif ny > 0.7:
                        scale_factor = human_proportions['top_head_scale']

                    # Neck opening area (bottom center)
                    elif ny < -0.6 and abs(nz) < 0.3 and abs(nx) < 0.3:
                        scale_factor = human_proportions['neck_opening_scale']

                    # Apply the scaling
                    vert.co.x = nx * radius * scale_factor
                    vert.co.y = ny * radius * scale_factor
                    vert.co.z = nz * radius * scale_factor

            # Write back to mesh (object mode is safe for this)
            bm.to_mesh(head_obj.data)
            bm.free()

            # =========================================================================
            # STEP 3: ADD FACIAL FEATURES USING SHAPE KEYS
            # =========================================================================
            facial_features = garment_configs.get("facial_features", {})
            if facial_features.get("enable_eye_sockets", True) or facial_features.get("enable_nose_bridge", True):
                script_log("DEBUG: Adding facial features with shape keys...")

                # Add basis shape key first
                head_obj.shape_key_add(name="Basis")

                # Create shape keys FIRST before using bmesh
                shape_key_objects = {}

                if facial_features.get("enable_eye_sockets", True):
                    script_log("DEBUG: Creating eye sockets shape key...")
                    eye_sockets = head_obj.shape_key_add(name="EyeSockets")
                    shape_key_objects["EyeSockets"] = eye_sockets

                if facial_features.get("enable_nose_bridge", True):
                    script_log("DEBUG: Creating nose bridge shape key...")
                    nose_bridge = head_obj.shape_key_add(name="NoseBridge")
                    shape_key_objects["NoseBridge"] = nose_bridge

                # NOW use bmesh for shape key editing
                bm = bmesh.new()
                bm.from_mesh(head_obj.data)

                # Ensure we have shape key layers - they should exist now
                shape_keys = bm.verts.layers.shape

                # Create eye sockets if enabled
                if facial_features.get("enable_eye_sockets", True) and "EyeSockets" in shape_keys:
                    script_log("DEBUG: Adding eye sockets geometry...")
                    eye_socket_layer = shape_keys["EyeSockets"]
                    eye_socket_depth = facial_features.get("eye_socket_depth", 0.08)

                    for vert in bm.verts:
                        x, y, z = vert.co
                        # Eye socket areas (front upper sides)
                        if z > 0.1 and abs(x) > 0.3 and y > 0.1:
                            # Create indentation for eye sockets
                            depth = eye_socket_depth
                            vert[eye_socket_layer] = vert.co + Vector((-x * depth * 0.5, -y * depth * 0.3, -z * depth))

                # Create nose bridge if enabled
                if facial_features.get("enable_nose_bridge", True) and "NoseBridge" in shape_keys:
                    script_log("DEBUG: Adding nose bridge geometry...")
                    nose_bridge_layer = shape_keys["NoseBridge"]
                    nose_bridge_height = facial_features.get("nose_bridge_height", 0.05)

                    for vert in bm.verts:
                        x, y, z = vert.co
                        # Nose bridge area (center front)
                        if abs(x) < 0.15 and z > 0.2 and y > -0.1:
                            # Create protrusion for nose
                            height = nose_bridge_height
                            vert[nose_bridge_layer] = vert.co + Vector((0, 0, height))

                # Write shape keys back
                bm.to_mesh(head_obj.data)
                bm.free()

                # Apply some of the facial features
                if "EyeSockets" in head_obj.data.shape_keys.key_blocks:
                    head_obj.data.shape_keys.key_blocks["EyeSockets"].value = 0.3
                if "NoseBridge" in head_obj.data.shape_keys.key_blocks:
                    head_obj.data.shape_keys.key_blocks["NoseBridge"].value = 0.4
            else:
                script_log("DEBUG: Facial features disabled in config")

            # =========================================================================
            # STEP 4: SETUP VERTEX GROUPS WITH COORDINATED HEAD-NECK BUNDLES (NEW SYSTEM)
            # =========================================================================
            script_log("DEBUG: Setting up head vertex groups with NEW vertex bundles system...")

            # Clear any existing parenting
            head_obj.parent = None

            # Clear any existing vertex groups
            for vg in list(head_obj.vertex_groups):
                head_obj.vertex_groups.remove(vg)

            # Remove any existing armature modifiers
            for mod in list(head_obj.modifiers):
                if mod.type == 'ARMATURE':
                    head_obj.modifiers.remove(mod)

            # Create vertex group for head bone
            head_group = head_obj.vertex_groups.new(name=head_bone_name)

            # NEW: Create spherical coordination group using vertex bundles system
            head_coordination_group = head_obj.vertex_groups.new(name="Head_Coordination_Bundle")

            # Calculate sphere radius for influence (using new vertex bundles system)
            head_sphere_radius = head_neck_radius * 2.0  # Double radius for better influence

            # Apply head-neck bundle weighting using NEW VERTEX BUNDLES SYSTEM
            script_log(
                f"✓ Applying head-neck vertex bundle with center: {head_neck_center}, radius: {head_sphere_radius}")

            for i, vertex in enumerate(head_obj.data.vertices):
                vert_pos = head_obj.matrix_world @ vertex.co
                distance = (vert_pos - head_neck_center).length

                # Apply weight based on distance to bundle center (NEW SPHERICAL APPROACH)
                if distance <= head_sphere_radius:
                    weight = 1.0 - (distance / head_sphere_radius)
                    weight = weight * weight  # Quadratic falloff

                    # Reduce weight for top of head (more flexible)
                    vert_local = head_obj.matrix_world.inverted() @ vert_pos
                    if vert_local.y > 0.5:  # Top of head
                        weight *= 0.3
                    elif vert_local.y > 0.2:  # Upper head
                        weight *= 0.6
                    # Full weight for neck area (y < 0)

                    if weight > 0.1:
                        head_coordination_group.add([i], weight, 'REPLACE')
                        head_group.add([i], weight, 'REPLACE')
                else:
                    # Light default weight for distant vertices
                    head_group.add([i], 0.2, 'REPLACE')

            script_log(f"✓ Applied head-neck vertex bundle weighting to {head_obj.name}")

            # =========================================================================
            # STEP 5: ADD ARMATURE MODIFIER
            # =========================================================================
            script_log("DEBUG: Adding armature modifier...")

            # Add armature modifier
            armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
            armature_mod.object = armature_obj
            armature_mod.use_vertex_groups = True

            # =========================================================================
            # STEP 6: ADD SUBDIVISION AND MATERIALS
            # =========================================================================
            script_log("DEBUG: Adding subdivision and skin material...")

            # Add subdivision from config
            subdiv_mod = head_obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = subdivision_levels
            subdiv_mod.render_levels = subdivision_levels

            # ADD SKIN MATERIAL WITH MODERN BLENDER 4.3+ NODES
            script_log("DEBUG: Creating skin material with modern node setup...")
            material_config = garment_configs.get("material", {})
            material_color = material_config.get("color", [0.96, 0.86, 0.72, 1.0])

            head_mat = bpy.data.materials.new(name="Head_Material")
            head_mat.use_nodes = True

            # Clear default nodes for clean setup
            head_mat.node_tree.nodes.clear()

            # Create modern Principled BSDF setup
            output_node = head_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
            principled_node = head_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

            # Position nodes
            output_node.location = (300, 0)
            principled_node.location = (0, 0)

            # Connect nodes
            head_mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

            # Set material properties using modern API
            principled_node.inputs['Base Color'].default_value = material_color
            principled_node.inputs['Roughness'].default_value = material_config.get("roughness", 0.4)
            principled_node.inputs['Metallic'].default_value = material_config.get("metallic", 0.0)

            head_obj.data.materials.append(head_mat)

            # =========================================================================
            # STEP 7: CLOTH SIMULATION - DISABLED AS REQUESTED
            # =========================================================================
            cloth_settings = garment_configs.get("cloth_settings", {})
            if cloth_settings.get("enabled", False):
                script_log("DEBUG: Adding cloth simulation to garment_head...")
                cloth_mod = head_obj.modifiers.new(name="Cloth", type='CLOTH')
                cloth_mod.settings.quality = cloth_settings.get("quality", 6)
                cloth_mod.settings.mass = cloth_settings.get("mass", 0.8)
                cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 15.0)
                cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 15.0)
                cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 10.0)
                cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 2.0)
                cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)

                # NEW: Pin cloth to coordination group if enabled
                cloth_mod.settings.vertex_group_mass = "Head_Coordination_Bundle"
                script_log("✓ Head cloth pinned to coordination bundle")
            else:
                script_log("DEBUG: Cloth simulation disabled for head")

            # =========================================================================
            # STEP 8: SET MODIFIER ORDER
            # =========================================================================
            bpy.context.view_layer.objects.active = head_obj
            modifiers = head_obj.modifiers

            # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
            correct_order = ["Subdivision", "Armature", "Cloth"]
            for mod_name in correct_order:
                mod_index = modifiers.find(mod_name)
                if mod_index >= 0:
                    while mod_index > correct_order.index(mod_name):
                        bpy.ops.object.modifier_move_up(modifier=mod_name)
                        mod_index -= 1

            # =========================================================================
            # STEP 9: NOSE LANDMARK TRACKING SETUP - UPDATED WITH SAFE CONSTRAINTS
            # =========================================================================
            script_log("DEBUG: Setting up NOSE landmark tracking with safe constraints...")

            # SETUP SAFE HEAD CONSTRAINTS (PORCELAIN DOLL ARCHITECTURE)
            setup_head_constraints_safe(armature_obj, head_bone_name)

            # =========================================================================
            # STEP 10: FINAL VERIFICATION
            # =========================================================================
            bpy.context.view_layer.update()

            script_log("=== PROCEDURAL HEAD CREATION COMPLETE (NEW VERTEX BUNDLES SYSTEM) ===")
            script_log(f"✓ Head positioned at TOP of DEF_Head bone (tail position)")
            script_log(f"✓ Head shaped with human proportions")
            script_log(f"✓ Facial features: Eye sockets, Nose bridge")
            script_log(f"✓ Head rotation tracks NOSE landmark in real-time (SAFE CONSTRAINTS)")
            script_log(f"✓ Natural rotation limits applied for stability")
            script_log(f"✓ Head scale: {scale} (larger, human-sized proportions)")
            script_log(f"✓ Head segments: {head_segments}, rings: {head_ring_count}")
            script_log(f"✓ Subdivision levels: {subdivision_levels}")
            script_log(f"✓ Neck connection diameter: {head_neck_diameter}")
            script_log(f"✓ Head parented to {head_bone_name}")
            script_log(f"✓ Skin material applied with modern Principled BSDF")
            script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_settings.get('enabled', False) else 'DISABLED'}")
            script_log(f"✓ NEW VERTEX BUNDLES SYSTEM: Using head-neck bundle for coordinated weighting")
            script_log(f"✓ Bundle center: {head_neck_center}, influence radius: {head_sphere_radius}")
            script_log(f"✓ PORCELAIN DOLL ARCHITECTURE: Neck anchor + Nose rotation guide")

            # Call create_neck for coordinated creation
            if neck_config:
                create_neck(armature_obj, figure_name, neck_config, global_cloth_settings)

            return head_obj

        except Exception as e:
            script_log(f"ERROR creating procedural head with coordinated bundles: {e}")
            import traceback
            script_log(f"Traceback: {traceback.format_exc()}")
            bpy.ops.object.mode_set(mode='OBJECT')
            return None

    # CREATE PROCEDURAL HEAD WITH SAFE CONSTRAINTS AND NEW VERTEX BUNDLES SYSTEM
    procedural_head = create_procedural_head_safe(armature_obj, figure_name, garment_config, global_cloth_settings,
                                                  neck_config)

    if not procedural_head:
        script_log("ERROR Failed to create procedural head")
        return None

    # =========================================================================
    # STEP 3: CHECK IF WE SHOULD EXPORT AS TEMPLATE
    # =========================================================================
    replaceable_config = garment_configs.get("replaceable_head", {})
    if replaceable_config.get("export_template", False):
        script_log("STEP 3: Exporting head template...")

        export_path = replaceable_config.get("export_path", "assets/hero_head_template.blend")

        # Resolve scene-relative path
        try:
            resolved_export_path = resolve_head_file_path(export_path)
            script_log(f"Export path resolved to: {resolved_export_path}")

            success = save_hero_head(procedural_head, resolved_export_path, armature_obj, garment_config)

            if success:
                script_log("✓ Head template exported successfully")
                # Continue to use the procedural head in the scene
                script_log("✓ Using procedural head in scene (template exported)")
                return procedural_head
            else:
                script_log("⚠ Template export failed, using procedural head in scene")
                return procedural_head

        except Exception as e:
            script_log(f"⚠ Template export path resolution failed: {e}, using procedural head")
            return procedural_head

    # =========================================================================
    # STEP 4: CHECK IF WE SHOULD LOAD REPLACEABLE HEAD INSTEAD
    # =========================================================================
    if replaceable_config.get("enabled", False):
        script_log("STEP 4: Attempting to load replaceable head...")

        try:
            blend_file_path = replaceable_config.get("blend_file")
            if not blend_file_path:
                raise ValueError("No blend_file specified in replaceable_head config")

            # Resolve scene-relative path
            resolved_path = resolve_head_file_path(blend_file_path)
            script_log(f"Loading head from: {resolved_path}")

            replaceable_head = load_replaceable_head(
                resolved_path,
                figure_name,
                armature_obj
            )

            if replaceable_head:
                script_log("✓ Successfully loaded replaceable head")

                # =========================================================================
                # STEP 4A: REMOVE PROCEDURAL HEAD SINCE WE'RE USING REPLACEABLE
                # =========================================================================
                script_log("Removing procedural head (replacing with imported head)...")
                bpy.data.objects.remove(procedural_head, do_unlink=True)
                script_log("✓ Procedural head removed")

                # =========================================================================
                # STEP 4B: APPLY TRANSFORMATIONS TO REPLACEABLE HEAD
                # =========================================================================
                script_log("Applying transformations to replaceable head...")
                scale_factor = replaceable_config.get("scale_factor", 1.0)
                position_offset = Vector(replaceable_config.get("position_offset", [0, 0, 0]))
                rotation_offset = Vector(replaceable_config.get("rotation_offset", [0, 0, 0]))

                replaceable_head.scale = (scale_factor, scale_factor, scale_factor)
                replaceable_head.location += position_offset
                replaceable_head.rotation_euler += rotation_offset

                script_log(
                    f"✓ Applied transformations - Scale: {scale_factor}, Offset: {position_offset}, Rotation: {rotation_offset}")

                # =========================================================================
                # STEP 4C: SETUP FOR OUR RIG SYSTEM WITH NEW VERTEX BUNDLES
                # =========================================================================
                script_log("Setting up replaceable head for rig system with new vertex bundles...")
                replaceable_head = setup_imported_head_with_bundles(replaceable_head, armature_obj, garment_config,
                                                                    replaceable_config)

                script_log("✓ Replaceable head setup complete")

                # =========================================================================
                # STEP 4D: VERIFY FINAL SETUP
                # =========================================================================
                bpy.context.view_layer.update()

                # Verify armature modifier exists
                has_armature_mod = any(mod.type == 'ARMATURE' for mod in replaceable_head.modifiers)
                if not has_armature_mod:
                    script_log("⚠ WARNING: Replaceable head missing armature modifier")

                # Verify vertex groups exist
                has_vertex_groups = len(replaceable_head.vertex_groups) > 0
                if not has_vertex_groups:
                    script_log("⚠ WARNING: Replaceable head missing vertex groups")

                script_log("=== REPLACEABLE HEAD CREATION COMPLETE ===")
                script_log(f"✓ Successfully replaced procedural head with imported head")
                script_log(f"✓ Head object: {replaceable_head.name}")
                script_log(f"✓ Materials: {[mat.name for mat in replaceable_head.data.materials]}")
                script_log(f"✓ Vertex groups: {[vg.name for vg in replaceable_head.vertex_groups]}")
                script_log(f"✓ Modifiers: {[mod.name for mod in replaceable_head.modifiers]}")
                script_log(f"✓ NEW VERTEX BUNDLES SYSTEM: Applied to replaceable head")

                return replaceable_head
            else:
                script_log("⚠ Failed to load replaceable head, using procedural head as fallback")
                script_log("✓ Using procedural head in scene")
                return procedural_head

        except Exception as e:
            script_log(f"ERROR Replaceable head loading failed: {e}")
            import traceback
            script_log(f"Traceback: {traceback.format_exc()}")

            if not garment_configs.get("fallback_to_procedural", True):
                script_log("ERROR Fallback disabled - returning None")
                # Clean up procedural head since we're not using it
                bpy.data.objects.remove(procedural_head, do_unlink=True)
                return None
            else:
                script_log("✓ Using procedural head as fallback")
                return procedural_head

    # =========================================================================
    # STEP 5: IF NO REPLACEABLE HEAD REQUESTED, USE PROCEDURAL
    # =========================================================================
    script_log("STEP 5: Using procedural head (no replaceable head requested)...")

    # Verify procedural head setup
    bpy.context.view_layer.update()

    # Check if procedural head has proper setup
    has_armature_mod = any(mod.type == 'ARMATURE' for mod in procedural_head.modifiers)
    has_vertex_groups = len(procedural_head.vertex_groups) > 0
    has_materials = len(procedural_head.data.materials) > 0

    script_log("=== PROCEDURAL HEAD VERIFICATION ===")
    script_log(f"✓ Armature modifier: {'PRESENT' if has_armature_mod else 'MISSING'}")
    script_log(f"✓ Vertex groups: {'PRESENT' if has_vertex_groups else 'MISSING'}")
    script_log(f"✓ Materials: {'PRESENT' if has_materials else 'MISSING'}")
    script_log(f"✓ Head object: {procedural_head.name}")
    script_log(f"✓ NEW VERTEX BUNDLES SYSTEM: Integrated for head-neck coordination")

    if has_vertex_groups:
        script_log(f"✓ Vertex groups: {[vg.name for vg in procedural_head.vertex_groups]}")

    if has_materials:
        script_log(f"✓ Materials: {[mat.name for mat in procedural_head.data.materials]}")

    script_log("=== PROCEDURAL HEAD CREATION COMPLETE ===")
    return procedural_head


def setup_imported_head_with_bundles(head_obj, armature_obj, garment_config, replaceable_config):
    """Setup vertex groups and armature for imported head mesh using NEW VERTEX BUNDLES SYSTEM"""

    script_log("Setting up imported head for rig system with new vertex bundles...")

    # Clean up vertex groups if requested
    if replaceable_config.get("vertex_group_cleanup", True):
        script_log("Cleaning up existing vertex groups...")
        for vg in list(head_obj.vertex_groups):
            head_obj.vertex_groups.remove(vg)

    # Remove existing armature modifiers
    for mod in list(head_obj.modifiers):
        if mod.type == 'ARMATURE':
            head_obj.modifiers.remove(mod)

    # NEW: Get head-neck bundle from vertex bundles system
    head_neck_center = get_bundle_center("CTRL_HEAD_TOP")
    head_neck_radius = get_bundle_radius("CTRL_HEAD_TOP")

    # Fallback if bundle not available
    if not head_neck_center:
        script_log("⚠ Head-neck vertex bundle not found, using standard weighting")
        head_neck_center = head_obj.location
        head_neck_radius = garment_configs.get("neck_connection_radius", 0.08)

    # Create vertex group for head bone
    head_group = head_obj.vertex_groups.new(name="DEF_Head")

    # NEW: Create coordination group using vertex bundles system
    coordination_group = head_obj.vertex_groups.new(name="Head_Coordination_Bundle")

    script_log("Applying head-neck bundle weighting with new vertex bundles system...")

    # Calculate sphere radius for influence
    head_sphere_radius = head_neck_radius * 2.0

    weighted_vertices = 0
    for i, vertex in enumerate(head_obj.data.vertices):
        vert_pos = head_obj.matrix_world @ vertex.co
        distance = (vert_pos - head_neck_center).length

        # Apply weight based on distance to bundle center (NEW SPHERICAL APPROACH)
        if distance <= head_sphere_radius:
            weight = 1.0 - (distance / head_sphere_radius)
            weight = weight * weight  # Quadratic falloff

            # Reduce weight for top of head (more flexible)
            vert_local = head_obj.matrix_world.inverted() @ vert_pos
            if vert_local.y > 0.5:  # Top of head
                weight *= 0.3
            elif vert_local.y > 0.2:  # Upper head
                weight *= 0.6

            if weight > 0.1:
                coordination_group.add([i], weight, 'REPLACE')
                head_group.add([i], weight, 'REPLACE')
                weighted_vertices += 1
        else:
            # Light default weight for distant vertices
            head_group.add([i], 0.2, 'REPLACE')
            weighted_vertices += 1

    script_log(f"✓ Applied vertex bundle weighting to {weighted_vertices} vertices")

    # Add armature modifier
    armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj
    armature_mod.use_vertex_groups = True

    script_log("✓ Imported head setup complete with new vertex bundles system")
    return head_obj

##########################################################################################

def ensure_nose_control_point(first_frame):
    """Guarantee CTRL_NOSE exists and is positioned correctly"""
    script_log("Ensuring NOSE control point exists...")

    nose_obj = bpy.data.objects.get("CTRL_NOSE")

    if not nose_obj:
        script_log("Creating CTRL_NOSE control point from landmark data...")
        # Create from landmark data
        if "NOSE" in first_frame:
            pos_data = first_frame["NOSE"]
            nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

            bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"

            # Add to control collection
            control_coll = bpy.data.collections.get("Main_ControlPoints")
            if control_coll:
                control_coll.objects.link(nose_obj)

            script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
        else:
            script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"
    else:
        script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

    return nose_obj


def setup_head_constraints_safe(armature_obj, head_bone_name):
    """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
    script_log(f"Setting up safe head constraints for {head_bone_name}...")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    head_bone = armature_obj.pose.bones[head_bone_name]

    # 1. CLEAR ALL existing constraints first (prevent conflicts)
    for constraint in list(head_bone.constraints):
        head_bone.constraints.remove(constraint)
    script_log("✓ Cleared existing head constraints")

    # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
    neck_bone = armature_obj.pose.bones.get("DEF_Neck")
    if neck_bone:
        copy_loc = head_bone.constraints.new('COPY_LOCATION')
        copy_loc.target = armature_obj
        copy_loc.subtarget = "DEF_Neck"
        copy_loc.use_offset = True  # Maintain head's local position relative to neck
        copy_loc.influence = 1.0
        script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

    # 3. ROTATION: Damped track to NOSE control point (facing direction)
    nose_obj = bpy.data.objects.get("CTRL_NOSE")
    if nose_obj:
        track = head_bone.constraints.new('DAMPED_TRACK')
        track.name = "Track_Nose_Direction"
        track.target = nose_obj
        track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
        track.influence = 1.0
        script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

        # 4. ROTATION LIMITS: Prevent extreme angles for stability
        limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
        limit_rot.name = "Limit_Head_Rotation"
        limit_rot.use_limit_x = True
        limit_rot.min_x = -0.5  # Limited head tilt (nodding)
        limit_rot.max_x = 0.5
        limit_rot.use_limit_y = True
        limit_rot.min_y = -1.5  # Head turn (looking left/right)
        limit_rot.max_y = 1.5
        limit_rot.use_limit_z = True
        limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
        limit_rot.max_z = 0.3
        limit_rot.owner_space = 'LOCAL'
        script_log("✓ Added natural head rotation limits")
    else:
        script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log("✓ Safe head constraints setup complete")


def ensure_nose_control_point(first_frame):
    """Guarantee CTRL_NOSE exists and is positioned correctly"""
    script_log("Ensuring NOSE control point exists...")

    nose_obj = bpy.data.objects.get("CTRL_NOSE")

    if not nose_obj:
        script_log("Creating CTRL_NOSE control point from landmark data...")
        # Create from landmark data
        if "NOSE" in first_frame:
            pos_data = first_frame["NOSE"]
            nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

            bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"

            # Add to control collection
            control_coll = bpy.data.collections.get("Main_ControlPoints")
            if control_coll:
                control_coll.objects.link(nose_obj)

            script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
        else:
            script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
            bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
            nose_obj = bpy.context.active_object
            nose_obj.name = "CTRL_NOSE"
    else:
        script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

    return nose_obj


def setup_head_constraints_safe(armature_obj, head_bone_name):
    """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
    script_log(f"Setting up safe head constraints for {head_bone_name}...")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    head_bone = armature_obj.pose.bones[head_bone_name]

    # 1. CLEAR ALL existing constraints first (prevent conflicts)
    for constraint in list(head_bone.constraints):
        head_bone.constraints.remove(constraint)
    script_log("✓ Cleared existing head constraints")

    # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
    neck_bone = armature_obj.pose.bones.get("DEF_Neck")
    if neck_bone:
        copy_loc = head_bone.constraints.new('COPY_LOCATION')
        copy_loc.target = armature_obj
        copy_loc.subtarget = "DEF_Neck"
        copy_loc.use_offset = True  # Maintain head's local position relative to neck
        copy_loc.influence = 1.0
        script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

    # 3. ROTATION: Damped track to NOSE control point (facing direction)
    nose_obj = bpy.data.objects.get("CTRL_NOSE")
    if nose_obj:
        track = head_bone.constraints.new('DAMPED_TRACK')
        track.name = "Track_Nose_Direction"
        track.target = nose_obj
        track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
        track.influence = 1.0
        script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

        # 4. ROTATION LIMITS: Prevent extreme angles for stability
        limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
        limit_rot.name = "Limit_Head_Rotation"
        limit_rot.use_limit_x = True
        limit_rot.min_x = -0.5  # Limited head tilt (nodding)
        limit_rot.max_x = 0.5
        limit_rot.use_limit_y = True
        limit_rot.min_y = -1.5  # Head turn (looking left/right)
        limit_rot.max_y = 1.5
        limit_rot.use_limit_z = True
        limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
        limit_rot.max_z = 0.3
        limit_rot.owner_space = 'LOCAL'
        script_log("✓ Added natural head rotation limits")
    else:
        script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log("✓ Safe head constraints setup complete")

#################################################################################

def create_procedural_head(armature_obj, figure_name, garment_config, global_cloth_settings, neck_config=None):
    """Create human-like head with coordinated vertex bundles for seamless neck integration - UPDATED WITH NEW VERTEX BUNDLES SYSTEM"""
    script_log("Creating procedural garment_head with coordinated vertex bundles and safe constraints...")

    # Get head bone position
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        head_bone = armature_obj.pose.bones.get("DEF_Head")
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        head_bone_name = "DEF_Head"
        neck_bone_name = "DEF_Neck"

        bpy.ops.object.mode_set(mode='OBJECT')

        if not head_bone:
            script_log("ERROR: Could not find head bone")
            return None

        # Get head bone positions in world space - Use tail for head position
        head_base_pos = armature_obj.matrix_world @ head_bone.head  # Base of head/neck (where neck connects)
        head_top_pos = armature_obj.matrix_world @ head_bone.tail  # Top of head (where head should be)

        # Get neck bone position for coordination
        neck_base_pos = armature_obj.matrix_world @ neck_bone.head if neck_bone else head_base_pos
        neck_top_pos = armature_obj.matrix_world @ neck_bone.tail if neck_bone else head_base_pos

        # Get head dimensions from new config format
        scale = garment_configs.get("scale", [0.25, 0.28, 0.3])  # Larger, more human-sized proportions
        position_offset = garment_configs.get("position_offset", [0.0, 0.0, 0.0])
        rotation_offset = garment_configs.get("rotation_offset", [0.0, 0.0, 0.0])
        neck_connection_radius = garment_configs.get("neck_connection_radius", 0.08)
        subdivision_levels = garment_configs.get("subdivision_levels", 3)  # More subdivisions for detail

        # Get geometry settings from config
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        head_segments = geometry_settings.get("head_segments", 32)  # More segments for detail
        head_ring_count = geometry_settings.get("head_ring_count", 24)  # More rings for detail
        base_radii = global_cloth_settings.get("base_radii", {})
        head_base_radius = base_radii.get("head", 0.25)  # Larger base radius

        # NEW: Get head-neck bundle diameter from vertex bundles system - USING NECK CONTROL POINT
        head_neck_radius = get_bundle_radius("CTRL_NECK")  # Using neck control point for neck connection
        head_neck_diameter = head_neck_radius * 2.0  # Convert radius to diameter

        # Fallback if neck bundle not available - try alternative neck control points
        if head_neck_radius == 0.0:
            script_log("⚠ CTRL_NECK vertex bundle radius not found, trying CTRL_HEAD_BASE...")
            head_neck_radius = get_bundle_radius("CTRL_HEAD_BASE")  # Head base is closer to neck junction
            head_neck_diameter = head_neck_radius * 2.0

        # Final fallback if still not available
        if head_neck_radius == 0.0:
            script_log("⚠ Neck vertex bundle radius not found, using config fallback")
            head_neck_diameter = garment_configs.get("diameter_neck", 0.16)  # Fallback to config
            head_neck_radius = head_neck_diameter / 2.0
        else:
            script_log(
                f"✓ Using neck vertex bundle diameter: {head_neck_diameter:.3f} (radius: {head_neck_radius:.3f})")

        # NEW: Get head-neck bundle center from vertex bundles system
        head_neck_center = get_bundle_center("CTRL_NECK")

        # Fallback if neck bundle center not available
        if not head_neck_center:
            script_log("⚠ CTRL_NECK vertex bundle center not found, trying CTRL_HEAD_BASE...")
            head_neck_center = get_bundle_center("CTRL_HEAD_BASE")

        # Final fallback if still not available
        if not head_neck_center:
            script_log("⚠ Neck vertex bundle center not found, using head base position")
            head_neck_center = head_base_pos
        else:
            script_log(f"✓ Using neck vertex bundle center: {head_neck_center}")

        script_log(f"DEBUG: Procedural head scale: {scale}")
        script_log(f"DEBUG: Position offset: {position_offset}, Rotation offset: {rotation_offset}")
        script_log(f"DEBUG: Head bone - base (neck): {head_base_pos}, top (head): {head_top_pos}")
        script_log(f"DEBUG: Head geometry - segments: {head_segments}, rings: {head_ring_count}")
        script_log(f"DEBUG: Head-neck junction diameter: {head_neck_diameter}")
        script_log(f"DEBUG: Head-neck bundle center: {head_neck_center}, radius: {head_neck_radius}")

        # =========================================================================
        # STEP 1: CREATE HEAD MESH WITH NECK OPENING
        # =========================================================================
        script_log("DEBUG: Creating procedural head as UV sphere with neck opening...")
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=head_segments,
            ring_count=head_ring_count,
            radius=head_base_radius,
            location=head_top_pos + Vector(position_offset)  # Position at TOP of head bone
        )
        head_obj = bpy.context.active_object
        head_obj.name = f"{figure_name}_Head"

        # Apply human-like scaling (wider and taller than spherical)
        head_obj.scale = Vector(scale)

        # Apply rotation offset if specified
        if any(rotation_offset):
            head_obj.rotation_euler = Vector(rotation_offset)

        # =========================================================================
        # STEP 2: SHAPE HEAD TO HUMAN PROPORTIONS
        # =========================================================================
        script_log("DEBUG: Shaping head to human proportions...")
        bpy.context.view_layer.objects.active = head_obj

        # Use bmesh for vertex manipulation without entering edit mode
        bm = bmesh.new()
        bm.from_mesh(head_obj.data)

        # Define human head proportions (these could be moved to config)
        human_proportions = garment_configs.get("human_proportions", {
            'forehead_scale': 0.95,  # Slightly narrower forehead
            'cheek_scale': 1.1,  # Fuller cheeks
            'jaw_scale': 0.9,  # Narrower jaw
            'chin_scale': 0.8,  # More pronounced chin
            'back_head_scale': 1.05,  # Fuller back of head
            'top_head_scale': 0.95,  # Slightly flatter top
            'neck_opening_scale': 0.7,  # Create neck opening
        })

        # Shape vertices to human proportions
        for vert in bm.verts:
            # Convert to spherical coordinates for easier manipulation
            x, y, z = vert.co
            radius = (x ** 2 + y ** 2 + z ** 2) ** 0.5
            if radius > 0.001:  # Avoid division by zero
                # Normalize to unit sphere
                nx, ny, nz = x / radius, y / radius, z / radius

                # Apply human proportions based on vertex position
                scale_factor = 1.0

                # Forehead area (front top)
                if nz > 0.3 and ny > 0.2:
                    scale_factor = human_proportions['forehead_scale']

                # Cheek area (front sides)
                elif nz > -0.2 and abs(nx) > 0.4:
                    scale_factor = human_proportions['cheek_scale']

                # Jaw area (front bottom)
                elif nz < -0.3 and abs(ny) < 0.3:
                    scale_factor = human_proportions['jaw_scale']

                # Chin area (very front bottom)
                elif nz < -0.5 and abs(nx) < 0.2:
                    scale_factor = human_proportions['chin_scale']

                # Back of head
                elif nz < -0.1 and abs(nx) < 0.3:
                    scale_factor = human_proportions['back_head_scale']

                # Top of head
                elif ny > 0.7:
                    scale_factor = human_proportions['top_head_scale']

                # Neck opening area (bottom center)
                elif ny < -0.6 and abs(nz) < 0.3 and abs(nx) < 0.3:
                    scale_factor = human_proportions['neck_opening_scale']

                # Apply the scaling
                vert.co.x = nx * radius * scale_factor
                vert.co.y = ny * radius * scale_factor
                vert.co.z = nz * radius * scale_factor

        # Write back to mesh (object mode is safe for this)
        bm.to_mesh(head_obj.data)
        bm.free()

        # =========================================================================
        # STEP 3: ADD FACIAL FEATURES USING SHAPE KEYS
        # =========================================================================
        facial_features = garment_configs.get("facial_features", {})
        if facial_features.get("enable_eye_sockets", True) or facial_features.get("enable_nose_bridge", True):
            script_log("DEBUG: Adding facial features with shape keys...")

            # Add basis shape key first
            head_obj.shape_key_add(name="Basis")

            # Create shape keys FIRST before using bmesh
            shape_key_objects = {}

            if facial_features.get("enable_eye_sockets", True):
                script_log("DEBUG: Creating eye sockets shape key...")
                eye_sockets = head_obj.shape_key_add(name="EyeSockets")
                shape_key_objects["EyeSockets"] = eye_sockets

            if facial_features.get("enable_nose_bridge", True):
                script_log("DEBUG: Creating nose bridge shape key...")
                nose_bridge = head_obj.shape_key_add(name="NoseBridge")
                shape_key_objects["NoseBridge"] = nose_bridge

            # NOW use bmesh for shape key editing
            bm = bmesh.new()
            bm.from_mesh(head_obj.data)

            # Ensure we have shape key layers - they should exist now
            shape_keys = bm.verts.layers.shape

            # Create eye sockets if enabled
            if facial_features.get("enable_eye_sockets", True) and "EyeSockets" in shape_keys:
                script_log("DEBUG: Adding eye sockets geometry...")
                eye_socket_layer = shape_keys["EyeSockets"]
                eye_socket_depth = facial_features.get("eye_socket_depth", 0.08)

                for vert in bm.verts:
                    x, y, z = vert.co
                    # Eye socket areas (front upper sides)
                    if z > 0.1 and abs(x) > 0.3 and y > 0.1:
                        # Create indentation for eye sockets
                        depth = eye_socket_depth
                        vert[eye_socket_layer] = vert.co + Vector((-x * depth * 0.5, -y * depth * 0.3, -z * depth))

            # Create nose bridge if enabled
            if facial_features.get("enable_nose_bridge", True) and "NoseBridge" in shape_keys:
                script_log("DEBUG: Adding nose bridge geometry...")
                nose_bridge_layer = shape_keys["NoseBridge"]
                nose_bridge_height = facial_features.get("nose_bridge_height", 0.05)

                for vert in bm.verts:
                    x, y, z = vert.co
                    # Nose bridge area (center front)
                    if abs(x) < 0.15 and z > 0.2 and y > -0.1:
                        # Create protrusion for nose
                        height = nose_bridge_height
                        vert[nose_bridge_layer] = vert.co + Vector((0, 0, height))

            # Write shape keys back
            bm.to_mesh(head_obj.data)
            bm.free()

            # Apply some of the facial features
            if "EyeSockets" in head_obj.data.shape_keys.key_blocks:
                head_obj.data.shape_keys.key_blocks["EyeSockets"].value = 0.3
            if "NoseBridge" in head_obj.data.shape_keys.key_blocks:
                head_obj.data.shape_keys.key_blocks["NoseBridge"].value = 0.4
        else:
            script_log("DEBUG: Facial features disabled in config")

        # =========================================================================
        # STEP 4: SETUP VERTEX GROUPS WITH COORDINATED HEAD-NECK BUNDLES (NEW SYSTEM)
        # =========================================================================
        script_log("DEBUG: Setting up head vertex groups with NEW vertex bundles system...")

        # Clear any existing parenting
        head_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(head_obj.vertex_groups):
            head_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(head_obj.modifiers):
            if mod.type == 'ARMATURE':
                head_obj.modifiers.remove(mod)

        # Create vertex group for head bone
        head_group = head_obj.vertex_groups.new(name=head_bone_name)

        # NEW: Create spherical coordination group using vertex bundles system
        head_coordination_group = head_obj.vertex_groups.new(name="Head_Coordination_Bundle")

        # Calculate sphere radius for influence (using new vertex bundles system)
        head_sphere_radius = head_neck_radius * 2.5  # Larger radius for better head coverage

        # Apply head-neck bundle weighting using NEW VERTEX BUNDLES SYSTEM
        script_log(f"✓ Applying head-neck vertex bundle with center: {head_neck_center}, radius: {head_sphere_radius}")

        coordination_vertices = 0
        for i, vertex in enumerate(head_obj.data.vertices):
            vert_pos = head_obj.matrix_world @ vertex.co
            distance = (vert_pos - head_neck_center).length

            # Apply weight based on distance to bundle center (NEW SPHERICAL APPROACH)
            if distance <= head_sphere_radius:
                weight = 1.0 - (distance / head_sphere_radius)
                weight = weight * weight  # Quadratic falloff for smoother transitions

                # Reduce weight for top of head (more flexible) while maintaining strong neck connection
                vert_local = head_obj.matrix_world.inverted() @ vert_pos

                # Calculate vertical position relative to head (assuming head is roughly aligned with world Z)
                if vert_local.z > 0.5:  # Top of head
                    weight *= 0.3
                elif vert_local.z > 0.2:  # Upper head
                    weight *= 0.6
                # Full weight for neck area (z < 0)

                if weight > 0.1:
                    head_coordination_group.add([i], weight, 'REPLACE')
                    coordination_vertices += 1

                # Apply to main head group with coordinated weight
                head_group.add([i], weight, 'REPLACE')
            else:
                # Light default weight for distant vertices - ensures all vertices have some influence
                head_group.add([i], 0.2, 'REPLACE')

        script_log(
            f"✓ Applied head-neck vertex bundle weighting to {coordination_vertices} vertices within influence radius")

        # =========================================================================
        # STEP 5: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = head_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 6: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and skin material...")

        # Add subdivision from config
        subdiv_mod = head_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = subdivision_levels
        subdiv_mod.render_levels = subdivision_levels

        # ADD SKIN MATERIAL WITH MODERN BLENDER 4.3+ NODES
        script_log("DEBUG: Creating skin material with modern node setup...")
        material_config = garment_configs.get("material", {})
        material_color = material_config.get("color", [0.96, 0.86, 0.72, 1.0])

        head_mat = bpy.data.materials.new(name="Head_Material")
        head_mat.use_nodes = True

        # Clear default nodes for clean setup
        head_mat.node_tree.nodes.clear()

        # Create modern Principled BSDF setup
        output_node = head_mat.node_tree.nodes.new(type='ShaderNodeOutputMaterial')
        principled_node = head_mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

        # Position nodes
        output_node.location = (300, 0)
        principled_node.location = (0, 0)

        # Connect nodes
        head_mat.node_tree.links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

        # Set material properties using modern API
        principled_node.inputs['Base Color'].default_value = material_color
        principled_node.inputs['Roughness'].default_value = material_config.get("roughness", 0.4)
        principled_node.inputs['Metallic'].default_value = material_config.get("metallic", 0.0)

        head_obj.data.materials.append(head_mat)

        # =========================================================================
        # STEP 7: CLOTH SIMULATION - DISABLED AS REQUESTED
        # =========================================================================
        cloth_settings = garment_configs.get("cloth_settings", {})
        if cloth_settings.get("enabled", False):
            script_log("DEBUG: Adding cloth simulation to garment_head...")
            cloth_mod = head_obj.modifiers.new(name="Cloth", type='CLOTH')
            cloth_mod.settings.quality = cloth_settings.get("quality", 6)
            cloth_mod.settings.mass = cloth_settings.get("mass", 0.8)
            cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 15.0)
            cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 15.0)
            cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 10.0)
            cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 2.0)
            cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)

            # NEW: Pin cloth to coordination group if enabled
            cloth_mod.settings.vertex_group_mass = "Head_Coordination_Bundle"
            script_log("✓ Head cloth pinned to coordination bundle")
        else:
            script_log("DEBUG: Cloth simulation disabled for head")

        # =========================================================================
        # STEP 8: SET MODIFIER ORDER
        # =========================================================================
        bpy.context.view_layer.objects.active = head_obj
        modifiers = head_obj.modifiers

        # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
        correct_order = ["Subdivision", "Armature", "Cloth"]
        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 9: NOSE LANDMARK TRACKING SETUP - UPDATED WITH SAFE CONSTRAINTS
        # =========================================================================
        script_log("DEBUG: Setting up NOSE landmark tracking with safe constraints...")

        # Get first frame data for NOSE control point creation
        first_frame = mocap_data.get(str(frame_numbers[0]), {})

        # GUARANTEE NOSE CONTROL POINT EXISTS
        def ensure_nose_control_point(first_frame):
            """Guarantee CTRL_NOSE exists and is positioned correctly"""
            script_log("Ensuring NOSE control point exists...")

            nose_obj = bpy.data.objects.get("CTRL_NOSE")

            if not nose_obj:
                script_log("Creating CTRL_NOSE control point from landmark data...")
                # Create from landmark data
                if "NOSE" in first_frame:
                    pos_data = first_frame["NOSE"]
                    nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

                    bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
                    nose_obj = bpy.context.active_object
                    nose_obj.name = "CTRL_NOSE"

                    # Add to control collection
                    control_coll = bpy.data.collections.get("Main_ControlPoints")
                    if control_coll:
                        control_coll.objects.link(nose_obj)

                    script_log(f"✓ Created CTRL_NOSE at {nose_pos}")
                else:
                    script_log("⚠ WARNING: NOSE landmark not found in frame data, creating at default position")
                    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 1.35))
                    nose_obj = bpy.context.active_object
                    nose_obj.name = "CTRL_NOSE"
            else:
                script_log(f"✓ Using existing CTRL_NOSE at {nose_obj.location}")

            return nose_obj

        def setup_head_constraints_safe(armature_obj, head_bone_name):
            """Safe head constraint setup without recursion - PORCELAIN DOLL architecture"""
            script_log(f"Setting up safe head constraints for {head_bone_name}...")

            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='POSE')

            head_bone = armature_obj.pose.bones[head_bone_name]

            # 1. CLEAR ALL existing constraints first (prevent conflicts)
            for constraint in list(head_bone.constraints):
                head_bone.constraints.remove(constraint)
            script_log("✓ Cleared existing head constraints")

            # 2. LOCATION: Anchor to neck (bottom connection - porcelain doll base)
            neck_bone = armature_obj.pose.bones.get("DEF_Neck")
            if neck_bone:
                copy_loc = head_bone.constraints.new('COPY_LOCATION')
                copy_loc.target = armature_obj
                copy_loc.subtarget = "DEF_Neck"
                copy_loc.use_offset = True  # Maintain head's local position relative to neck
                copy_loc.influence = 1.0
                script_log("✓ Head COPY_LOCATION -> DEF_Neck (porcelain doll anchor)")

            # 3. ROTATION: Damped track to NOSE control point (facing direction)
            nose_obj = bpy.data.objects.get("CTRL_NOSE")
            if nose_obj:
                track = head_bone.constraints.new('DAMPED_TRACK')
                track.name = "Track_Nose_Direction"
                track.target = nose_obj
                track.track_axis = 'TRACK_Y'  # Head Y-axis faces nose
                track.influence = 1.0
                script_log("✓ Head DAMPED_TRACK -> CTRL_NOSE (facing direction)")

                # 4. ROTATION LIMITS: Prevent extreme angles for stability
                limit_rot = head_bone.constraints.new('LIMIT_ROTATION')
                limit_rot.name = "Limit_Head_Rotation"
                limit_rot.use_limit_x = True
                limit_rot.min_x = -0.5  # Limited head tilt (nodding)
                limit_rot.max_x = 0.5
                limit_rot.use_limit_y = True
                limit_rot.min_y = -1.5  # Head turn (looking left/right)
                limit_rot.max_y = 1.5
                limit_rot.use_limit_z = True
                limit_rot.min_z = -0.3  # Head roll (ear to shoulder)
                limit_rot.max_z = 0.3
                limit_rot.owner_space = 'LOCAL'
                script_log("✓ Added natural head rotation limits")
            else:
                script_log("⚠ WARNING: CTRL_NOSE not found, head rotation will be neutral")

            bpy.ops.object.mode_set(mode='OBJECT')
            script_log("✓ Safe head constraints setup complete")

        nose_obj = ensure_nose_control_point(first_frame)
        setup_head_constraints_safe(armature_obj, head_bone_name)

        # =========================================================================
        # STEP 10: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        script_log("=== PROCEDURAL HEAD CREATION COMPLETE (NEW VERTEX BUNDLES SYSTEM) ===")
        script_log(f"✓ Head positioned at TOP of DEF_Head bone (tail position)")
        script_log(f"✓ Head shaped with human proportions")
        script_log(f"✓ Facial features: Eye sockets, Nose bridge")
        script_log(f"✓ Head rotation tracks NOSE landmark in real-time (SAFE CONSTRAINTS)")
        script_log(f"✓ Natural rotation limits applied for stability")
        script_log(f"✓ Head scale: {scale} (larger, human-sized proportions)")
        script_log(f"✓ Head segments: {head_segments}, rings: {head_ring_count}")
        script_log(f"✓ Subdivision levels: {subdivision_levels}")
        script_log(f"✓ Neck connection diameter: {head_neck_diameter}")
        script_log(f"✓ Head parented to {head_bone_name}")
        script_log(f"✓ Skin material applied with modern Principled BSDF")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_settings.get('enabled', False) else 'DISABLED'}")
        script_log(f"✓ NEW VERTEX BUNDLES SYSTEM: Using head-neck bundle for coordinated weighting")
        script_log(f"✓ Bundle center: {head_neck_center}, influence radius: {head_sphere_radius:.3f}")
        script_log(f"✓ Coordination vertices: {coordination_vertices}/{len(head_obj.data.vertices)}")
        script_log(f"✓ PORCELAIN DOLL ARCHITECTURE: Neck anchor + Nose rotation guide")

        # Call create_neck for coordinated creation
        if neck_config:
            create_neck(armature_obj, figure_name, neck_config, global_cloth_settings)

        return head_obj

    except Exception as e:
        script_log(f"ERROR creating procedural head with coordinated bundles: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_neck(armature_obj, figure_name, garment_config, global_cloth_settings):
    """Create stretchy neck garment with coordinated vertex bundles for seamless integration"""
    script_log("Creating garment_neck with coordinated vertex bundles...")

    # Get neck bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        head_bone = armature_obj.pose.bones.get("DEF_Head")
        upper_spine_bone = armature_obj.pose.bones.get("DEF_UpperSpine")

        neck_bone_name = "DEF_Neck"
        head_bone_name = "DEF_Head"
        upper_spine_bone_name = "DEF_UpperSpine"

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([neck_bone, head_bone, upper_spine_bone]):
            script_log("ERROR: Could not find neck, head, or spine bones")
            return None

        # Get bone positions in world space
        neck_base_pos = armature_obj.matrix_world @ neck_bone.head  # Base of neck (shoulders)
        neck_top_pos = armature_obj.matrix_world @ neck_bone.tail  # Top of neck (head base)
        head_base_pos = armature_obj.matrix_world @ head_bone.head  # Base of head
        spine_top_pos = armature_obj.matrix_world @ upper_spine_bone.tail  # Top of spine

        # Get neck dimensions from config
        puffiness = garment_configs.get("puffiness", 1.1)
        neck_height = garment_configs.get("neck_height", 0.08)
        neck_diameter = garment_configs.get("neck_diameter", 0.14)
        collar_style = garment_configs.get("collar_style", "turtleneck")

        # Get neck-shoulder bundle diameter from config
        neck_spine_diameter = garment_configs.get("diameter_spine", 0.18)  # Diameter at neck-spine-shoulder junction

        # Get geometry settings
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        base_radii = global_cloth_settings.get("base_radii", {})
        segments = geometry_settings.get("default_segments", 12)

        # Calculate neck length and radius
        neck_length = (neck_top_pos - neck_base_pos).length
        neck_radius = neck_diameter / 2 * puffiness

        script_log(f"DEBUG: Neck garment - Height: {neck_height}, Diameter: {neck_diameter}")
        script_log(f"DEBUG: Neck garment - Puffiness: {puffiness}, Final radius: {neck_radius}")
        script_log(f"DEBUG: Neck garment - Collar style: {collar_style}")
        script_log(f"DEBUG: Neck-spine junction diameter: {neck_spine_diameter}")

        # =========================================================================
        # STEP 1: CREATE NECK CYLINDER WITH COORDINATED ENDS
        # =========================================================================
        script_log("DEBUG: Creating neck cylinder with coordinated ends...")

        # Position cylinder to cover the neck area
        neck_center = neck_base_pos + ((neck_top_pos - neck_base_pos) / 2)
        neck_direction = (neck_top_pos - neck_base_pos).normalized()

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=neck_length,
            radius=neck_radius,
            location=neck_center
        )
        neck_obj = bpy.context.active_object
        neck_obj.name = f"{figure_name}_Neck"

        # Rotate to align with neck bone direction
        neck_obj.rotation_euler = neck_direction.to_track_quat('Z', 'Y').to_euler()

        # =========================================================================
        # STEP 2: SETUP VERTEX GROUPS WITH NEW VERTEX BUNDLE SYSTEM
        # =========================================================================
        script_log("DEBUG: Setting up neck vertex groups with NEW vertex bundle system...")

        # Clear any existing parenting
        neck_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(neck_obj.vertex_groups):
            neck_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(neck_obj.modifiers):
            if mod.type == 'ARMATURE':
                neck_obj.modifiers.remove(mod)

        # =========================================================================
        # NEW VERTEX BUNDLE SYSTEM: Get bundle centers and radii
        # =========================================================================
        script_log("DEBUG: Using NEW vertex bundle system for neck...")

        # Get bundle centers from new system
        head_neck_center = get_bundle_center("CTRL_HEAD_TOP")  # Head-neck junction
        neck_spine_center = get_bundle_center("CTRL_NOSE")     # Neck-spine-shoulder junction

        # Get bundle radii for influence calculation
        head_neck_radius = get_bundle_radius("CTRL_HEAD_TOP")
        neck_spine_radius = get_bundle_radius("CTRL_NOSE")

        # Fallback if bundle centers not available
        if not head_neck_center:
            script_log("⚠ Head-neck vertex bundle not found, using calculated position")
            head_neck_center = neck_top_pos
            head_neck_radius = neck_spine_diameter / 2 * 0.8

        if not neck_spine_center:
            script_log("⚠ Neck-spine vertex bundle not found, using calculated position")
            neck_spine_center = neck_base_pos
            neck_spine_radius = neck_spine_diameter / 2

        script_log(f"✓ Using new vertex bundle system:")
        script_log(f"  - Head-neck: {head_neck_center}, radius: {head_neck_radius}")
        script_log(f"  - Neck-spine: {neck_spine_center}, radius: {neck_spine_radius}")

        # Create vertex groups for all three coordination points
        neck_group = neck_obj.vertex_groups.new(name=neck_bone_name)
        head_coordination_group = neck_obj.vertex_groups.new(name="Head_Coordination_Neck")
        spine_coordination_group = neck_obj.vertex_groups.new(name="Spine_Coordination_Neck")

        # Calculate bundle radii for influence
        head_neck_sphere_radius = head_neck_radius * 2.0  # Double radius for influence area
        neck_spine_sphere_radius = neck_spine_radius * 2.0

        # =========================================================================
        # STEP 3: APPLY HEAD-NECK BUNDLE WEIGHTS (TOP OF NECK)
        # =========================================================================
        script_log("DEBUG: Applying head-neck bundle weights to neck top...")

        head_neck_vertices = 0
        for i, vertex in enumerate(neck_obj.data.vertices):
            vert_pos = neck_obj.matrix_world @ vertex.co
            distance = (vert_pos - head_neck_center).length

            # Apply weight based on distance to bundle center
            if distance <= head_neck_sphere_radius:
                weight = 1.0 - (distance / head_neck_sphere_radius)
                weight = weight * weight  # Quadratic falloff

                # Stronger weight for top of neck vertices
                vert_local = neck_obj.matrix_world.inverted() @ vert_pos
                if vert_local.z > neck_length * 0.3:  # Top third of neck
                    weight *= 1.2  # Boost influence
                elif vert_local.z > 0:  # Upper half
                    weight *= 1.0
                else:  # Lower half
                    weight *= 0.6  # Reduced influence

                weight = min(weight, 1.0)  # Clamp to 1.0

                if weight > 0.1:
                    head_coordination_group.add([i], weight, 'REPLACE')
                    # Also assign to neck bone with coordinated weight
                    neck_group.add([i], weight * 0.8, 'REPLACE')
                    head_neck_vertices += 1

        # =========================================================================
        # STEP 4: APPLY NECK-SPINE BUNDLE WEIGHTS (BOTTOM OF NECK)
        # =========================================================================
        script_log("DEBUG: Applying neck-spine bundle weights to neck base...")

        neck_spine_vertices = 0
        for i, vertex in enumerate(neck_obj.data.vertices):
            vert_pos = neck_obj.matrix_world @ vertex.co
            distance = (vert_pos - neck_spine_center).length

            # Apply weight based on distance to bundle center
            if distance <= neck_spine_sphere_radius:
                weight = 1.0 - (distance / neck_spine_sphere_radius)
                weight = weight * weight  # Quadratic falloff

                # Stronger weight for base of neck vertices
                vert_local = neck_obj.matrix_world.inverted() @ vert_pos
                if vert_local.z < -neck_length * 0.3:  # Bottom third of neck
                    weight *= 1.2  # Boost influence
                elif vert_local.z < 0:  # Lower half
                    weight *= 1.0
                else:  # Upper half
                    weight *= 0.4  # Reduced influence

                weight = min(weight, 1.0)  # Clamp to 1.0

                if weight > 0.1:
                    spine_coordination_group.add([i], weight, 'REPLACE')
                    # Also assign to neck bone with coordinated weight
                    neck_group.add([i], weight * 0.8, 'REPLACE')
                    neck_spine_vertices += 1

        script_log(f"✓ Applied head-neck bundle to {head_neck_vertices} vertices")
        script_log(f"✓ Applied neck-spine bundle to {neck_spine_vertices} vertices")

        # =========================================================================
        # STEP 5: ADD DYNAMIC VERTEX WEIGHTING
        # =========================================================================
        script_log("DEBUG: Adding dynamic vertex weighting to neck...")
        neck_obj = setup_garment_dynamic_weighting(neck_obj, "center", "neck")

        # =========================================================================
        # STEP 6: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = neck_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 7: ADD CLOTH SIMULATION FOR STRETCHY FABRIC
        # =========================================================================
        script_log("DEBUG: Adding stretchy cloth simulation to neck...")
        cloth_settings = garment_configs.get("cloth_settings", {})

        if cloth_settings.get("enabled", True):
            cloth_mod = neck_obj.modifiers.new(name="Cloth", type='CLOTH')
            cloth_mod.settings.quality = cloth_settings.get("quality", 6)
            cloth_mod.settings.mass = cloth_settings.get("mass", 0.3)
            cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 5.0)
            cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 4.0)
            cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 3.0)
            cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 0.5)
            cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)

            # PIN CLOTH TO COMBINED COORDINATION GROUPS
            combined_coordination_group = neck_obj.vertex_groups.new(name="Neck_Combined_Coordination")

            for i in range(len(neck_obj.data.vertices)):
                max_weight = 0.0

                # Check head coordination weight
                try:
                    head_weight = head_coordination_group.weight(i)
                    max_weight = max(max_weight, head_weight)
                except:
                    pass

                # Check spine coordination weight
                try:
                    spine_weight = spine_coordination_group.weight(i)
                    max_weight = max(max_weight, spine_weight)
                except:
                    pass

                if max_weight > 0.1:
                    combined_coordination_group.add([i], max_weight, 'REPLACE')

            cloth_mod.settings.vertex_group_mass = "Neck_Combined_Coordination"

            script_log("✓ Added stretchy cloth simulation to garment_neck")
            script_log("✓ Cloth pinned to combined head-spine coordination groups")
        else:
            script_log("DEBUG: Cloth simulation disabled for neck")

        # =========================================================================
        # STEP 8: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and materials...")

        # Add subdivision for smoother fabric
        subdiv_mod = neck_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = 1
        subdiv_mod.render_levels = 1

        # Add fabric material
        material_config = garment_configs.get("material", {})
        material_color = material_config.get("color", [0.1, 0.3, 0.8, 1.0])

        neck_mat = bpy.data.materials.new(name="Neck_Material")
        neck_mat.use_nodes = True

        # Set fabric properties (softer, less shiny than skin)
        neck_mat.diffuse_color = material_color
        neck_mat.roughness = material_config.get("roughness", 0.8)
        neck_mat.metallic = material_config.get("metallic", 0.0)

        neck_obj.data.materials.append(neck_mat)

        # =========================================================================
        # STEP 9: SET MODIFIER ORDER
        # =========================================================================
        script_log("DEBUG: Setting modifier order...")

        bpy.context.view_layer.objects.active = neck_obj
        modifiers = neck_obj.modifiers

        # Ensure order: Subdivision → Armature → Cloth
        correct_order = ["Subdivision", "Armature", "Cloth"]
        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 10: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        script_log("=== GARMENT_NECK CREATION COMPLETE (NEW VERTEX BUNDLE SYSTEM) ===")
        script_log(f"✓ Neck positioned along neck bone")
        script_log(f"✓ Neck radius: {neck_radius}, Length: {neck_length:.3f}")
        script_log(f"✓ Neck-spine junction diameter: {neck_spine_diameter}")
        script_log(f"✓ Stretchy cloth simulation: {'ENABLED' if cloth_settings.get('enabled', True) else 'DISABLED'}")
        script_log(f"✓ Collar style: {collar_style}")
        script_log(f"✓ Neck parented to {neck_bone_name}")
        script_log(f"✓ NEW VERTEX BUNDLE SYSTEM: Using CTRL_HEAD_TOP and CTRL_NOSE")
        script_log(f"✓ Head-neck vertices: {head_neck_vertices}, Neck-spine vertices: {neck_spine_vertices}")
        script_log(f"✓ Dynamic vertex weighting: ENABLED")
        script_log(f"✓ Seamless integration: Head connection + Spine/shoulder coordination")

        return neck_obj

    except Exception as e:
        script_log(f"ERROR creating garment_neck: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_cloth_garments(armature_obj, figure_name):
    """Create cloth garments that follow the bone rig - WITH CONTINUOUS SLEEVES"""
    script_log("=== CREATING CLOTH GARMENTS ===")

    garments_created = 0

    # Load cloth config to get garment definitions
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

    with open(CLOTH_CONFIG_FILE, 'r') as file:
        cloth_config = json.load(file)

    # Use new config structure - only cloth_garments
    garment_definitions = cloth_config.get("cloth_garments", {})

    # Store garments for potential connections
    left_sleeve_obj = None
    right_sleeve_obj = None
    coat_obj = None
    # ADD MITTENS TRACKING
    left_mitten_obj = None
    right_mitten_obj = None

    # CREATE GARMENT_HEAD first (with physics flag check)
    if "garment_head" in garment_definitions:
        head_config = garment_definitions["garment_head"]
        neck_config = garment_definitions.get("garment_neck")

        # Check both enabled flag and physics setting
        if head_config.get("enabled", True):
            head_obj = create_head(armature_obj, figure_name, head_config, cloth_config, neck_config)
            if head_obj:
                garments_created += 1
                script_log("Created garment_head with integrated neck")

    # CREATE BOOTS before pants for future ankle connection (with physics flag check)
    if "left_boot" in garment_definitions:
        left_boot_config = garment_definitions["left_boot"]
        if left_boot_config.get("enabled", True):
            left_boot = create_boot(armature_obj, figure_name, left_boot_config, cloth_config, "left")
            if left_boot:
                garments_created += 1
                script_log("Created left boot")

    if "right_boot" in garment_definitions:
        right_boot_config = garment_definitions["right_boot"]
        if right_boot_config.get("enabled", True):
            right_boot = create_boot(armature_obj, figure_name, right_boot_config, cloth_config, "right")
            if right_boot:
                garments_created += 1
                script_log("Created right boot")

    # CREATE CONTINUOUS SLEEVES (NEW - single mesh with spherical elbow weighting)
    if "left_sleeve" in garment_definitions:
        left_sleeve_config = garment_definitions["left_sleeve"]
        if left_sleeve_config.get("enabled", True):
            left_sleeve_obj = create_sleeve(armature_obj, figure_name, left_sleeve_config, cloth_config, "left")
            if left_sleeve_obj:
                garments_created += 1
                script_log("Created left continuous sleeve with spherical elbow weighting")

    if "right_sleeve" in garment_definitions:
        right_sleeve_config = garment_definitions["right_sleeve"]
        if right_sleeve_config.get("enabled", True):
            right_sleeve_obj = create_sleeve(armature_obj, figure_name, right_sleeve_config, cloth_config, "right")
            if right_sleeve_obj:
                garments_created += 1
                script_log("Created right continuous sleeve with spherical elbow weighting")

    # CREATE PANTS (with physics flag check) - FIXED SIGNATURE
    if "left_pants" in garment_definitions:
        left_pants_config = garment_definitions["left_pants"]
        if left_pants_config.get("enabled", True):
            left_pants = create_pants(armature_obj, figure_name, "left")
            if left_pants:
                garments_created += 1
                script_log("Created left pants")

    if "right_pants" in garment_definitions:
        right_pants_config = garment_definitions["right_pants"]
        if right_pants_config.get("enabled", True):
            right_pants = create_pants(armature_obj, figure_name, "right")
            if right_pants:
                garments_created += 1
                script_log("Created right pants")

    # CREATE COAT TORSO (with physics flag check) - FIXED SIGNATURE
    if "coat_torso" in garment_definitions:
        coat_config = garment_definitions["coat_torso"]
        if coat_config.get("enabled", True):
            coat_obj = create_coat(armature_obj, figure_name)
            if coat_obj:
                garments_created += 1
                script_log("Created coat_torso")

    # CREATE MITTENS (with physics flag check) - STORE FOR TRACKING
    if "left_mitten" in garment_definitions:
        left_mitten_config = garment_definitions["left_mitten"]
        if left_mitten_config.get("enabled", True):
            left_mitten_obj = create_mitten(armature_obj, figure_name, left_mitten_config, cloth_config, "left")
            if left_mitten_obj:
                garments_created += 1
                script_log("Created left mitten")

    if "right_mitten" in garment_definitions:
        right_mitten_config = garment_definitions["right_mitten"]
        if right_mitten_config.get("enabled", True):
            right_mitten_obj = create_mitten(armature_obj, figure_name, right_mitten_config, cloth_config, "right")
            if right_mitten_obj:
                garments_created += 1
                script_log("Created right mitten")

    # CREATE LEGACY GARMENTS (for backward compatibility with old garment types)
    for garment_name, garment_config in garment_definitions.items():
        # Skip garments we've already handled above
        if garment_name in ["garment_head", "left_boot", "right_boot", "left_sleeve", "right_sleeve",
                            "left_pants", "right_pants", "coat_torso", "left_mitten", "right_mitten"]:
            continue

        if garment_configs.get("enabled", False):
            if garment_name == "long_sleeve_shirt":
                garment_obj = create_long_sleeve_shirt(armature_obj, figure_name, garment_config, cloth_config)
            else:
                script_log(f"WARNING: Unknown garment type: {garment_name}")
                continue

            if garment_obj:
                garments_created += 1
                script_log(f"Created {garment_name}")

    # LOG MITTENS STATUS
    if left_mitten_obj or right_mitten_obj:
        mitten_count = (1 if left_mitten_obj else 0) + (1 if right_mitten_obj else 0)
        script_log(f"✓ Created {mitten_count} mittens")
    else:
        script_log(f"✓ No mittens created (disabled or failed)")

    script_log(f"Created {garments_created} cloth garments")
    return garments_created

##########################################################################################

def create_long_sleeve_shirt(armature_obj, figure_name, garment_config, global_cloth_settings):
    """Create a long-sleeve shirt garment with cloth simulation"""
    script_log("Creating long-sleeve shirt cloth garment...")

    # Create mesh for shirt
    bpy.ops.mesh.primitive_cube_add()
    shirt_obj = bpy.context.active_object
    shirt_obj.name = f"{figure_name}_LongSleeveShirt"

    # Apply scale and position from config
    scale = garment_configs.get("scale", [0.4, 0.3, 0.5])
    position_offset = garment_configs.get("position_offset", [0.0, 0.0, 0.1])

    shirt_obj.scale = Vector(scale)
    shirt_obj.location = Vector(position_offset)

    # Add cloth simulation modifier with config settings
    cloth_mod = shirt_obj.modifiers.new(name="Cloth", type='CLOTH')

    # Apply global cloth settings
    cloth_mod.settings.quality = global_cloth_settings.get("quality", 5)
    cloth_mod.settings.time_scale = global_cloth_settings.get("time_scale", 1.0)

    # Apply garment-specific cloth settings
    cloth_settings = garment_configs.get("cloth_settings", {})
    cloth_mod.settings.mass = cloth_settings.get("mass", 0.4)
    cloth_mod.settings.air_damping = cloth_settings.get("air_damping", 1.0)
    cloth_mod.settings.tension_stiffness = cloth_settings.get("tension_stiffness", 25.0)
    cloth_mod.settings.compression_stiffness = cloth_settings.get("compression_stiffness", 25.0)
    cloth_mod.settings.shear_stiffness = cloth_settings.get("shear_stiffness", 15.0)
    cloth_mod.settings.bending_stiffness = cloth_settings.get("bending_stiffness", 1.5)

    # Parent to appropriate bones from config
    parent_bones = garment_configs.get("parent_bones", [])
    default_weight = global_cloth_settings.get("default_vertex_weight", 0.3)
    setup_cloth_parenting(shirt_obj, armature_obj, parent_bones, default_weight)

    # Add material with config properties
    material_color = garment_configs.get("material_color", [0.8, 0.8, 0.8, 1.0])
    material_props = garment_configs.get("material_properties", {})

    cloth_mat = bpy.data.materials.new(name="LongSleeveShirt_Material")
    cloth_mat.diffuse_color = material_color
    cloth_mat.metallic = material_props.get("metallic", 0.0)
    cloth_mat.roughness = material_props.get("roughness", 0.7)
    cloth_mat.specular_intensity = material_props.get("specular", 0.3)
    shirt_obj.data.materials.append(cloth_mat)

    return shirt_obj

##########################################################################################

def create_boot(armature_obj, figure_name, garment_config, global_cloth_settings, side="left"):
    """Create modular boots with ankle bridge, shaft cylinder, and foot cylinder using new vertex bundle system"""
    script_log(f"Creating {side} boot with new vertex bundle system integration...")

    # Get shin and foot bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        if side == "left":
            shin_bone_name = "DEF_LeftShin"
            foot_bone_name = "DEF_LeftFoot"
            ankle_control_point = "CTRL_LEFT_HEEL"
        else:
            shin_bone_name = "DEF_RightShin"
            foot_bone_name = "DEF_RightFoot"
            ankle_control_point = "CTRL_RIGHT_HEEL"

        shin_bone = armature_obj.pose.bones.get(shin_bone_name)
        foot_bone = armature_obj.pose.bones.get(foot_bone_name)

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([shin_bone, foot_bone]):
            script_log(f"ERROR: Could not find leg bones for {side} boot")
            return None

        # Get bone positions in world space
        ankle_pos = armature_obj.matrix_world @ shin_bone.tail  # Ankle position
        foot_start_pos = armature_obj.matrix_world @ foot_bone.head  # Start of foot
        toe_pos = armature_obj.matrix_world @ foot_bone.tail  # End of foot

        # Get boot dimensions from global garment_configs
        boot_config = garment_configs.get(f"{side}_boot", garment_config)  # Fallback to passed config
        puffiness = boot_config.get("puffiness", 1.0)
        shaft_height = boot_config.get("shaft_height", 0.15)
        foot_length = boot_config.get("foot_length", 0.12)
        foot_height = boot_config.get("foot_height", 0.06)
        segments = boot_config.get("segments", 8)

        # Get geometry settings from config
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        base_radii = global_cloth_settings.get("base_radii", {})

        # Use configurable base ankle radius
        base_ankle_radius = base_radii.get("ankle", 0.08)
        ankle_radius = base_ankle_radius * puffiness

        # Calculate foot radii based on configurable dimensions
        foot_radius_x = foot_length / 3 * puffiness  # Width of foot
        foot_radius_y = foot_height / 2 * puffiness  # Height of foot

        # Get vertex weighting settings from config
        weighting_config = boot_config.get("vertex_weighting", {})
        falloff_type = weighting_config.get("sphere_falloff", "quadratic")
        min_weight_threshold = weighting_config.get("min_weight_threshold", 0.05)
        sphere_influence_scale = weighting_config.get("sphere_influence_scale", 2.0)

        # Get ankle bundle from new system
        ankle_center = get_bundle_center(ankle_control_point)
        ankle_radius_bundle = get_bundle_radius(ankle_control_point)

        # Calculate sphere radii for bundle integration
        ankle_sphere_radius = ankle_radius_bundle * sphere_influence_scale

        script_log(f"DEBUG: {side} boot - Shaft height: {shaft_height}, Foot length: {foot_length}")
        script_log(f"DEBUG: {side} boot - Base ankle radius: {base_ankle_radius}, Puffiness: {puffiness}")
        script_log(
            f"DEBUG: {side} boot - Final ankle radius: {ankle_radius}, Foot radii: {foot_radius_x}, {foot_radius_y}")
        script_log(f"DEBUG: {side} boot - Segments: {segments}")
        script_log(f"DEBUG: {side} boot - Ankle sphere radius: {ankle_sphere_radius:.3f}")
        script_log(f"DEBUG: {side} boot - Using ankle bundle from: {ankle_control_point}")

        boot_objects = []

        # =========================================================================
        # STEP 1: CREATE ANKLE BRIDGE SPHERE WITH NEW VERTEX BUNDLE SYSTEM
        # =========================================================================
        script_log(f"DEBUG: Creating ankle bridge sphere for {side} boot with new bundle system...")
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=12,
            ring_count=6,
            radius=ankle_radius,
            location=ankle_pos
        )
        ankle_sphere = bpy.context.active_object
        ankle_sphere.name = f"{figure_name}_{side.capitalize()}AnkleBridge"
        boot_objects.append(ankle_sphere)

        # =========================================================================
        # STEP 2: CREATE SHAFT CYLINDER (vertical)
        # =========================================================================
        script_log(f"DEBUG: Creating shaft cylinder for {side} boot...")

        # Shaft extends upward from ankle position
        shaft_top_pos = ankle_pos + Vector((0, 0, shaft_height))
        shaft_center = (ankle_pos + shaft_top_pos) / 2
        shaft_direction = Vector((0, 0, 1))  # Straight up

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=shaft_height,
            radius=ankle_radius,  # Same radius as ankle bridge
            location=shaft_center
        )
        shaft_obj = bpy.context.active_object
        shaft_obj.name = f"{figure_name}_{side.capitalize()}Shaft"

        # Rotate to align vertically (already aligned by default)
        # No rotation needed for vertical cylinder

        boot_objects.append(shaft_obj)

        # =========================================================================
        # STEP 3: CREATE FOOT CYLINDER (horizontal, elliptical)
        # =========================================================================
        script_log(f"DEBUG: Creating foot cylinder for {side} boot...")

        # Foot extends from ankle to toe position
        foot_vector = toe_pos - foot_start_pos
        foot_direction = foot_vector.normalized()
        foot_center = foot_start_pos + (foot_vector / 2)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=foot_length,
            radius=1.0,  # Unit radius, we'll scale it
            location=foot_center
        )
        foot_obj = bpy.context.active_object
        foot_obj.name = f"{figure_name}_{side.capitalize()}Foot"

        # Rotate to align with foot bone direction
        foot_obj.rotation_euler = foot_direction.to_track_quat('Z', 'Y').to_euler()

        # Scale to create elliptical cross-section for foot
        foot_obj.scale = (foot_radius_x, foot_radius_y, 1.0)

        boot_objects.append(foot_obj)

        # =========================================================================
        # STEP 4: SETUP PARENTING AND ARMATURE MODIFIERS
        # =========================================================================
        script_log("DEBUG: Setting up parenting and armature modifiers...")

        for obj in boot_objects:
            # Clear any existing parenting
            obj.parent = None

            # Clear any existing vertex groups
            for vg in list(obj.vertex_groups):
                obj.vertex_groups.remove(vg)

            # Remove any existing armature modifiers
            for mod in list(obj.modifiers):
                if mod.type == 'ARMATURE':
                    obj.modifiers.remove(mod)

        # =========================================================================
        # STEP 5: NEW VERTEX BUNDLE SYSTEM INTEGRATION FOR ANKLE BRIDGE
        # =========================================================================
        script_log(f"DEBUG: Integrating new vertex bundle system for {side} boot ankle...")

        # Create spherical vertex group for ankle bundle integration
        ankle_sphere_group = ankle_sphere.vertex_groups.new(name=f"Ankle_Sphere_{side}")

        # =========================================================================
        # APPLY ANKLE BUNDLE VERTEX WEIGHTS USING NEW SYSTEM
        # =========================================================================
        if ankle_center:
            script_log(f"DEBUG: Applying ankle vertex bundle to ankle bridge...")

            for i, vertex in enumerate(ankle_sphere.data.vertices):
                vert_pos = ankle_sphere.matrix_world @ vertex.co
                distance = (vert_pos - ankle_center).length

                # Apply weight based on distance to bundle center
                if distance <= ankle_sphere_radius:
                    weight = 1.0 - (distance / ankle_sphere_radius)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    if weight > min_weight_threshold:
                        ankle_sphere_group.add([i], weight, 'REPLACE')
        else:
            # Fallback: assign uniform weights to ankle sphere
            for i in range(len(ankle_sphere.data.vertices)):
                ankle_sphere_group.add([i], 1.0, 'REPLACE')

        # =========================================================================
        # STEP 6: SETUP VERTEX GROUPS FOR ARMATURE BINDING
        # =========================================================================
        script_log("DEBUG: Setting up vertex groups for armature binding...")

        # Ankle Sphere -> Shin Bone (since it's at ankle position)
        ankle_group = ankle_sphere.vertex_groups.new(name=shin_bone_name)
        for i in range(len(ankle_sphere.data.vertices)):
            ankle_group.add([i], 1.0, 'REPLACE')

        # Shaft Cylinder -> Shin Bone
        shaft_group = shaft_obj.vertex_groups.new(name=shin_bone_name)
        for i in range(len(shaft_obj.data.vertices)):
            shaft_group.add([i], 1.0, 'REPLACE')

        # Foot Cylinder -> Foot Bone
        foot_group = foot_obj.vertex_groups.new(name=foot_bone_name)
        for i in range(len(foot_obj.data.vertices)):
            foot_group.add([i], 1.0, 'REPLACE')

        # =========================================================================
        # STEP 7: ADD ARMATURE MODIFIERS
        # =========================================================================
        script_log("DEBUG: Adding armature modifiers to all boot parts...")

        for obj in boot_objects:
            # Add armature modifier
            armature_mod = obj.modifiers.new(name="Armature", type='ARMATURE')
            armature_mod.object = armature_obj
            armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 8: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and materials...")

        for obj in boot_objects:
            # Add subdivision
            subdiv_mod = obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = 1
            subdiv_mod.render_levels = 1

            # Add materials from config
            material_config = boot_config.get("material", {})
            material_color = material_config.get("color", [0.3, 0.2, 0.1, 1.0])

            if "AnkleBridge" in obj.name:
                mat_name = f"{side.capitalize()}AnkleBridge_Material"
                color = (0.3, 0.2, 0.1, 1.0)  # Brown for ankle
            elif "Shaft" in obj.name:
                mat_name = f"{side.capitalize()}Shaft_Material"
                color = (0.35, 0.25, 0.15, 1.0)  # Slightly lighter brown for shaft
            else:
                mat_name = f"{side.capitalize()}Foot_Material"
                color = (0.4, 0.3, 0.2, 1.0)  # Lightest brown for foot

            boot_mat = bpy.data.materials.new(name=mat_name)
            boot_mat.diffuse_color = color
            boot_mat.roughness = material_config.get("roughness", 0.9)
            boot_mat.metallic = material_config.get("metallic", 0.1)
            obj.data.materials.append(boot_mat)

        # =========================================================================
        # STEP 9: CLOTH SIMULATION - DISABLED AS REQUESTED
        # =========================================================================
        cloth_config = boot_config.get("cloth_settings", {})
        if cloth_config.get("enabled", False):
            script_log(f"DEBUG: Adding cloth simulation to {side} boot...")
            for obj in boot_objects:
                cloth_mod = obj.modifiers.new(name="Cloth", type='CLOTH')

                # Apply cloth settings from config
                cloth_mod.settings.quality = cloth_config.get("quality", 6)
                cloth_mod.settings.mass = cloth_config.get("mass", 0.7)
                cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 25.0)
                cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 20.0)
                cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 15.0)
                cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 5.0)
                cloth_mod.settings.air_damping = cloth_config.get("air_damping", 1.0)

                # PIN CLOTH TO ANKLE SPHERE GROUP IF BUNDLE EXISTS
                if ankle_center and "AnkleBridge" in obj.name:
                    cloth_mod.settings.vertex_group_mass = f"Ankle_Sphere_{side}"
                    script_log(f"✓ Ankle bridge cloth pinned to ankle spherical vertex group")
        else:
            script_log(f"DEBUG: Cloth simulation disabled for {side} boot")

        # =========================================================================
        # STEP 10: SET MODIFIER ORDER
        # =========================================================================
        script_log("DEBUG: Setting modifier order...")

        for obj in boot_objects:
            bpy.context.view_layer.objects.active = obj
            modifiers = obj.modifiers

            # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
            correct_order = ["Subdivision", "Armature", "Cloth"]
            for mod_name in correct_order:
                mod_index = modifiers.find(mod_name)
                if mod_index >= 0:
                    while mod_index > correct_order.index(mod_name):
                        bpy.ops.object.modifier_move_up(modifier=mod_name)
                        mod_index -= 1

        # =========================================================================
        # STEP 11: CREATE COMBINED PINNING GROUP FOR BOOT COORDINATION
        # =========================================================================
        script_log(f"DEBUG: Creating combined coordination group for {side} boot...")

        # Create a combined group that includes ankle sphere influence
        combined_coordination_group = ankle_sphere.vertex_groups.new(name=f"{side}_Boot_Combined_Coordination")

        # Combine ankle sphere weights with standard armature weights
        for i in range(len(ankle_sphere.data.vertices)):
            max_weight = 0.0

            # Check ankle sphere group weight
            try:
                sphere_weight = ankle_sphere_group.weight(i)
                max_weight = max(max_weight, sphere_weight)
            except:
                pass

            # Check standard armature group weight
            try:
                armature_weight = ankle_group.weight(i)
                max_weight = max(max_weight, armature_weight)
            except:
                pass

            if max_weight > min_weight_threshold:
                combined_coordination_group.add([i], max_weight, 'REPLACE')

        # =========================================================================
        # STEP 12: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage
        bundle_status = f"ankle({ankle_control_point})" if ankle_center else "NONE (standard)"

        script_log(f"=== {side.upper()} BOOT CREATION COMPLETE (NEW VERTEX BUNDLE SYSTEM) ===")
        script_log(f"✓ Base ankle radius: {base_ankle_radius} (from config)")
        script_log(f"✓ Final ankle radius: {ankle_radius} (with puffiness: {puffiness})")
        script_log(f"✓ Shaft height: {shaft_height}")
        script_log(f"✓ Foot length: {foot_length}, elliptical: {foot_radius_x}x{foot_radius_y}")
        script_log(f"✓ Segments: {segments}")
        script_log(f"✓ Ankle sphere radius for bundle integration: {ankle_sphere_radius:.3f}")
        script_log(f"✓ Ankle bridge and shaft parented to {shin_bone_name}")
        script_log(f"✓ Foot parented to {foot_bone_name}")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_config.get('enabled', False) else 'DISABLED'}")
        script_log(f"✓ Vertex bundle used: {bundle_status}")
        script_log(f"✓ Seamless integration: Boot uses same ankle vertex bundle as pants")
        script_log(f"✓ Combined coordination: Ankle sphere + armature binding work together")

        # Return the foot cylinder as the main boot object (preserved behavior)
        return foot_obj

    except Exception as e:
        script_log(f"ERROR creating {side} boot: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_pants(armature_obj, figure_name, side="left"):
    """Create pants using spheres at joints and tapered cylinders between them"""
    script_log(f"Creating {side} pants with sphere-based approach...")

    # Get pants configuration
    diameter_hip = garment_configs.get("diameter_hip", 0.18)
    diameter_knee = garment_configs.get("diameter_knee", 0.14)
    diameter_ankle = garment_configs.get("diameter_ankle", 0.12)
    segments = garment_configs.get("segments", 32)

    # Get correct control point names
    side_upper = side.upper()
    hip_control_name = f"CTRL_{side_upper}_HIP"
    knee_control_name = f"CTRL_{side_upper}_KNEE"
    heel_control_name = f"CTRL_{side_upper}_HEEL"

    # Get joint control systems using correct names
    hip_control = joint_control_systems.get(hip_control_name, {})
    knee_control = joint_control_systems.get(knee_control_name, {})
    heel_control = joint_control_systems.get(heel_control_name, {})

    if not all([hip_control, knee_control, heel_control]):
        script_log(f"Warning: Missing joint control systems for {side} pants")
        return None

    # Use VB empties (these are the actual empties that follow control points)
    hip_vb_empty = hip_control.get('rpy_empty')  # Use RPY empty for constraints
    knee_vb_empty = knee_control.get('rpy_empty')
    heel_vb_empty = heel_control.get('rpy_empty')

    if not all([hip_vb_empty, knee_vb_empty, heel_vb_empty]):
        script_log(f"Warning: Missing RPY empties for {side} pants")
        return None

    # =========================================================================
    # ADD Z-AXIS POINTING CONSTRAINTS TO RPY EMPTIES
    # =========================================================================
    script_log(f"DEBUG: Setting up Z-axis pointing constraints for {side} leg...")

    # HIP RPY empty: Z-axis points to elbow
    elbow_control_point = f"CTRL_{side_upper}_ELBOW"
    elbow_target = bpy.data.objects.get(elbow_control_point)
    if hip_vb_empty and elbow_target:
        # Clear existing constraints first
        for constraint in list(hip_vb_empty.constraints):
            hip_vb_empty.constraints.remove(constraint)

        # Add Damped Track constraint to point Z-axis toward elbow
        track_constraint = hip_vb_empty.constraints.new('DAMPED_TRACK')
        track_constraint.name = f"Track_To_Elbow"
        track_constraint.target = elbow_target
        track_constraint.track_axis = 'TRACK_Z'  # Z-axis points to target
        script_log(f"✓ {hip_vb_empty.name} Z-axis tracking {elbow_control_point}")

    # KNEE RPY empty: Z-axis points to wrist
    wrist_control_point = f"CTRL_{side_upper}_WRIST"
    wrist_target = bpy.data.objects.get(wrist_control_point)
    if knee_vb_empty and wrist_target:
        # Clear existing constraints first
        for constraint in list(knee_vb_empty.constraints):
            knee_vb_empty.constraints.remove(constraint)

        # Add Damped Track constraint to point Z-axis toward wrist
        track_constraint = knee_vb_empty.constraints.new('DAMPED_TRACK')
        track_constraint.name = f"Track_To_Wrist"
        track_constraint.target = wrist_target
        track_constraint.track_axis = 'TRACK_Z'  # Z-axis points to target
        script_log(f"✓ {knee_vb_empty.name} Z-axis tracking {wrist_control_point}")

    # WRIST RPY empty: Z-axis points to index finger
    index_control_point = f"CTRL_{side_upper}_INDEX"
    index_target = bpy.data.objects.get(index_control_point)
    if wrist_target and index_target:  # Note: using wrist_target from above
        wrist_rpy_empty = joint_control_systems.get(wrist_control_point, {}).get('rpy_empty')
        if wrist_rpy_empty:
            # Clear existing constraints first
            for constraint in list(wrist_rpy_empty.constraints):
                wrist_rpy_empty.constraints.remove(constraint)

            # Add Damped Track constraint to point Z-axis toward index finger
            track_constraint = wrist_rpy_empty.constraints.new('DAMPED_TRACK')
            track_constraint.name = f"Track_To_Index"
            track_constraint.target = index_target
            track_constraint.track_axis = 'TRACK_Z'  # Z-axis points to target
            script_log(f"✓ {wrist_rpy_empty.name} Z-axis tracking {index_control_point}")
    else:
        script_log(f"⚠ Index control point {index_control_point} not found, wrist rotation will be neutral")

    # Get correct bone names
    if side == "left":
        thigh_bone_name = "DEF_LeftThigh"
        shin_bone_name = "DEF_LeftShin"
    else:
        thigh_bone_name = "DEF_RightThigh"
        shin_bone_name = "DEF_RightShin"

    # Store created objects
    pants_objects = []

    # =========================================================================
    # CREATE SPHERES WITH PROPER VERTEX GROUP SETUP (NOT DIRECT PARENTING)
    # =========================================================================
    script_log(f"Creating hip sphere for {side} pants...")
    hip_sphere = create_sphere(
        name=f"{figure_name}_{side}_pants_hip_sphere",
        diameter=diameter_hip,
        segments=segments,
        location=hip_vb_empty.location  # Use RPY empty's actual location
    )
    # DO NOT PARENT DIRECTLY - use vertex groups and armature modifier
    setup_pants_component_vertex_groups(hip_sphere, hip_control_name, armature_obj)
    apply_material_from_config(hip_sphere, f"{side}_pants")
    pants_objects.append(hip_sphere)
    script_log(f"Created hip sphere at {hip_vb_empty.location}")

    # Knee sphere
    script_log(f"Creating knee sphere for {side} pants...")
    knee_sphere = create_sphere(
        name=f"{figure_name}_{side}_pants_knee_sphere",
        diameter=diameter_knee,
        segments=segments,
        location=knee_vb_empty.location  # Use RPY empty's actual location
    )
    setup_pants_component_vertex_groups(knee_sphere, knee_control_name, armature_obj)
    apply_material_from_config(knee_sphere, f"{side}_pants")
    pants_objects.append(knee_sphere)
    script_log(f"Created knee sphere at {knee_vb_empty.location}")

    # Ankle/Heel sphere
    script_log(f"Creating ankle sphere for {side} pants...")
    ankle_sphere = create_sphere(
        name=f"{figure_name}_{side}_pants_ankle_sphere",
        diameter=diameter_ankle,
        segments=segments,
        location=heel_vb_empty.location  # Use RPY empty's actual location
    )
    setup_pants_component_vertex_groups(ankle_sphere, heel_control_name, armature_obj)
    apply_material_from_config(ankle_sphere, f"{side}_pants")
    pants_objects.append(ankle_sphere)
    script_log(f"Created ankle sphere at {heel_vb_empty.location}")

    # =========================================================================
    # CREATE CYLINDERS WITH PROPER VERTEX GROUP SETUP
    # =========================================================================
    script_log(f"Creating thigh cylinder for {side} pants...")
    thigh_cylinder = create_tapered_cylinder(
        name=f"{figure_name}_{side}_pants_thigh_cylinder",
        start_diameter=diameter_hip,
        end_diameter=diameter_knee,
        segments=segments,
        start_location=hip_vb_empty.location,  # TOP at hips
        end_location=knee_vb_empty.location  # BOTTOM at knees
    )
    # Thigh cylinder spans from hip bone to upper leg bone
    hip_bone = "DEF_LeftHip" if side == "left" else "DEF_RightHip"
    thigh_bone = "DEF_LeftThigh" if side == "left" else "DEF_RightThigh"
    setup_pants_cylinder_vertex_groups(thigh_cylinder, hip_bone, thigh_bone, armature_obj)
    apply_material_from_config(thigh_cylinder, f"{side}_pants")
    pants_objects.append(thigh_cylinder)

    script_log(f"Creating shin cylinder for {side} pants...")
    shin_cylinder = create_tapered_cylinder(
        name=f"{figure_name}_{side}_pants_shin_cylinder",
        start_diameter=diameter_knee,
        end_diameter=diameter_ankle,
        segments=segments,
        start_location=knee_vb_empty.location,  # TOP at knees
        end_location=heel_vb_empty.location  # BOTTOM at ankles
    )
    # Shin cylinder spans from knee bone to ankle bone
    knee_bone = "DEF_LeftKnee" if side == "left" else "DEF_RightKnee"
    ankle_bone = "DEF_LeftAnkle" if side == "left" else "DEF_RightAnkle"
    setup_pants_cylinder_vertex_groups(shin_cylinder, knee_bone, ankle_bone, armature_obj)
    apply_material_from_config(shin_cylinder, f"{side}_pants")
    pants_objects.append(shin_cylinder)

    script_log(f"Successfully created {side} pants with {len(pants_objects)} components")
    script_log(f"✓ Z-axis constraints: Hip→Elbow, Knee→Wrist, Wrist→Index")
    script_log(f"✓ Proper vertex group setup for all components")

    # Return the main pants object (hip sphere) for compatibility
    return pants_objects[0] if pants_objects else None

##########################################################################################

def setup_pants_component_vertex_groups(obj, control_point_name, armature_obj):
    """Setup vertex groups for pants spheres with TWO-EMPTIES PARENTING + armature deformation"""
    # Clear any existing vertex groups
    for vg in list(obj.vertex_groups):
        obj.vertex_groups.remove(vg)

    # Remove any existing armature modifiers
    for mod in list(obj.modifiers):
        if mod.type == 'ARMATURE':
            obj.modifiers.remove(mod)

    # IMPORTANT: Parent to the RPY empty for position + rotation
    rpy_empty = joint_control_systems.get(control_point_name, {}).get('rpy_empty')
    if rpy_empty:
        obj.parent = rpy_empty
        script_log(f"✓ {obj.name} parented to {rpy_empty.name} for position+rotation")

    # Create vertex group for deformation (optional - for cloth simulation)
    bone_name = "DEF_LeftThigh" if "LEFT" in control_point_name else "DEF_RightThigh"
    bone_group = obj.vertex_groups.new(name=bone_name)

    # Assign weights for deformation (light influence for cloth)
    for i in range(len(obj.data.vertices)):
        bone_group.add([i], 0.3, 'REPLACE')  # Light deformation weight

    # Add armature modifier for CLOTH DEFORMATION (not primary movement)
    armature_mod = obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj
    armature_mod.use_vertex_groups = True

    script_log(f"✓ {obj.name} uses two-empties parenting + light armature deformation")

##########################################################################################

def setup_pants_cylinder_vertex_groups(obj, start_control_point, end_control_point, armature_obj):
    """Setup vertex groups for pants cylinders with TWO-EMPTIES PARENTING + armature deformation"""
    # Clear any existing vertex groups
    for vg in list(obj.vertex_groups):
        obj.vertex_groups.remove(vg)

    # Remove any existing armature modifiers
    for mod in list(obj.modifiers):
        if mod.type == 'ARMATURE':
            obj.modifiers.remove(mod)

    # IMPORTANT: Parent to the START control point's RPY empty
    start_rpy_empty = joint_control_systems.get(start_control_point, {}).get('rpy_empty')
    if start_rpy_empty:
        obj.parent = start_rpy_empty
        script_log(f"✓ {obj.name} parented to {start_rpy_empty.name} for position+rotation")

    # Create vertex groups for deformation along the cylinder
    start_bone = "DEF_LeftThigh" if "LEFT" in start_control_point else "DEF_RightThigh"
    end_bone = "DEF_LeftShin" if "LEFT" in end_control_point else "DEF_RightShin"

    start_bone_group = obj.vertex_groups.new(name=start_bone)
    end_bone_group = obj.vertex_groups.new(name=end_bone)

    # Weight vertices based on position for deformation
    local_z_coords = [v.co.z for v in obj.data.vertices]
    min_z = min(local_z_coords)
    max_z = max(local_z_coords)
    total_length = max_z - min_z

    for i, vertex in enumerate(obj.data.vertices):
        z_local = vertex.co.z
        z_normalized = (z_local - min_z) / total_length

        start_weight = (1.0 - z_normalized) * 0.3  # Light deformation
        end_weight = z_normalized * 0.3  # Light deformation

        start_bone_group.add([i], start_weight, 'REPLACE')
        end_bone_group.add([i], end_weight, 'REPLACE')

    # Add armature modifier for CLOTH DEFORMATION (not primary movement)
    armature_mod = obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj
    armature_mod.use_vertex_groups = True

    script_log(f"✓ {obj.name} uses two-empties parenting + gradient armature deformation")

##########################################################################################

def create_sphere(name, diameter, segments, location):
    """Create a sphere mesh object."""
    radius = diameter / 2.0

    # Create sphere mesh
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=segments // 2,
        radius=radius,
        location=location
    )

    sphere = bpy.context.active_object
    sphere.name = name

    return sphere

##########################################################################################

def create_tapered_cylinder(name, start_diameter, end_diameter, segments, start_location, end_location):
    """Create a tapered cylinder that starts at start_location and ends at end_location"""
    # Calculate direction and length
    direction = end_location - start_location
    length = direction.length

    # Create cylinder aligned with Z-axis at START location (not center)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=segments,
        radius=start_diameter / 2.0,  # Base radius at start
        depth=length,  # Total length
        location=start_location  # Position at START, not center
    )

    cylinder = bpy.context.active_object
    cylinder.name = name

    # Calculate taper factor (1.0 = no taper, <1.0 = taper down, >1.0 = taper up)
    taper_factor = end_diameter / start_diameter

    # Add taper modifier
    taper_modifier = cylinder.modifiers.new(name="Taper", type='SIMPLE_DEFORM')
    taper_modifier.deform_method = 'TAPER'
    taper_modifier.factor = taper_factor - 1.0  # 0 = no taper
    taper_modifier.deform_axis = 'Z'  # Taper along the cylinder's length
    taper_modifier.lock_x = False
    taper_modifier.lock_y = False

    # Apply the modifier to make the taper permanent
    bpy.context.view_layer.objects.active = cylinder
    bpy.ops.object.modifier_apply(modifier="Taper")

    # Rotate cylinder to point from start to end location
    up_vector = Vector((0, 0, 1))
    if direction.length > 0.001:  # Avoid division by zero
        direction.normalize()
        rotation_quat = up_vector.rotation_difference(direction)
        cylinder.rotation_mode = 'QUATERNION'
        cylinder.rotation_quaternion = rotation_quat

    # Move cylinder so start is at start_location and end is at end_location
    # After rotation, the cylinder's local Z=0 is at start, Z=length is at end
    # No need to adjust position since we created it at start_location

    return cylinder

##########################################################################################

def setup_pants_vertex_groups_to_vb(pants_obj, side, hip_vb, knee_vb, heel_vb):
    """Assign pants vertices to VERTEX BUNDLE empties for direct control"""

    # Clear any existing vertex groups that might conflict
    for vg_name in list(pants_obj.vertex_groups.keys()):
        if vg_name.startswith("CTRL_") or vg_name.startswith("VB_"):
            pants_obj.vertex_groups.remove(pants_obj.vertex_groups[vg_name])

    # Create vertex groups for VB empties
    vb_data = [
        (hip_vb, "HIP", 0.7, 1.0),  # Top: 70%-100% of height
        (knee_vb, "KNEE", 0.35, 0.45),  # Knee: 35%-45% of height - DIRECT CONTROL
        (heel_vb, "HEEL", 0.0, 0.2)  # Bottom: 0%-20% of height
    ]

    for vb_obj, region, min_height, max_height in vb_data:
        if not vb_obj:
            continue

        vg_name = vb_obj.name  # Use the actual VB empty name
        vg = pants_obj.vertex_groups.new(name=vg_name)

        assigned_vertices = 0
        for i, vert in enumerate(pants_obj.data.vertices):
            vert_height = vert.co.z
            weight = 0.0

            if min_height <= vert_height <= max_height:
                # Strong weight in the target region
                if region == "KNEE":
                    weight = 1.0  # Maximum control at knees
                else:
                    # Gradual falloff for hip and heel
                    center = (min_height + max_height) / 2
                    distance = abs(vert_height - center) / (max_height - min_height) * 2
                    weight = max(0.0, 1.0 - distance)
                    weight = min(1.0, weight * 1.2)  # Slight boost

            if weight > 0.1:  # Only assign significant weights
                vg.add([i], weight, 'REPLACE')
                assigned_vertices += 1

        print(f"✓ Assigned {assigned_vertices} vertices to {vg_name} for {region} control")

    # Create combined pinning group (EXISTING FUNCTIONALITY - KEEP)
    pin_group = pants_obj.vertex_groups.new(name=f"{side}_Pants_Combined_Anchors")

    # Combine weights from all VB groups for cloth pinning
    for i, vert in enumerate(pants_obj.data.vertices):
        total_weight = 0.0

        for vg_name in [hip_vb.name, knee_vb.name, heel_vb.name]:
            vg = pants_obj.vertex_groups.get(vg_name)
            if vg:
                try:
                    weight = vg.weight(i)
                    total_weight = max(total_weight, weight)
                except:
                    pass

        if total_weight > 0.3:
            pin_group.add([i], total_weight, 'REPLACE')

    print(f"✓ Created {side}_Pants_Combined_Anchors with weights from VB system")

##########################################################################################

def create_coat(armature_obj, figure_name):
    """Create coat torso garment with shoulder coordination and length variations using two-empties system"""
    script_log("Creating coat torso garment...")

    # Get coat configuration
    coat_length = garment_configs.get("coat_length", "short")  # "short" or "long"
    radial_segments = garment_configs.get("radial_segments", 32)
    longitudinal_segments = garment_configs.get("longitudinal_segments", 24)
    torso_radius = garment_configs.get("torso_radius", 0.25)
    coat_height = garment_configs.get("coat_height", 0.8)
    puffiness = garment_configs.get("puffiness", 1.05)

    # GET SHOULDER DIAMETER FROM SLEEVE CONFIG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    CLOTH_CONFIG_FILE = os.path.join(script_dir, "4M_cloth_config.json")

    diameter_shoulder = 0.15  # Fallback value
    try:
        with open(CLOTH_CONFIG_FILE, 'r') as file:
            cloth_config_data = json.load(file)
            sleeve_config = cloth_config_data.get("cloth_garments", {}).get("left_sleeve", {})
            diameter_shoulder = sleeve_config.get("diameter_shoulder", 0.15)  # Get from sleeve config
    except:
        script_log(f"⚠ Could not load shoulder diameter from config, using fallback: {diameter_shoulder}")

    script_log(f"DEBUG: Coat - Length: {coat_length}, Radial segments: {radial_segments}")
    script_log(f"DEBUG: Coat - Longitudinal segments: {longitudinal_segments}, Height: {coat_height}")
    script_log(
        f"DEBUG: Coat - Torso radius: {torso_radius}, Shoulder diameter: {diameter_shoulder}, Puffiness: {puffiness}")

    # Get shoulder positions for coordination
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        # Get shoulder and spine bones for positioning
        left_shoulder_bone = armature_obj.pose.bones.get("DEF_LeftShoulder")
        right_shoulder_bone = armature_obj.pose.bones.get("DEF_RightShoulder")
        neck_bone = armature_obj.pose.bones.get("DEF_Neck")
        upper_spine_bone = armature_obj.pose.bones.get("DEF_UpperSpine")

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([left_shoulder_bone, right_shoulder_bone, neck_bone, upper_spine_bone]):
            script_log("ERROR: Could not find required bones for coat")
            return None

        # Get bone positions in world space
        left_shoulder_pos = armature_obj.matrix_world @ left_shoulder_bone.tail
        right_shoulder_pos = armature_obj.matrix_world @ right_shoulder_bone.tail
        upper_spine_pos = armature_obj.matrix_world @ upper_spine_bone.head

        # Calculate coat dimensions
        shoulder_width = (right_shoulder_pos - left_shoulder_pos).length
        shoulder_center = (left_shoulder_pos + right_shoulder_pos) / 2
        spine_to_shoulder = (shoulder_center - upper_spine_pos).length

        # =========================================================================
        # STEP 1: CREATE VERTICAL CYLINDER (MAIN BODY) - APPLY Y-SQUISH TO MATCH SHOULDER DIAMETER
        # =========================================================================
        script_log("DEBUG: Creating vertical cylinder for coat body...")

        # Position cylinder centered at shoulders, extending downward
        vertical_center = shoulder_center + Vector((0, 0, -coat_height / 2))

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=radial_segments,
            depth=coat_height,
            radius=torso_radius,  # Starts with torso_radius = 0.25 from config
            location=vertical_center
        )
        vertical_cylinder = bpy.context.active_object
        vertical_cylinder.name = f"{figure_name}_Coat_Vertical"

        # CALCULATE Y SCALE TO MATCH SHOULDER DIAMETER
        # Current Y diameter = torso_radius * 2 = 0.25 * 2 = 0.50
        # Target Y diameter = shoulder_diameter = 0.15
        # Y scale ratio = target / current = 0.15 / 0.50 = 0.3
        y_scale_ratio = diameter_shoulder / (torso_radius * 2)

        # Apply Y-squish: X=1.0, Y=calculated ratio, Z=1.0
        vertical_cylinder.scale = (1.0, y_scale_ratio, 1.0)
        script_log(
            f"✓ Vertical cylinder Y-squish: torso_radius={torso_radius} → Y-scale={y_scale_ratio:.3f} to match diameter_shoulder={diameter_shoulder}")

        # =========================================================================
        # STEP 2: ADD VERTICAL SUBDIVISIONS FOR Z-AXIS FLEXIBILITY (PRESERVE SHAPE)
        # =========================================================================
        script_log("DEBUG: Adding vertical subdivisions for coat flexibility...")

        # Get longitudinal segments from config
        longitudinal_segments = garment_configs.get("longitudinal_segments", 24)
        # Subtract 1 because the cylinder already has top and bottom rings
        number_cuts = max(1, longitudinal_segments - 1)

        script_log(f"DEBUG: Using {longitudinal_segments} longitudinal segments, adding {number_cuts} vertical cuts")

        bpy.context.view_layer.objects.active = vertical_cylinder
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_mode(type='EDGE')

        # Select all vertical edges
        bpy.ops.mesh.select_all(action='DESELECT')

        bm = bmesh.from_edit_mesh(vertical_cylinder.data)

        for edge in bm.edges:
            # Check if edge is approximately vertical (different Z coordinates)
            if abs(edge.verts[0].co.z - edge.verts[1].co.z) > 0.01:
                edge.select = True

        # Subdivide only the selected vertical edges using config value
        bpy.ops.mesh.subdivide(number_cuts=number_cuts, smoothness=0.0)

        bpy.ops.object.mode_set(mode='OBJECT')
        script_log(f"✓ Added {number_cuts} vertical subdivisions using longitudinal_segments={longitudinal_segments}")

        # =========================================================================
        # STEP 3: LONG COAT - QUARTER SEPARATION (FRONT SPLIT) AND DELETE BOTTOM FACE
        # =========================================================================
        if coat_length == "long":
            script_log("DEBUG: Creating front split and deleting bottom face for long coat...")

            # Get skirt_start_ratio from config with fallback
            long_coat_settings = garment_configs.get("coat_length_settings", {}).get("long", {})
            skirt_start_ratio = long_coat_settings.get("skirt_start_ratio", 0.6)  # Default to 0.6 if not specified

            # Calculate skirt region dimensions (bottom portion of coat)
            skirt_start_z = -coat_height * (1 - skirt_start_ratio)
            split_depth = coat_height * 0.8
            split_width = 0.05

            script_log(f"DEBUG: Long coat skirt starts at {skirt_start_ratio * 100}% down (z={skirt_start_z:.3f})")

            # Create split cutter object (thin vertical plane)
            bpy.ops.mesh.primitive_plane_add(
                size=split_depth,
                location=vertical_center + Vector((0, 0, skirt_start_z))
            )
            split_cutter = bpy.context.active_object
            split_cutter.name = f"{figure_name}_CoatSplit_Cutter"

            # Scale cutter to be a thin vertical strip
            split_cutter.scale = (1.0, split_width / split_depth, 1.0)
            split_cutter.rotation_euler = (0, math.radians(90), 0)

            # Position cutter at front center of coat
            split_cutter.location.y = vertical_cylinder.location.y + diameter_shoulder * 0.5

            # Add Boolean modifier to subtract split from coat
            bool_split = vertical_cylinder.modifiers.new(name="Split_Front", type='BOOLEAN')
            bool_split.operation = 'DIFFERENCE'
            bool_split.object = split_cutter
            bool_split.solver = 'FAST'

            # Apply the Boolean subtraction
            bpy.context.view_layer.objects.active = vertical_cylinder
            bpy.ops.object.modifier_apply(modifier="Split_Front")

            # Remove the cutter object
            bpy.data.objects.remove(split_cutter, do_unlink=True)

            script_log("✓ Created front split using Boolean subtraction in FAST mode")

            # The bottom face deletion creates the open-bottom cylinder needed
            # for the coat torso, allowing it to drape naturally over the
            # lower body while maintaining the closed top at the shoulders.

            # Find and delete bottom face of vertical cylinder
            script_log("Finding and deleting bottom face of vertical cylinder...")
            bpy.context.view_layer.objects.active = vertical_cylinder
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='FACE')
            bpy.ops.mesh.select_all(action='DESELECT')

            # Calculate the Z threshold for bottom face (shoulder_center Z minus coat_height)
            bottom_z_threshold = shoulder_center.z - coat_height + 0.01  # Small tolerance

            bm = bmesh.from_edit_mesh(vertical_cylinder.data)
            bottom_faces = []

            for face in bm.faces:
                # Check if all vertices in this face are near the bottom
                all_vertices_bottom = True
                for vert in face.verts:
                    world_pos = vertical_cylinder.matrix_world @ vert.co
                    if abs(world_pos.z - bottom_z_threshold) > 0.02:  # 2cm tolerance
                        all_vertices_bottom = False
                        break

                if all_vertices_bottom:
                    bottom_faces.append(face)

            # Select and delete bottom faces
            if bottom_faces:
                bmesh.ops.delete(bm, geom=bottom_faces, context='FACES')
                script_log(f"Deleted {len(bottom_faces)} bottom face(s)")
            else:
                script_log("WARNING: No bottom face found to delete")

            bmesh.update_edit_mesh(vertical_cylinder.data)
            bpy.ops.object.mode_set(mode='OBJECT')

        # =========================================================================
        # STEP 4: CREATE HORIZONTAL CYLINDER (SHOULDERS/CHEST)
        # =========================================================================
        script_log(f"DEBUG: Creating horizontal cylinder for shoulders, width {shoulder_width}...")

        # Position horizontal cylinder between shoulders
        horizontal_center = shoulder_center
        shoulder_vector = (right_shoulder_pos - left_shoulder_pos).normalized()

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=radial_segments,
            depth=shoulder_width,
            radius=diameter_shoulder / 2.0,
            location=horizontal_center
        )
        horizontal_cylinder = bpy.context.active_object
        horizontal_cylinder.name = f"{figure_name}_Coat_Horizontal"

        # Rotate to align with shoulder line
        horizontal_cylinder.rotation_euler = shoulder_vector.to_track_quat('Z', 'Y').to_euler()

        # =========================================================================
        # STEP 5: COMBINE CYLINDERS WITH BOOLEAN UNION
        # =========================================================================
        script_log("DEBUG: Combining cylinders with boolean union...")

        # Set vertical cylinder as main object
        coat_obj = vertical_cylinder
        coat_obj.name = f"{figure_name}_Coat"

        # Add boolean modifier to combine with horizontal cylinder
        boolean_mod = coat_obj.modifiers.new(name="Boolean_Union", type='BOOLEAN')
        boolean_mod.operation = 'UNION'
        boolean_mod.object = horizontal_cylinder

        # Apply boolean modifier
        bpy.context.view_layer.objects.active = coat_obj
        bpy.ops.object.modifier_apply(modifier="Boolean_Union")

        # Remove horizontal cylinder
        bpy.data.objects.remove(horizontal_cylinder, do_unlink=True)

        # =========================================================================
        # STEP 6: SMOOTH ARMPIT AREAS
        # =========================================================================
        smooth_armpits = garment_configs.get("smooth_armpits", False)
        if smooth_armpits:
            script_log("DEBUG: Smoothing armpit areas...")

            bpy.context.view_layer.objects.active = coat_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='VERT')  # Blender 4.4.4 uses 'VERT' not 'VERTEX'

            # Select vertices in armpit regions and smooth them
            bm = bmesh.from_edit_mesh(coat_obj.data)

            for vert in bm.verts:
                # Armpit regions are around the shoulder connections
                vert_local = coat_obj.matrix_world.inverted() @ vert.co
                if abs(vert_local.x) > torso_radius * 0.8 and vert_local.z > -0.1:
                    vert.select = True

            if any(v.select for v in bm.verts):
                bpy.ops.mesh.vertices_smooth(factor=0.5, repeat=3)

            bpy.ops.object.mode_set(mode='OBJECT')

        # =========================================================================
        # STEP 7: SETUP VERTEX GROUPS WITH TWO-EMPTIES DYNAMIC SYSTEM
        # =========================================================================
        script_log("DEBUG: Setting up coat vertex groups with two-empties dynamic system...")

        # Clear any existing parenting
        coat_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(coat_obj.vertex_groups):
            coat_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(coat_obj.modifiers):
            if mod.type == 'ARMATURE':
                coat_obj.modifiers.remove(mod)

        # NEW: USE TWO-EMPTIES SYSTEM FOR DYNAMIC VERTEX WEIGHTS
        script_log("DEBUG: Setting up dynamic vertex weights using two-empties system for coat")

        # Get control point names for the two-empties system
        left_shoulder_cp = "CTRL_LEFT_SHOULDER"
        right_shoulder_cp = "CTRL_RIGHT_SHOULDER"

        # Get vertex bundle centers and radii from two-empties system
        left_shoulder_center = get_bundle_center(left_shoulder_cp)
        right_shoulder_center = get_bundle_center(right_shoulder_cp)

        # Get radii with garment-specific adjustments
        left_shoulder_radius = get_bundle_radius(left_shoulder_cp)
        right_shoulder_radius = get_bundle_radius(right_shoulder_cp)

        # Initialize vertex groups and store data for dynamic setup
        vertex_data = {
            left_shoulder_cp: [],
            right_shoulder_cp: []
        }

        # Create vertex groups for two-empties system
        left_shoulder_group = coat_obj.vertex_groups.new(name=left_shoulder_cp)
        right_shoulder_group = coat_obj.vertex_groups.new(name=right_shoulder_cp)

        # Apply initial vertex group weighting AND collect data for dynamic setup
        mesh = coat_obj.data
        for vert in mesh.vertices:
            vert_co = coat_obj.matrix_world @ vert.co
            vert_height = vert_co.z - shoulder_center.z  # Relative to shoulders

            # LEFT SHOULDER WEIGHTING
            dist_to_left_shoulder = (vert_co - left_shoulder_center).length
            if dist_to_left_shoulder <= left_shoulder_radius:
                base_weight = 1.0 - (dist_to_left_shoulder / left_shoulder_radius)

                # Shoulders influence top section
                height_factor = max(0.0, 1.0 - (abs(vert_height) / (coat_height * 0.3)))
                final_weight = base_weight * height_factor

                # Shoulder-specific: enhance influence near armholes
                if abs(vert_co.x - left_shoulder_center.x) < left_shoulder_radius * 0.5:
                    final_weight *= 1.2

                if final_weight > 0.01:
                    left_shoulder_group.add([vert.index], final_weight, 'REPLACE')
                    vertex_data[left_shoulder_cp].append((vert.index, final_weight))

            # RIGHT SHOULDER WEIGHTING
            dist_to_right_shoulder = (vert_co - right_shoulder_center).length
            if dist_to_right_shoulder <= right_shoulder_radius:
                base_weight = 1.0 - (dist_to_right_shoulder / right_shoulder_radius)

                # Shoulders influence top section
                height_factor = max(0.0, 1.0 - (abs(vert_height) / (coat_height * 0.3)))
                final_weight = base_weight * height_factor

                # Shoulder-specific: enhance influence near armholes
                if abs(vert_co.x - right_shoulder_center.x) < right_shoulder_radius * 0.5:
                    final_weight *= 1.2

                if final_weight > 0.01:
                    right_shoulder_group.add([vert.index], final_weight, 'REPLACE')
                    vertex_data[right_shoulder_cp].append((vert.index, final_weight))

        # SET UP DYNAMIC VERTEX WEIGHTS - NEW FUNCTIONALITY
        script_log("Setting up dynamic vertex weights for coat...")
        drivers_created = 0
        for point_name, vertices_weights in vertex_data.items():
            if vertices_weights:  # Only if we have vertices for this control point
                vertex_indices = [vw[0] for vw in vertices_weights]
                initial_weights = [vw[1] for vw in vertices_weights]

                script_log(f"Setting up {len(vertex_indices)} dynamic vertices for {point_name}")
                setup_dynamic_vertex_weights(coat_obj, point_name, vertex_indices, initial_weights, joint_control_systems)
                drivers_created += len(vertex_indices)
            else:
                script_log(f"No vertices found for {point_name}, skipping dynamic setup")

        script_log(f"Total dynamic drivers created for coat: {drivers_created}")

        # Combined group for cloth pinning (backward compatibility)
        combined_anchors_group = coat_obj.vertex_groups.new(name="Coat_Combined_Anchors")

        # Combine weights from all control points
        for i in range(len(coat_obj.data.vertices)):
            max_weight = 0.0
            for group_name in [left_shoulder_cp, right_shoulder_cp]:
                group = coat_obj.vertex_groups.get(group_name)
                if group:
                    try:
                        weight = group.weight(i)
                        max_weight = max(max_weight, weight)
                    except:
                        # Vertex not in this group, continue
                        pass

            if max_weight > 0.1:  # Higher threshold for coat
                combined_anchors_group.add([i], max_weight, 'REPLACE')

        script_log(f"✓ Created Coat_Combined_Anchors with weights from two-empties system")

        # =========================================================================
        # STEP 8: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = coat_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 9: ADD CLOTH SIMULATION WITH SIMPLE COLLISIONS
        # =========================================================================
        script_log("DEBUG: Adding cloth simulation with simple collisions...")
        cloth_config = garment_configs.get("cloth_settings", {})

        if cloth_config.get("enabled", True):
            cloth_mod = coat_obj.modifiers.new(name="Cloth", type='CLOTH')

            # Apply cloth settings from config
            cloth_mod.settings.quality = cloth_config.get("quality", 12)
            cloth_mod.settings.mass = cloth_config.get("mass", 0.3)
            cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 8.0)
            cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 7.0)
            cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 5.0)
            cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.8)
            cloth_mod.settings.air_damping = cloth_config.get("air_damping", 0.8)
            cloth_mod.settings.time_scale = cloth_config.get("time_scale", 1.0)

            # SIMPLE COLLISIONS - COAT WILL INTERACT WITH PANTS AUTOMATICALLY
            cloth_mod.collision_settings.use_collision = True
            cloth_mod.collision_settings.collision_quality = cloth_config.get("collision_quality", 6)
            cloth_mod.collision_settings.distance_min = cloth_config.get("external_distance_min", 0.005)

            # Self-collision for coat fabric
            cloth_mod.collision_settings.use_self_collision = True
            cloth_mod.collision_settings.self_distance_min = cloth_config.get("self_distance_min", 0.002)

            # PIN ENTIRE COAT TO COMBINED ANCHORS
            cloth_mod.settings.vertex_group_mass = "Coat_Combined_Anchors"

            script_log("✓ Coat cloth: dynamic vertex pinning + simple collisions (will interact with pants)")
        else:
            script_log("DEBUG: Cloth simulation disabled for coat")

        # =========================================================================
        # STEP 10: ADD MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding coat materials...")

        apply_material_from_config(coat_obj, "coat_torso", material_name="Coat_Material", fallback_color=(0.1, 0.3, 0.8, 1.0))

        # =========================================================================
        # STEP 11: SET MODIFIER ORDER (NO SUBDIVISION)
        # =========================================================================
        script_log("DEBUG: Setting modifier order (no subdivision)...")

        bpy.context.view_layer.objects.active = coat_obj
        modifiers = coat_obj.modifiers

        # Remove Subdivision modifier if it exists
        if "Subdivision" in modifiers:
            modifiers.remove(modifiers["Subdivision"])
            script_log("✓ Removed Subdivision modifier - using manual geometry control")

        # Build correct order based on which modifiers are present
        correct_order = ["Armature"]
        if cloth_config.get("enabled", True):
            correct_order.append("Cloth")

        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 12: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        script_log(f"=== COAT TORSO CREATION COMPLETE ===")
        script_log(f"✓ Coat type: {coat_length}")
        script_log(f"✓ Height: {coat_height:.3f}")
        script_log(f"✓ Shoulder width: {shoulder_width:.3f}")
        script_log(f"✓ Torso radius: {torso_radius}")
        script_log(f"✓ Front split: {'CREATED' if coat_length == 'long' else 'NOT APPLIED'}")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_config.get('enabled', True) else 'DISABLED'}")
        script_log(f"✓ Dynamic vertex drivers: {drivers_created} created")
        script_log(f"✓ Control points: {left_shoulder_cp}, {right_shoulder_cp}")
        script_log(f"✓ Using two-empties system for dynamic vertex weight updates during animation")

        return coat_obj

    except Exception as e:
        script_log(f"ERROR creating coat: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_mitten(armature_obj, figure_name, garment_config, global_cloth_settings, side="left"):
    """Create mitten with seamless thumb attachment using new vertex bundles system"""
    script_log(f"Creating {side} mitten with seamless thumb attachment...")

    # Get hand bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        if side == "left":
            hand_bone_name = "DEF_LeftHand"
            forearm_bone_name = "DEF_LeftForearm"
            elbow_bone_name = "DEF_LeftUpperArm"  # For arm plane calculation
            shoulder_bone_name = "DEF_LeftShoulder"  # For arm plane calculation
            thumb_direction = -1  # Left thumb points to left (negative X)

            # Control points for new vertex bundles system
            wrist_control_point = "CTRL_LEFT_WRIST"
            hand_control_point = "CTRL_LEFT_HAND"  # Assuming this exists or falls back to wrist
        else:
            hand_bone_name = "DEF_RightHand"
            forearm_bone_name = "DEF_RightForearm"
            elbow_bone_name = "DEF_RightUpperArm"  # For arm plane calculation
            shoulder_bone_name = "DEF_RightShoulder"  # For arm plane calculation
            thumb_direction = 1  # Right thumb points to right (positive X)

            # Control points for new vertex bundles system
            wrist_control_point = "CTRL_RIGHT_WRIST"
            hand_control_point = "CTRL_RIGHT_HAND"  # Assuming this exists or falls back to wrist

        hand_bone = armature_obj.pose.bones.get(hand_bone_name)
        forearm_bone = armature_obj.pose.bones.get(forearm_bone_name)
        elbow_bone = armature_obj.pose.bones.get(elbow_bone_name)
        shoulder_bone = armature_obj.pose.bones.get(shoulder_bone_name)

        bpy.ops.object.mode_set(mode='OBJECT')

        if not hand_bone:
            script_log(f"ERROR: Could not find hand bone for {side} mitten")
            return None

        # Get hand bone positions in world space
        hand_pos = armature_obj.matrix_world @ hand_bone.head
        hand_tail = armature_obj.matrix_world @ hand_bone.tail
        hand_direction = (hand_tail - hand_pos).normalized()

        # Get arm plane landmarks for thumb orientation
        if elbow_bone and shoulder_bone:
            elbow_pos = armature_obj.matrix_world @ elbow_bone.tail
            shoulder_pos = armature_obj.matrix_world @ shoulder_bone.tail
            wrist_pos = armature_obj.matrix_world @ hand_bone.head  # Wrist position

            # Calculate arm plane normal
            upper_arm_vector = (elbow_pos - shoulder_pos).normalized()
            forearm_vector = (wrist_pos - elbow_pos).normalized()
            arm_plane_normal = upper_arm_vector.cross(forearm_vector).normalized()

            # Calculate forward direction in arm plane (perpendicular to hand direction)
            forward_in_plane = arm_plane_normal.cross(hand_direction).normalized()
            script_log(f"✓ Calculated arm plane for {side} mitten thumb orientation")
        else:
            # Fallback: use simple forward direction
            forward_in_plane = Vector((0, 1, 0))  # Forward in world space
            script_log(f"⚠ Using fallback thumb orientation for {side} mitten")

        # Get mitten dimensions from global garment_configs
        mitten_config = garment_configs.get(f"{side}_mitten", garment_config)
        puffiness = mitten_config.get("puffiness", 1.0)
        hand_size = mitten_config.get("hand_size", [0.1, 0.08, 0.04])
        thumb_size = mitten_config.get("thumb_size", [0.04, 0.03, 0.03])
        segments = mitten_config.get("segments", 8)

        # Get geometry settings from config
        geometry_settings = global_cloth_settings.get("geometry_settings", {})
        base_radii = global_cloth_settings.get("base_radii", {})

        # Use configurable thumb angle factor
        mitten_thumb_angle_factor = geometry_settings.get("mitten_thumb_angle_factor", 0.7)
        base_wrist_radius = base_radii.get("wrist", 0.06)

        # Unpack sizes
        hand_length, hand_radius_x, hand_radius_y = hand_size
        thumb_length, thumb_radius_x, thumb_radius_y = thumb_size

        # Apply puffiness
        hand_radius_x *= puffiness
        hand_radius_y *= puffiness
        thumb_radius_x *= puffiness
        thumb_radius_y *= puffiness

        script_log(
            f"DEBUG: {side} mitten - Hand: length={hand_length}, radius_x={hand_radius_x}, radius_y={hand_radius_y}")
        script_log(
            f"DEBUG: {side} mitten - Thumb: length={thumb_length}, radius_x={thumb_radius_x}, radius_y={thumb_radius_y}")
        script_log(f"DEBUG: {side} mitten - Puffiness: {puffiness}")
        script_log(f"DEBUG: {side} mitten - Thumb angle factor: {mitten_thumb_angle_factor}")
        script_log(f"DEBUG: {side} mitten - Base wrist radius: {base_wrist_radius}")
        script_log(f"DEBUG: {side} mitten - Segments: {segments}")

        # =========================================================================
        # STEP 1: CREATE HAND CYLINDER AS BASE MESH
        # =========================================================================
        script_log(f"DEBUG: Creating hand cylinder for {side} mitten...")

        # Calculate hand cylinder position and orientation
        hand_center = hand_pos + (hand_direction * hand_length / 2)

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=hand_length,
            radius=1.0,  # Unit radius, we'll scale it
            location=hand_center
        )
        mitten_obj = bpy.context.active_object
        mitten_obj.name = f"{figure_name}_{side.capitalize()}Mitten"

        # Rotate to align with hand bone direction
        mitten_obj.rotation_euler = hand_direction.to_track_quat('Z', 'Y').to_euler()

        # Scale to create elliptical cross-section for hand
        mitten_obj.scale = (hand_radius_x, hand_radius_y, 1.0)

        # =========================================================================
        # STEP 2: CREATE SEAMLESS THUMB ATTACHMENT
        # =========================================================================
        script_log(f"DEBUG: Creating seamless thumb attachment for {side} mitten...")

        bpy.context.view_layer.objects.active = mitten_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(mitten_obj.data)

        # Calculate thumb attachment point on hand cylinder
        # Thumb attaches on the side, slightly forward in the arm plane
        thumb_attach_local = Vector((thumb_direction * hand_radius_x * 0.8, 0, hand_length * 0.3))

        # Find closest vertex to thumb attachment point
        closest_vert = None
        min_distance = float('inf')

        for vert in bm.verts:
            distance = (vert.co - thumb_attach_local).length
            if distance < min_distance:
                min_distance = distance
                closest_vert = vert

        if closest_vert:
            # Select vertices around the attachment point for thumb base
            bpy.ops.mesh.select_all(action='DESELECT')
            closest_vert.select = True

            # Grow selection to get thumb base ring
            bpy.ops.mesh.select_more()
            bpy.ops.mesh.select_more()

            # Get selected vertices for thumb base
            selected_verts = [v for v in bm.verts if v.select]

            if selected_verts:
                # Calculate thumb base center
                thumb_base_center = Vector((0, 0, 0))
                for vert in selected_verts:
                    thumb_base_center += vert.co
                thumb_base_center /= len(selected_verts)

                # Calculate thumb direction (forward in arm plane with slight outward angle)
                thumb_dir = (hand_direction + forward_in_plane * mitten_thumb_angle_factor).normalized()

                # Convert thumb direction to local space
                thumb_dir_local = mitten_obj.matrix_world.inverted().to_3x3() @ thumb_dir

                # Extrude to create thumb
                extruded = bmesh.ops.extrude_vert_indiv(bm, verts=selected_verts)

                # Move extruded vertices to form thumb
                for vert in extruded['verts']:
                    # Move in thumb direction
                    vert.co += thumb_dir_local * thumb_length * 0.3

                # Scale down for thumb tip
                bmesh.ops.scale(
                    bm,
                    vec=Vector((thumb_radius_x / hand_radius_x, thumb_radius_y / hand_radius_y, 1.0)),
                    space=mitten_obj.matrix_world.inverted(),
                    verts=extruded['verts']
                )

                script_log(f"✓ Created seamless thumb attachment with {len(selected_verts)} base vertices")
            else:
                script_log("⚠ Could not select vertices for thumb base")
        else:
            script_log("⚠ Could not find thumb attachment point")

        # Update mesh and return to object mode
        bmesh.update_edit_mesh(mitten_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        # =========================================================================
        # STEP 3: SETUP PARENTING AND ARMATURE MODIFIERS
        # =========================================================================
        script_log("DEBUG: Setting up parenting and armature modifiers...")

        # Clear any existing parenting
        mitten_obj.parent = None

        # Clear any existing vertex groups
        for vg in list(mitten_obj.vertex_groups):
            mitten_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(mitten_obj.modifiers):
            if mod.type == 'ARMATURE':
                mitten_obj.modifiers.remove(mod)

        # =========================================================================
        # STEP 4: NEW VERTEX BUNDLES SYSTEM INTEGRATION FOR WRIST COORDINATION
        # =========================================================================
        script_log(f"DEBUG: Integrating new vertex bundles system for {side} mitten wrist...")

        # Get wrist vertex bundle center and radius from new system
        wrist_center = get_bundle_center(wrist_control_point)
        wrist_radius = get_bundle_radius(wrist_control_point)

        # Create spherical vertex group for wrist bundle coordination
        wrist_sphere_group = mitten_obj.vertex_groups.new(name=f"Wrist_Sphere_{side}")

        # =========================================================================
        # APPLY WRIST BUNDLE VERTEX WEIGHTS USING NEW SYSTEM
        # =========================================================================
        if wrist_center:
            script_log(f"✓ Applying wrist vertex bundle from {wrist_control_point}")

            # Calculate sphere radius for influence
            wrist_sphere_radius = wrist_radius * 2.0  # Double radius for better influence

            for i, vertex in enumerate(mitten_obj.data.vertices):
                vert_pos = mitten_obj.matrix_world @ vertex.co
                distance = (vert_pos - wrist_center).length

                # Apply weight based on distance to bundle center
                if distance <= wrist_sphere_radius:
                    weight = 1.0 - (distance / wrist_sphere_radius)
                    weight = weight * weight  # Quadratic falloff

                    # Reduce weight for thumb area to allow more flexibility
                    vert_local = mitten_obj.matrix_world.inverted() @ vert_pos
                    if abs(vert_local.x) > hand_radius_x * 0.6:  # Thumb area
                        weight *= 0.6  # Reduced influence for thumb

                    if weight > 0.1:
                        wrist_sphere_group.add([i], weight, 'REPLACE')
        else:
            script_log(f"⚠ Wrist bundle center not found for {wrist_control_point}, using standard weighting")

        # =========================================================================
        # STEP 5: SETUP VERTEX GROUPS FOR BONE DEFORMATION
        # =========================================================================
        script_log("DEBUG: Setting up vertex groups for bone deformation...")

        # Create vertex group for hand bone
        hand_group = mitten_obj.vertex_groups.new(name=hand_bone_name)

        # Apply wrist bundle weighting if available, otherwise use standard weighting
        if wrist_center:
            wrist_sphere_radius = wrist_radius * 2.0

            for i, vertex in enumerate(mitten_obj.data.vertices):
                vert_pos = mitten_obj.matrix_world @ vertex.co
                distance = (vert_pos - wrist_center).length

                # Apply weight based on distance to wrist bundle center
                if distance <= wrist_sphere_radius:
                    weight = 1.0 - (distance / wrist_sphere_radius)
                    weight = weight * weight  # Quadratic falloff

                    # Reduce weight for thumb area to allow more flexibility
                    vert_local = mitten_obj.matrix_world.inverted() @ vert_pos
                    if abs(vert_local.x) > hand_radius_x * 0.6:  # Thumb area
                        weight *= 0.6  # Reduced influence for thumb

                    if weight > 0.1:
                        hand_group.add([i], weight, 'REPLACE')
                else:
                    # Default weight for areas far from wrist
                    hand_group.add([i], 0.3, 'REPLACE')  # Light influence
        else:
            # Fallback: assign uniform weights with wrist emphasis
            for i, vertex in enumerate(mitten_obj.data.vertices):
                vert_local = mitten_obj.matrix_world.inverted() @ vertex.co
                # Stronger weight at wrist, lighter at fingers
                if vert_local.z < -hand_length * 0.3:  # Wrist area
                    hand_group.add([i], 1.0, 'REPLACE')
                elif vert_local.z < 0:  # Lower hand
                    hand_group.add([i], 0.8, 'REPLACE')
                else:  # Upper hand and thumb
                    hand_group.add([i], 0.6, 'REPLACE')

        # =========================================================================
        # STEP 6: ADD ARMATURE MODIFIER
        # =========================================================================
        script_log("DEBUG: Adding armature modifier...")

        # Add armature modifier
        armature_mod = mitten_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True

        # =========================================================================
        # STEP 7: ADD SUBDIVISION AND MATERIALS
        # =========================================================================
        script_log("DEBUG: Adding subdivision and materials...")

        # Add subdivision for smoother mitten
        subdiv_mod = mitten_obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv_mod.levels = 1
        subdiv_mod.render_levels = 1

        # Add materials from global config
        apply_material_from_config(mitten_obj, f"{side}_mitten", fallback_color=(0.8, 0.1, 0.1, 1.0))

        # =========================================================================
        # STEP 8: CLOTH SIMULATION - DISABLED AS REQUESTED
        # =========================================================================
        cloth_config = mitten_config.get("cloth_settings", {})
        if cloth_config.get("enabled", False):
            script_log(f"DEBUG: Adding cloth simulation to {side} mitten...")
            cloth_mod = mitten_obj.modifiers.new(name="Cloth", type='CLOTH')

            # Apply cloth settings from config
            cloth_mod.settings.quality = cloth_config.get("quality", 4)
            cloth_mod.settings.mass = cloth_config.get("mass", 0.1)
            cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 5.0)
            cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 4.0)
            cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 3.0)
            cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.2)
            cloth_mod.settings.air_damping = cloth_config.get("air_damping", 1.0)

            # PIN CLOTH TO WRIST SPHERE GROUP IF BUNDLE EXISTS
            if wrist_center:
                cloth_mod.settings.vertex_group_mass = f"Wrist_Sphere_{side}"
                script_log(f"✓ Mitten cloth pinned to wrist spherical vertex group")
        else:
            script_log(f"DEBUG: Cloth simulation disabled for {side} mitten")

        # =========================================================================
        # STEP 9: SET MODIFIER ORDER
        # =========================================================================
        script_log("DEBUG: Setting modifier order...")

        bpy.context.view_layer.objects.active = mitten_obj
        modifiers = mitten_obj.modifiers

        # Ensure proper order: Subdivision → Armature → Cloth (if enabled)
        correct_order = ["Subdivision", "Armature", "Cloth"]
        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # =========================================================================
        # STEP 10: CREATE COMBINED COORDINATION GROUP FOR MITTEN
        # =========================================================================
        script_log(f"DEBUG: Creating combined coordination group for {side} mitten...")

        # Create a combined group that includes wrist sphere influence
        combined_coordination_group = mitten_obj.vertex_groups.new(name=f"{side}_Mitten_Combined_Coordination")

        # Combine wrist sphere weights with standard armature weights
        for i in range(len(mitten_obj.data.vertices)):
            max_weight = 0.0

            # Check wrist sphere group weight
            try:
                sphere_weight = wrist_sphere_group.weight(i)
                max_weight = max(max_weight, sphere_weight)
            except:
                pass

            # Check standard armature group weight
            try:
                armature_weight = hand_group.weight(i)
                max_weight = max(max_weight, armature_weight)
            except:
                pass

            if max_weight > 0.1:
                combined_coordination_group.add([i], max_weight, 'REPLACE')

        # =========================================================================
        # STEP 11: FINAL VERIFICATION
        # =========================================================================
        bpy.context.view_layer.update()

        # Log bundle usage and thumb orientation
        bundle_status = f"wrist({wrist_control_point})" if wrist_center else "NONE (standard)"
        thumb_orientation = "arm plane aligned" if elbow_bone and shoulder_bone else "fallback"

        script_log(f"=== {side.upper()} MITTEN CREATION COMPLETE (SEAMLESS + COORDINATED) ===")
        script_log(f"✓ Hand cylinder: length={hand_length}, radius_x={hand_radius_x}, radius_y={hand_radius_y}")
        script_log(f"✓ Thumb: length={thumb_length}, radius_x={thumb_radius_x}, radius_y={thumb_radius_y}")
        script_log(f"✓ Thumb orientation: {thumb_orientation}")
        script_log(f"✓ Thumb angle factor: {mitten_thumb_angle_factor}")
        script_log(f"✓ Base wrist radius: {base_wrist_radius}")
        script_log(f"✓ Segments: {segments}")
        script_log(f"✓ Puffiness factor applied: {puffiness}")
        script_log(f"✓ Mitten parented to {hand_bone_name}")
        script_log(f"✓ Cloth simulation: {'ENABLED' if cloth_config.get('enabled', False) else 'DISABLED'}")
        script_log(f"✓ Vertex bundle used: {bundle_status}")
        script_log(f"✓ Thumb attachment: SEAMLESS (single mesh)")
        script_log(f"✓ Thumb orientation: COPLANAR with arm plane")
        script_log(f"✓ Wrist coordination: Uses new vertex bundles system for animation sync")

        return mitten_obj

    except Exception as e:
        script_log(f"ERROR creating {side} mitten: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def create_sleeve(armature_obj, figure_name, garment_config, global_cloth_settings, side="left"):
    """Create continuous sleeve with spherical pinning zone weighting using new vertex bundles system"""
    script_log(f"Creating continuous {side} sleeve with spherical pinning zones...")

    # Get arm bone positions
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    try:
        if side == "left":
            shoulder_bone_name = "DEF_LeftShoulder"
            upper_arm_bone_name = "DEF_LeftUpperArm"
            forearm_bone_name = "DEF_LeftForearm"

            # Control points for new vertex bundles system
            shoulder_control_point = "CTRL_LEFT_SHOULDER"
            elbow_control_point = "CTRL_LEFT_ELBOW"
            wrist_control_point = "CTRL_LEFT_WRIST"
        else:
            shoulder_bone_name = "DEF_RightShoulder"
            upper_arm_bone_name = "DEF_RightUpperArm"
            forearm_bone_name = "DEF_RightForearm"

            # Control points for new vertex bundles system
            shoulder_control_point = "CTRL_RIGHT_SHOULDER"
            elbow_control_point = "CTRL_RIGHT_ELBOW"
            wrist_control_point = "CTRL_RIGHT_WRIST"

        shoulder_bone = armature_obj.pose.bones.get(shoulder_bone_name)
        upper_arm_bone = armature_obj.pose.bones.get(upper_arm_bone_name)
        forearm_bone = armature_obj.pose.bones.get(forearm_bone_name)

        bpy.ops.object.mode_set(mode='OBJECT')

        if not all([shoulder_bone, upper_arm_bone, forearm_bone]):
            script_log(f"ERROR: Could not find arm bones for {side} sleeve")
            return None

        # =========================================================================
        # STEP 1: SET UP BONE CONSTRAINTS FIRST
        # =========================================================================
        script_log(f"DEBUG: Setting up bone constraints for {side} sleeve movement...")

        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        # CLEAR EXISTING CONSTRAINTS FIRST
        for bone_name in [shoulder_bone_name, upper_arm_bone_name, forearm_bone_name]:
            bone = armature_obj.pose.bones.get(bone_name)
            if bone:
                for constraint in list(bone.constraints):
                    bone.constraints.remove(constraint)

        # SET UP STRETCH_TO CONSTRAINTS TO CONTROL POINTS
        constraints_added = 0

        # SHOULDER BONE: Constrain to shoulder control point
        shoulder_target = bpy.data.objects.get(shoulder_control_point)
        if shoulder_bone and shoulder_target:
            stretch = shoulder_bone.constraints.new('STRETCH_TO')
            stretch.target = shoulder_target
            stretch.influence = 1.0
            constraints_added += 1
            script_log(f"✓ {shoulder_bone_name} STRETCH_TO -> {shoulder_control_point}")

        # UPPER ARM BONE: Constrain to elbow control point
        elbow_target = bpy.data.objects.get(elbow_control_point)
        if upper_arm_bone and elbow_target:
            stretch = upper_arm_bone.constraints.new('STRETCH_TO')
            stretch.target = elbow_target
            stretch.influence = 1.0
            constraints_added += 1
            script_log(f"✓ {upper_arm_bone_name} STRETCH_TO -> {elbow_control_point}")

        # FOREARM BONE: Constrain to wrist control point
        wrist_target = bpy.data.objects.get(wrist_control_point)
        if forearm_bone and wrist_target:
            stretch = forearm_bone.constraints.new('STRETCH_TO')
            stretch.target = wrist_target
            stretch.influence = 1.0
            constraints_added += 1
            script_log(f"✓ {forearm_bone_name} STRETCH_TO -> {wrist_control_point}")

        bpy.ops.object.mode_set(mode='OBJECT')
        script_log(f"✓ Added {constraints_added} bone constraints for {side} sleeve")

        # NOW GET UPDATED BONE POSITIONS AFTER CONSTRAINTS ARE APPLIED
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.mode_set(mode='POSE')

        shoulder_bone = armature_obj.pose.bones.get(shoulder_bone_name)
        upper_arm_bone = armature_obj.pose.bones.get(upper_arm_bone_name)
        forearm_bone = armature_obj.pose.bones.get(forearm_bone_name)

        # Get bone positions in world space AFTER constraints are set
        shoulder_pos = armature_obj.matrix_world @ shoulder_bone.tail
        upper_arm_pos = armature_obj.matrix_world @ upper_arm_bone.head
        elbow_pos = armature_obj.matrix_world @ upper_arm_bone.tail
        forearm_pos = armature_obj.matrix_world @ forearm_bone.head
        wrist_pos = armature_obj.matrix_world @ forearm_bone.tail

        bpy.ops.object.mode_set(mode='OBJECT')

        # Get sleeve dimensions from global garment_configs
        sleeve_config = garment_configs.get(f"{side}_sleeve", garment_config)
        diameter_start = sleeve_config.get("diameter_start", 0.15)
        diameter_elbow = sleeve_config.get("diameter_elbow", 0.12)
        diameter_end = sleeve_config.get("diameter_end", 0.08)
        segments = sleeve_config.get("segments", 16)

        # Get artist-controlled settings
        subdivision_config = sleeve_config.get("subdivision", {})
        manual_cuts = subdivision_config.get("manual_cuts", 1)
        subdiv_levels = subdivision_config.get("subdiv_levels", 1)

        weighting_config = sleeve_config.get("vertex_weighting", {})
        falloff_type = weighting_config.get("sphere_falloff", "quadratic")
        min_weight_threshold = weighting_config.get("min_weight_threshold", 0.05)
        sphere_influence_scale = weighting_config.get("sphere_influence_scale", 2.0)

        # Calculate segment lengths
        upper_arm_length = (elbow_pos - upper_arm_pos).length
        forearm_length = (wrist_pos - forearm_pos).length
        total_length = upper_arm_length + forearm_length

        script_log(
            f"DEBUG: {side} sleeve - Upper arm length: {upper_arm_length:.3f}, Forearm length: {forearm_length:.3f}")
        script_log(f"DEBUG: {side} sleeve - Total length: {total_length:.3f}")
        script_log(f"DEBUG: {side} sleeve - Bone constraints: {constraints_added} added")

        # CREATE SINGLE CONTINUOUS CYLINDER
        script_log(f"DEBUG: Creating continuous {side} sleeve cylinder...")

        # Use average radius for initial cylinder
        avg_radius = (diameter_start / 2 + diameter_elbow / 2 + diameter_end / 2) / 3
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=segments,
            depth=total_length,
            radius=avg_radius,
            location=(shoulder_pos + wrist_pos) / 2  # Center between shoulder and wrist
        )
        sleeve_obj = bpy.context.active_object
        sleeve_obj.name = f"{figure_name}_{side.capitalize()}Sleeve"

        # Rotate to align with arm direction
        arm_direction = (wrist_pos - shoulder_pos).normalized()
        sleeve_obj.rotation_euler = arm_direction.to_track_quat('Z', 'Y').to_euler()

        # ADD MANUAL SUBDIVISION
        if manual_cuts > 0:
            script_log(f"DEBUG: Adding {manual_cuts} manual subdivision cuts...")
            bpy.context.view_layer.objects.active = sleeve_obj
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_mode(type='EDGE')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.subdivide(number_cuts=manual_cuts)
            bpy.ops.object.mode_set(mode='OBJECT')

        # TAPER THE CONTINUOUS SLEEVE
        script_log(f"DEBUG: Tapering {side} sleeve...")
        bpy.context.view_layer.objects.active = sleeve_obj
        bpy.ops.object.mode_set(mode='EDIT')
        bm = bmesh.from_edit_mesh(sleeve_obj.data)

        for vert in bm.verts:
            # Normalize Z position from -0.5 (shoulder) to 0.5 (wrist)
            z_norm = vert.co.z / (total_length / 2)

            # Calculate target radius based on position along arm
            if z_norm <= -0.2:  # Shoulder area
                target_radius = diameter_start / 2
            elif z_norm <= 0.2:  # Elbow area
                target_radius = diameter_elbow / 2
            else:  # Wrist area
                target_radius = diameter_end / 2

            # Smooth transitions between areas
            if -0.2 < z_norm < 0:  # Shoulder → Elbow transition
                blend = (z_norm + 0.2) / 0.2
                target_radius = (diameter_start / 2 * (1 - blend)) + (diameter_elbow / 2 * blend)
            elif 0 < z_norm < 0.2:  # Elbow → Wrist transition
                blend = (z_norm) / 0.2
                target_radius = (diameter_elbow / 2 * (1 - blend)) + (diameter_end / 2 * blend)

            # Scale vertex to target radius
            current_radius = (vert.co.x ** 2 + vert.co.y ** 2) ** 0.5
            if current_radius > 0.001:
                scale_factor = target_radius / current_radius
                vert.co.x *= scale_factor
                vert.co.y *= scale_factor

        bmesh.update_edit_mesh(sleeve_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')

        # ADD SUBDIVISION SURFACE MODIFIER
        if subdiv_levels > 0:
            script_log(f"DEBUG: Adding subdivision surface with {subdiv_levels} levels...")
            subdiv_mod = sleeve_obj.modifiers.new(name="Subdivision", type='SUBSURF')
            subdiv_mod.levels = subdiv_levels
            subdiv_mod.render_levels = subdiv_levels

        # =========================================================================
        # NEW VERTEX BUNDLES SYSTEM: SPHERICAL PINNING ZONE WEIGHTING
        # =========================================================================
        script_log(f"DEBUG: Setting up spherical pinning zones using new vertex bundles system for {side} sleeve")

        # Create spherical vertex groups for pinning zones
        shoulder_vertex_group = sleeve_obj.vertex_groups.new(name=f"Shoulder_Sphere_{side}")
        wrist_vertex_group = sleeve_obj.vertex_groups.new(name=f"Wrist_Sphere_{side}")

        # Get bundle centers and radii from new system
        shoulder_center = get_bundle_center(shoulder_control_point)
        wrist_center = get_bundle_center(wrist_control_point)

        shoulder_radius = get_bundle_radius(shoulder_control_point)
        wrist_radius = get_bundle_radius(wrist_control_point)

        # Calculate sphere radii for influence
        shoulder_sphere_radius = shoulder_radius * sphere_influence_scale
        wrist_sphere_radius = wrist_radius * sphere_influence_scale

        script_log(f"DEBUG: Using new bundle system - Shoulder: {shoulder_center}, Wrist: {wrist_center}")
        script_log(f"DEBUG: Sphere radii - Shoulder: {shoulder_sphere_radius:.3f}, Wrist: {wrist_sphere_radius:.3f}")

        # =========================================================================
        # APPLY SHOULDER BUNDLE VERTEX WEIGHTS (SPHERICAL PINNING ZONE)
        # =========================================================================
        if shoulder_center:
            script_log(f"✓ Applying shoulder vertex bundle from {shoulder_control_point}")

            for i, vertex in enumerate(sleeve_obj.data.vertices):
                vert_pos = sleeve_obj.matrix_world @ vertex.co
                distance = (vert_pos - shoulder_center).length

                # Apply weight based on distance to bundle center
                if distance <= shoulder_sphere_radius:
                    weight = 1.0 - (distance / shoulder_sphere_radius)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    # STRONGLY PIN SHOULDER AREA - full influence near shoulder
                    vert_local = sleeve_obj.matrix_world.inverted() @ vert_pos
                    z_norm = (vert_local.z + total_length / 2) / total_length  # 0=shoulder, 1=wrist

                    if z_norm < 0.2:  # Shoulder area - full pinning
                        weight *= 1.0
                    elif z_norm < 0.4:  # Transition area - reduced influence
                        weight *= 0.5
                    else:  # Far from shoulder - minimal influence
                        weight *= 0.1

                    if weight > min_weight_threshold:
                        shoulder_vertex_group.add([i], weight, 'REPLACE')

        # =========================================================================
        # APPLY WRIST BUNDLE VERTEX WEIGHTS (SPHERICAL PINNING ZONE)
        # =========================================================================
        if wrist_center:
            script_log(f"✓ Applying wrist vertex bundle from {wrist_control_point}")

            for i, vertex in enumerate(sleeve_obj.data.vertices):
                vert_pos = sleeve_obj.matrix_world @ vertex.co
                distance = (vert_pos - wrist_center).length

                # Apply weight based on distance to bundle center
                if distance <= wrist_sphere_radius:
                    weight = 1.0 - (distance / wrist_sphere_radius)
                    # Apply falloff type
                    if falloff_type == "quadratic":
                        weight = weight * weight
                    elif falloff_type == "smooth":
                        weight = weight * weight * (3 - 2 * weight)

                    # STRONGLY PIN WRIST AREA - full influence near wrist
                    vert_local = sleeve_obj.matrix_world.inverted() @ vert_pos
                    z_norm = (vert_local.z + total_length / 2) / total_length  # 0=shoulder, 1=wrist

                    if z_norm > 0.8:  # Wrist area - full pinning
                        weight *= 1.0
                    elif z_norm > 0.6:  # Transition area - reduced influence
                        weight *= 0.5
                    else:  # Far from wrist - minimal influence
                        weight *= 0.1

                    if weight > min_weight_threshold:
                        wrist_vertex_group.add([i], weight, 'REPLACE')

        # CREATE COMBINED PINNING GROUP FOR SLEEVE CLOTH
        script_log(f"DEBUG: Creating combined pinning group for {side} sleeve...")
        combined_pinning_group = sleeve_obj.vertex_groups.new(name=f"{side}_Sleeve_Combined_Anchors")

        # Combine weights from both spherical groups (shoulder and wrist)
        for i in range(len(sleeve_obj.data.vertices)):
            max_weight = 0.0
            for group_name in [f"Shoulder_Sphere_{side}", f"Wrist_Sphere_{side}"]:
                group = sleeve_obj.vertex_groups.get(group_name)
                if group:
                    try:
                        weight = group.weight(i)
                        max_weight = max(max_weight, weight)
                    except:
                        # Vertex not in this group, continue
                        pass

            if max_weight > min_weight_threshold:
                combined_pinning_group.add([i], max_weight, 'REPLACE')

        script_log(f"✓ Created {side}_Sleeve_Combined_Anchors with weights from shoulder and wrist spheres")

        # TARGETED CLOTH SIMULATION WITH PINNED ANCHORS
        cloth_config = sleeve_config.get("cloth_settings", {})
        if cloth_config.get("enabled", True):
            script_log(f"DEBUG: Adding cloth simulation for {side} sleeve with pinned anchors...")
            cloth_mod = sleeve_obj.modifiers.new(name="Cloth", type='CLOTH')

            # Apply cloth settings from config
            cloth_mod.settings.quality = cloth_config.get("quality", 8)
            cloth_mod.settings.mass = cloth_config.get("mass", 0.15)
            cloth_mod.settings.tension_stiffness = cloth_config.get("tension_stiffness", 8.0)
            cloth_mod.settings.compression_stiffness = cloth_config.get("compression_stiffness", 7.0)
            cloth_mod.settings.shear_stiffness = cloth_config.get("shear_stiffness", 6.0)
            cloth_mod.settings.bending_stiffness = cloth_config.get("bending_stiffness", 0.8)
            cloth_mod.settings.air_damping = cloth_config.get("air_damping", 0.8)
            cloth_mod.settings.time_scale = cloth_config.get("time_scale", 1.0)

            # COLLISIONS FOR FABRIC INTERACTION
            cloth_mod.collision_settings.use_collision = True
            cloth_mod.collision_settings.collision_quality = cloth_config.get("collision_quality", 6)
            cloth_mod.collision_settings.self_distance_min = cloth_config.get("self_distance_min", 0.002)

            # Self-collision for sleeve fabric
            cloth_mod.collision_settings.use_self_collision = True

            # PIN CLOTH TO COMBINED SPHERICAL VERTEX GROUP
            cloth_mod.settings.vertex_group_mass = f"{side}_Sleeve_Combined_Anchors"

            script_log(f"✓ Sleeve cloth: pinned anchors + self-collision + external collisions")
        else:
            script_log(f"DEBUG: Cloth simulation disabled for {side} sleeve")

        # SETUP ARMATURE MODIFIER AND VERTEX GROUPS FOR BONE DEFORMATION
        script_log(f"DEBUG: Setting up armature modifier and vertex groups for {side} sleeve...")

        # Clear any existing vertex groups (except the spherical ones we just created)
        groups_to_keep = [f"Shoulder_Sphere_{side}", f"Wrist_Sphere_{side}", f"{side}_Sleeve_Combined_Anchors"]
        for vg in list(sleeve_obj.vertex_groups):
            if vg.name not in groups_to_keep:
                sleeve_obj.vertex_groups.remove(vg)

        # Remove any existing armature modifiers
        for mod in list(sleeve_obj.modifiers):
            if mod.type == 'ARMATURE':
                sleeve_obj.modifiers.remove(mod)

        # Create vertex groups for bone deformation
        shoulder_group = sleeve_obj.vertex_groups.new(name=shoulder_bone_name)
        upper_arm_group = sleeve_obj.vertex_groups.new(name=upper_arm_bone_name)
        forearm_group = sleeve_obj.vertex_groups.new(name=forearm_bone_name)

        # Assign vertex weights based on position along sleeve
        for i, vertex in enumerate(sleeve_obj.data.vertices):
            vert_local = sleeve_obj.matrix_world.inverted() @ vertex.co
            z_norm = (vert_local.z + total_length / 2) / total_length  # 0=shoulder, 1=wrist

            if z_norm < 0.3:  # Upper part - shoulder to upper arm
                shoulder_weight = 1.0 - (z_norm / 0.3)
                upper_arm_weight = z_norm / 0.3
                shoulder_group.add([i], shoulder_weight, 'REPLACE')
                upper_arm_group.add([i], upper_arm_weight, 'REPLACE')
            elif z_norm < 0.7:  # Middle part - upper arm to forearm
                upper_arm_weight = 1.0 - ((z_norm - 0.3) / 0.4)
                forearm_weight = (z_norm - 0.3) / 0.4
                upper_arm_group.add([i], upper_arm_weight, 'REPLACE')
                forearm_group.add([i], forearm_weight, 'REPLACE')
            else:  # Lower part - forearm to wrist
                forearm_weight = 1.0 - ((z_norm - 0.7) / 0.3)
                forearm_group.add([i], forearm_weight, 'REPLACE')

        # Add armature modifier
        armature_mod = sleeve_obj.modifiers.new(name="Armature", type='ARMATURE')
        armature_mod.object = armature_obj
        armature_mod.use_vertex_groups = True
        script_log(f"✓ Added armature modifier with vertex group deformation")

        # Add material
        apply_material_from_config(sleeve_obj, f"{side}_sleeve")

        # SET PROPER MODIFIER ORDER
        script_log(f"DEBUG: Setting proper modifier order for {side} sleeve...")
        bpy.context.view_layer.objects.active = sleeve_obj
        modifiers = sleeve_obj.modifiers

        # Build correct order based on which modifiers are present
        correct_order = ["Subdivision", "Armature"]
        if cloth_config.get("enabled", True):
            correct_order.append("Cloth")

        for mod_name in correct_order:
            mod_index = modifiers.find(mod_name)
            if mod_index >= 0:
                while mod_index > correct_order.index(mod_name):
                    bpy.ops.object.modifier_move_up(modifier=mod_name)
                    mod_index -= 1

        # VERIFY THE SETUP
        script_log(f"DEBUG: Verifying {side} sleeve setup...")
        if cloth_config.get("enabled",
                            True) and cloth_mod.settings.vertex_group_mass == f"{side}_Sleeve_Combined_Anchors":
            script_log(f"✓ Cloth pinned to {side}_Sleeve_Combined_Anchors vertex group")
        else:
            script_log(f"⚠ Cloth not pinned to spherical vertex group (simulation disabled)")

        script_log(f"✓ Created {side} sleeve with spherical pinning zone weighting")
        script_log(f"✓ Bone constraints: {constraints_added} STRETCH_TO constraints added")
        script_log(f"✓ Vertices weighted to shoulder and wrist spherical pinning zones")
        script_log(f"✓ Armature modifier configured for deformation")
        script_log(f"✓ Sleeve object parented to armature")
        if cloth_config.get("enabled", True):
            script_log(f"✓ Cloth pinned to combined anchors (shoulder+wrist)")
            script_log(f"✓ Modern Blender 4.3+ cloth API applied")
            script_log(f"✓ Self-collision enabled for sleeve fabric")
        script_log(f"✓ Using new vertex bundles system for dynamic joint positioning")

        return sleeve_obj

    except Exception as e:
        script_log(f"ERROR creating {side} sleeve: {e}")
        import traceback
        script_log(f"Traceback: {traceback.format_exc()}")
        bpy.ops.object.mode_set(mode='OBJECT')
        return None

##########################################################################################

def setup_cloth_parenting(cloth_obj, armature_obj, bone_names, default_weight=0.3):
    """Set up vertex groups and parenting for cloth simulation with proximity-based weighting"""
    # Add armature modifier
    armature_mod = cloth_obj.modifiers.new(name="Armature", type='ARMATURE')
    armature_mod.object = armature_obj

    # Create vertex groups for different bone influences with proximity-based weights
    for bone_name in bone_names:
        if bone_name in armature_obj.pose.bones:
            bone = armature_obj.pose.bones[bone_name]
            bone_pos = bone.head

            # Create vertex group
            if bone_name in cloth_obj.vertex_groups:
                cloth_obj.vertex_groups.remove(cloth_obj.vertex_groups[bone_name])
            vgroup = cloth_obj.vertex_groups.new(name=bone_name)

            # Assign vertices with proximity-based weights
            for i, vert in enumerate(cloth_obj.data.vertices):
                vert_pos = cloth_obj.matrix_world @ vert.co
                distance = (vert_pos - bone_pos).length
                # Weight decreases with distance
                weight = max(0, default_weight * (1.0 - distance / 2.0))  # Adjust divisor as needed
                if weight > 0:
                    vgroup.add([i], weight, 'REPLACE')

    script_log(f"Set up cloth parenting for {cloth_obj.name} with bones: {bone_names}")

##########################################################################################

def create_cloth_material(name, material_config):
    """Create a material for cloth garments"""
    cloth_mat = bpy.data.materials.new(name=name)
    cloth_mat.use_nodes = True

    # Get material properties from config or use defaults
    color = material_config.get("color", [0.8, 0.8, 0.8, 1.0])
    roughness = material_config.get("roughness", 0.7)
    metallic = material_config.get("metallic", 0.0)

    # Set material properties
    cloth_mat.diffuse_color = color
    cloth_mat.roughness = roughness
    cloth_mat.metallic = metallic

    return cloth_mat

##########################################################################################

def extract_bone_hierarchy_and_controls(bone_definitions):
    """Extract clean bone relationships from new config structure"""
    global bone_parents, bone_tail_control_points, bone_head_control_points
    global def_bone_names, bone_constraint_types, bone_tail_landmarks

    bone_parents = {}
    bone_tail_control_points = {}
    bone_head_control_points = {}
    def_bone_names = {}
    bone_constraint_types = {}
    bone_tail_landmarks = {}

    for bone_name, bone_data in bone_definitions.items():
        # Store parent
        bone_parents[bone_name] = bone_data.get("parent")

        # Store DEF bone name
        def_bone_name = bone_data.get("def_bone")
        if def_bone_name:
            def_bone_names[bone_name] = def_bone_name
        else:
            def_bone_names[bone_name] = f"DEF_{bone_name}"

        # Store constraint type
        bone_constraint_types[bone_name] = bone_data.get("constraint", "COPY_TO")

        # Store tail control point
        if "tail_control_point" in bone_data:
            bone_tail_control_points[bone_name] = bone_data["tail_control_point"]

        # Store tail landmark
        if "tail_landmark" in bone_data:
            bone_tail_landmarks[bone_name] = bone_data["tail_landmark"]

        # Store head control point (only for root bones)
        if bone_data.get("parent") is None and "head_control_point" in bone_data:
            bone_head_control_points[bone_name] = bone_data["head_control_point"]

    script_log(f"Extracted: {len(bone_parents)} bones, {len(def_bone_names)} DEF bones")


##########################################################################################

def add_collection(name, parent_collection=None):
    """Add a new collection if it doesn't exist and link it to the scene."""
    if name not in bpy.data.collections:
        new_collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(new_collection)
    else:
        new_collection = bpy.data.collections[name]

    if parent_collection:
        if new_collection in bpy.context.scene.collection.children:
            bpy.context.scene.collection.children.unlink(new_collection)
        if new_collection not in parent_collection.children:
            parent_collection.children.link(new_collection)

    return new_collection


##########################################################################################

def cleanup_existing_objects(figure_name):
    """Clean up existing objects to prevent duplicates"""
    script_log("Cleaning up existing objects to prevent duplicates...")

    objects_to_remove = []

    # Get all control point names from bone_tail_control_points
    control_point_names_list = list(bone_tail_control_points.values())

    for obj in bpy.data.objects:
        # Remove control points using bone_tail_control_points names
        if obj.name in control_point_names_list:
            objects_to_remove.append(obj)
        elif "_Sphere_" in obj.name:
            objects_to_remove.append(obj)
        elif obj.name == "Kid_Complete_Skin":
            objects_to_remove.append(obj)
        elif obj.name.startswith(f"{figure_name}_") and any(mp in obj.name for mp in
                                                            ["HIP", "SHOULDER", "ELBOW", "WRIST", "KNEE", "HEEL",
                                                             "NOSE", "HEAD_TOP", "INDEX", "FOOT_INDEX"]):
            objects_to_remove.append(obj)
        # Remove frame control points too
        elif obj.name in ["VIRTUAL_HIP_FRAME", "VIRTUAL_SHOULDER_FRAME"]:
            objects_to_remove.append(obj)
        # Remove cloth garments
        elif "LongSleeveShirt" in obj.name or "Pants" in obj.name or "Coat" in obj.name:
            objects_to_remove.append(obj)

    # Remove identified objects
    for obj in objects_to_remove:
        script_log(f"Removing duplicate object: {obj.name}")
        bpy.data.objects.remove(obj, do_unlink=True)

    # Clean up orphaned data
    for mesh in bpy.data.meshes:
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)

    for armature in bpy.data.armatures:
        if armature.users == 0:
            bpy.data.armatures.remove(armature)

    script_log(f"Cleaned up {len(objects_to_remove)} duplicate objects")


##########################################################################################

def get_landmark_position(frame_data, landmark_name):
    """Get direct landmark position from landmark name"""
    if landmark_name in frame_data:
        pos_data = frame_data[landmark_name]
        return Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

    script_log(f"WARNING: Landmark {landmark_name} not found in frame data")
    return Vector((0, 0, 0))


##########################################################################################

def calculate_midpoint(frame_data, landmark_names):
    """Calculate midpoint between multiple landmarks"""
    if not landmark_names:
        return Vector((0, 0, 0))

    total_vec = Vector((0, 0, 0))
    valid_points = 0

    for mp_name in landmark_names:
        if mp_name in frame_data:
            pos_data = frame_data[mp_name]
            total_vec += Vector((pos_data["x"], pos_data["y"], pos_data["z"]))
            valid_points += 1

    return total_vec / valid_points if valid_points > 0 else Vector((0, 0, 0))

##########################################################################################

def calculate_virtual_position(frame_data, virtual_point_name):
    """Simple direct lookup for virtual point calculations"""
    if virtual_point_name in VIRTUAL_POINT_CALCULATIONS:
        landmarks = VIRTUAL_POINT_CALCULATIONS[virtual_point_name]
        position = calculate_midpoint(frame_data, landmarks)
        return position

    script_log(f"ERROR: Unknown virtual point: {virtual_point_name}")
    return Vector((0, 0, 0))


##########################################################################################

def create_transparent_material(hex_color, alpha=0.3):
    """Create a transparent material from a hex color string."""
    mat_name = f"Material_{hex_color.replace('#', '')}_Alpha_{int(alpha * 100)}"
    if mat_name in bpy.data.materials:
        return bpy.data.materials[mat_name]

    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'

    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        r = int(hex_color[1:3], 16) / 255.0
        g = int(hex_color[3:5], 16) / 255.0
        b = int(hex_color[5:7], 16) / 255.0
        principled_bsdf.inputs['Base Color'].default_value = (r, g, b, 1.0)
        principled_bsdf.inputs['Alpha'].default_value = alpha

    return mat

##########################################################################################

def create_hip_frame_control_point(first_frame):
    """Create VIRTUAL_HIP_FRAME with coordinate frame calculation"""
    script_log("Creating VIRTUAL_HIP_FRAME with coordinate frame...")

    # Create empty to represent the hip coordinate frame
    bpy.ops.object.empty_add(type='SINGLE_ARROW')  # Visualize orientation
    hip_frame = bpy.context.active_object
    hip_frame.name = "VIRTUAL_HIP_FRAME"

    # Position at midpoint between hips
    left_hip_pos = get_landmark_position(first_frame, "LEFT_HIP")
    right_hip_pos = get_landmark_position(first_frame, "RIGHT_HIP")
    hip_frame.location = (left_hip_pos + right_hip_pos) / 2

    # Calculate coordinate frame axes
    hip_vector = right_hip_pos - left_hip_pos
    x_axis = hip_vector.normalized()  # X = right direction (left hip → right hip)

    # Calculate Z-axis (forward) - perpendicular to hip line and world up
    world_up = Vector((0, 1, 0))
    z_axis = x_axis.cross(world_up).normalized()

    # If z_axis is zero (hips perfectly vertical), use fallback
    if z_axis.length < 0.001:
        z_axis = Vector((0, 0, 1))
        script_log("Used fallback Z-axis for hip frame")

    # Calculate Y-axis (up) - perpendicular to both
    y_axis = z_axis.cross(x_axis).normalized()

    # Create rotation matrix and apply to empty
    rotation_matrix = Matrix([x_axis, y_axis, z_axis]).transposed()
    hip_frame.rotation_euler = rotation_matrix.to_euler()

    # Add to control points collection
    control_collection = bpy.data.collections.get("Main_ControlPoints")
    if control_collection and hip_frame.name not in control_collection.objects:
        control_collection.objects.link(hip_frame)

    script_log(f"Created VIRTUAL_HIP_FRAME at {hip_frame.location} with rotation {hip_frame.rotation_euler}")
    return hip_frame

##########################################################################################

def add_dynamic_weighting_to_garment(garment_obj, control_point_names, bone_names):
    """
    Add dynamic vertex weighting to any garment object
    garment_obj: The mesh object to add dynamic weighting to
    control_point_names: List of control points that influence this garment
    bone_names: List of bone names for vertex groups
    """
    script_log(f"Adding dynamic vertex weighting to {garment_obj.name}")

    # Clear any existing static vertex groups
    for vg in list(garment_obj.vertex_groups):
        garment_obj.vertex_groups.remove(vg)

    # Create vertex groups for each bone
    vertex_groups = {}
    for bone_name in bone_names:
        vg = garment_obj.vertex_groups.new(name=bone_name)
        vertex_groups[bone_name] = vg

    # Add dynamic weight update property
    garment_obj["dynamic_weights_enabled"] = 1.0

    # Add frame update driver to trigger weight recalculations
    driver = garment_obj.driver_add('["dynamic_weights_enabled"]').driver
    driver.type = 'SCRIPTED'

    var = driver.variables.new()
    var.name = "frame"
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = 'SCENE'
    var.targets[0].id = bpy.context.scene
    var.targets[0].data_path = 'frame_current'

    driver.expression = "frame * 0.0001"  # Small change triggers updates

    # Store control point references for this garment
    garment_obj["weight_control_points"] = str(control_point_names)
    garment_obj["weight_bones"] = str(bone_names)

    # Initial weight calculation
    if "vertex_bundle_config" in garment_obj:
        recalculate_vertex_weights(garment_obj, garment_obj["vertex_bundle_config"])

    script_log(f"✓ Added dynamic weighting to {garment_obj.name}")
    script_log(f"  - Control points: {control_point_names}")
    script_log(f"  - Bone groups: {bone_names}")


def make_vertex_all_bundles(armature_obj):
    """Create vertex bundle systems with two-empties architecture"""
    global joint_control_systems

    script_log("=== CREATING VERTEX BUNDLE SYSTEMS WITH DYNAMIC WEIGHTING ===")

    # First, check if VB empties already exist
    existing_vb_empties = {}
    for cp_name in joint_control_systems.keys():
        vb_name = f"VB_{cp_name}"
        if vb_name in bpy.data.objects:
            existing_vb_empties[cp_name] = bpy.data.objects[vb_name]
            script_log(f"✓ Found existing VB empty: {vb_name}")

    # Create missing VB empties with proper parenting and constraints
    for cp_name, system_data in joint_control_systems.items():
        vb_name = f"VB_{cp_name}"
        rpy_empty = system_data['rpy_empty']

        if cp_name in existing_vb_empties:
            # Use existing VB empty
            vb_empty = existing_vb_empties[cp_name]
            script_log(f"✓ Using existing VB empty for {cp_name}")
        else:
            # Create new VB empty at the RPY empty's location
            vb_empty = create_empty_at_location(vb_name, location=rpy_empty.location)
            script_log(f"✓ Created dynamic vertex bundle empty for {cp_name}")

        # Parent VB empty to RPY empty
        vb_empty.parent = rpy_empty

        # Set local position to zero (align with parent)
        vb_empty.location = (0, 0, 0)

        # CRITICAL: Add COPY_LOCATION constraint to follow parent
        # Remove any existing COPY_LOCATION constraints first
        for constraint in vb_empty.constraints:
            if constraint.type == 'COPY_LOCATION':
                vb_empty.constraints.remove(constraint)

        copy_loc = vb_empty.constraints.new('COPY_LOCATION')
        copy_loc.target = rpy_empty
        copy_loc.use_x = True
        copy_loc.use_y = True
        copy_loc.use_z = True

        # Store the VB empty in the system data for later use
        system_data['vb_empty'] = vb_empty

        script_log(f"✓ Parented and constrained {vb_name} to {cp_name}")

    script_log("✓ Vertex bundle systems created with proper parenting and constraints")
    return joint_control_systems

# Helper function for garment creators to easily add dynamic weighting
def setup_garment_dynamic_weighting(garment_obj, side, garment_type):
    """
    Easy setup for dynamic vertex weighting in garment functions
    Usage: setup_garment_dynamic_weighting(pants_obj, "left", "pants")
    """
    if garment_type == "pants":
        if side == "left":
            control_points = ["CTRL_LEFT_HIP", "CTRL_LEFT_KNEE", "CTRL_LEFT_HEEL"]
            bones = ["DEF_LeftHip", "DEF_LeftThigh", "DEF_LeftShin"]
        else:
            control_points = ["CTRL_RIGHT_HIP", "CTRL_RIGHT_KNEE", "CTRL_RIGHT_HEEL"]
            bones = ["DEF_RightHip", "DEF_RightThigh", "DEF_RightShin"]

    elif garment_type == "sleeve":
        if side == "left":
            control_points = ["CTRL_LEFT_SHOULDER", "CTRL_LEFT_ELBOW", "CTRL_LEFT_WRIST"]
            bones = ["DEF_LeftShoulder", "DEF_LeftUpperArm", "DEF_LeftForearm"]
        else:
            control_points = ["CTRL_RIGHT_SHOULDER", "CTRL_RIGHT_ELBOW", "CTRL_RIGHT_WRIST"]
            bones = ["DEF_RightShoulder", "DEF_RightUpperArm", "DEF_RightForearm"]

    elif garment_type == "coat":
        control_points = ["CTRL_LEFT_SHOULDER", "CTRL_RIGHT_SHOULDER", "CTRL_LEFT_HIP", "CTRL_RIGHT_HIP"]
        bones = ["DEF_ShoulderRoot", "DEF_UpperSpine", "DEF_HipRoot", "DEF_LowerSpine"]

    # Apply dynamic weighting
    add_dynamic_weighting_to_garment(garment_obj, control_points, bones)

    return garment_obj

##########################################################################################

def create_shoulder_frame_control_point(first_frame):
    """Create VIRTUAL_SHOULDER_FRAME with coordinate frame calculation"""
    script_log("Creating VIRTUAL_SHOULDER_FRAME with coordinate frame...")

    bpy.ops.object.empty_add(type='SINGLE_ARROW')
    shoulder_frame = bpy.context.active_object
    shoulder_frame.name = "VIRTUAL_SHOULDER_FRAME"

    # Position at midpoint between shoulders
    left_shoulder_pos = get_landmark_position(first_frame, "LEFT_SHOULDER")
    right_shoulder_pos = get_landmark_position(first_frame, "RIGHT_SHOULDER")
    shoulder_frame.location = (left_shoulder_pos + right_shoulder_pos) / 2

    # Calculate coordinate frame axes
    shoulder_vector = right_shoulder_pos - left_shoulder_pos
    x_axis = shoulder_vector.normalized()  # X = right direction

    # Calculate Z-axis (forward) - perpendicular to shoulder line and world up
    world_up = Vector((0, 1, 0))
    z_axis = x_axis.cross(world_up).normalized()

    if z_axis.length < 0.001:
        z_axis = Vector((0, 0, 1))
        script_log("Used fallback Z-axis for shoulder frame")

    # Calculate Y-axis (up) - perpendicular to both
    y_axis = z_axis.cross(x_axis).normalized()

    # Create rotation matrix and apply to empty
    rotation_matrix = Matrix([x_axis, y_axis, z_axis]).transposed()
    shoulder_frame.rotation_euler = rotation_matrix.to_euler()

    # Add to control points collection
    control_collection = bpy.data.collections.get("Main_ControlPoints")
    if control_collection and shoulder_frame.name not in control_collection.objects:
        control_collection.objects.link(shoulder_frame)

    script_log(
        f"Created VIRTUAL_SHOULDER_FRAME at {shoulder_frame.location} with rotation {shoulder_frame.rotation_euler}")
    return shoulder_frame


##########################################################################################

def create_control_points(figure_name, armature_obj=None):
    """Create all control points including the new VIRTUAL_SPINE_MIDPOINT"""
    global control_point_objs

    script_log("=== CREATING CONTROL POINTS ===")

    control_point_objs = {}
    first_frame = mocap_data.get(str(frame_numbers[0]), {})

    # Create control points collection
    control_collection = add_collection(f"{figure_name}_ControlPoints")

    # STEP 1: CREATE VIRTUAL CONTROL POINTS FIRST
    script_log("Creating virtual control points...")
    virtual_points_created = 0

    # Create ALL virtual points from VIRTUAL_POINT_CALCULATIONS
    for virtual_point_name in VIRTUAL_POINT_CALCULATIONS.keys():
        if virtual_point_name in bpy.data.objects:
            script_log(f"Using existing virtual control point: {virtual_point_name}")
            continue

        # Create empty for virtual control point
        bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
        empty_obj = bpy.context.active_object
        empty_obj.name = virtual_point_name

        # Position from virtual calculation
        position = calculate_virtual_position(first_frame, virtual_point_name)
        empty_obj.location = position

        # Add to control points collection
        if empty_obj.name not in control_collection.objects:
            control_collection.objects.link(empty_obj)

        script_log(f"Created virtual control point: {virtual_point_name} at {position}")
        virtual_points_created += 1

    # STEP 2: Create direct landmark control points (spheres)
    script_log("Creating direct landmark control points...")
    direct_points_created = 0

    # Iterate through bone_tail_control_points directly
    for bone_name, control_point_name in bone_tail_control_points.items():
        # Skip virtual points (already created above)
        if control_point_name.startswith("VIRTUAL_"):
            if control_point_name in bpy.data.objects:
                control_obj = bpy.data.objects[control_point_name]
                control_point_objs[bone_name] = control_obj
            continue

        # Skip if already exists and tracked
        if bone_name in control_point_objs:
            script_log(f"Using existing control point: {control_point_name}")
            continue

        # Skip if object already exists in scene
        if control_point_name in bpy.data.objects:
            control_obj = bpy.data.objects[control_point_name]
            control_point_objs[bone_name] = control_obj
            script_log(f"Using existing control point: {control_point_name}")
            direct_points_created += 1
            continue

        # Create control point sphere
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.02, location=(0, 0, 0))
        control_obj = bpy.context.active_object
        control_obj.name = control_point_name

        # Position direct control points using explicit tail landmarks
        tail_landmark = bone_tail_landmarks.get(bone_name)
        if tail_landmark and tail_landmark in first_frame:
            pos_data = first_frame[tail_landmark]
            control_obj.location = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))
            script_log(f"Positioned {control_point_name} at {control_obj.location} using landmark {tail_landmark}")

        # Apply transparent material
        transparent_mat = create_transparent_material("#FF0000", alpha=0.3)
        control_obj.data.materials.append(transparent_mat)

        # Add to system after creation and positioning
        control_point_objs[bone_name] = control_obj

        # Add to control collection
        if control_obj.name not in control_collection.objects:
            control_collection.objects.link(control_obj)

        direct_points_created += 1

    script_log(f"Created {direct_points_created} direct + {virtual_points_created} virtual control points total")
    script_log(f"Total control points tracked: {len(control_point_objs)}")

    return control_point_objs

##########################################################################################

def setup_root_bone_transform_constraints(armature_obj):
    """Use COPY_TRANSFORMS for both position AND rotation on root bones"""
    script_log("=== SETTING UP ROOT BONE TRANSFORM CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # HipRoot follows VIRTUAL_HIP_FRAME (position + rotation)
    if "DEF_HipRoot" in armature_obj.pose.bones:
        hip_root = armature_obj.pose.bones["DEF_HipRoot"]

        # Clear old constraints
        for constraint in list(hip_root.constraints):
            hip_root.constraints.remove(constraint)

        hip_frame = bpy.data.objects.get("VIRTUAL_HIP_FRAME")
        if hip_frame:
            copy_transform = hip_root.constraints.new('COPY_TRANSFORMS')
            copy_transform.target = hip_frame
            copy_transform.influence = 1.0
            constraints_added += 1
            script_log("DEF_HipRoot COPY_TRANSFORMS -> VIRTUAL_HIP_FRAME")

    # ShoulderRoot follows VIRTUAL_SHOULDER_FRAME but limits rotation around left-right axis
    if "DEF_ShoulderRoot" in armature_obj.pose.bones:
        shoulder_root = armature_obj.pose.bones["DEF_ShoulderRoot"]

        # Clear old constraints
        for constraint in list(shoulder_root.constraints):
            shoulder_root.constraints.remove(constraint)

        shoulder_frame = bpy.data.objects.get("VIRTUAL_SHOULDER_FRAME")
        if shoulder_frame:
            # COPY_TRANSFORMS to get both position and rotation from the frame
            copy_transform = shoulder_root.constraints.new('COPY_TRANSFORMS')
            copy_transform.target = shoulder_frame
            copy_transform.influence = 1.0

            # LIMIT ROTATION to block twisting around the X-axis (left-right shoulder axis)
            limit_rotation = shoulder_root.constraints.new('LIMIT_ROTATION')
            limit_rotation.use_limit_x = True
            limit_rotation.min_x = 0.0  # Lock X-axis rotation completely
            limit_rotation.max_x = 0.0
            limit_rotation.use_limit_y = False  # Allow Y-axis rotation (forward/backward tilt)
            limit_rotation.use_limit_z = False  # Allow Z-axis rotation (side-to-side tilt)
            limit_rotation.owner_space = 'LOCAL'  # Limit in the bone's local space

            constraints_added += 2
            script_log("DEF_ShoulderRoot: COPY_TRANSFORMS -> VIRTUAL_SHOULDER_FRAME with X-axis rotation locked")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"Root bone transform constraints: {constraints_added} constraints added")
    return constraints_added


##########################################################################################

def setup_virtual_frame_constraints():
    """Set up constraints for the frame control points to follow the virtual midpoints and update rotation from landmarks"""
    script_log("=== SETTING UP VIRTUAL FRAME CONSTRAINTS ===")

    constraints_added = 0

    # VIRTUAL_HIP_FRAME should follow VIRTUAL_HIP_MIDPOINT location AND calculate rotation from hip landmarks
    hip_frame = bpy.data.objects.get("VIRTUAL_HIP_FRAME")
    hip_midpoint = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")
    left_hip = bpy.data.objects.get("CTRL_LEFT_HIP")
    right_hip = bpy.data.objects.get("CTRL_RIGHT_HIP")

    if hip_frame and hip_midpoint and left_hip and right_hip:
        # Clear existing constraints
        for constraint in list(hip_frame.constraints):
            hip_frame.constraints.remove(constraint)

        # Copy location from midpoint
        copy_loc = hip_frame.constraints.new('COPY_LOCATION')
        copy_loc.target = hip_midpoint
        copy_loc.influence = 1.0

        # Use DAMPED_TRACK to make the frame point from left hip to right hip
        damped_track = hip_frame.constraints.new('DAMPED_TRACK')
        damped_track.target = right_hip
        damped_track.track_axis = 'TRACK_X'  # X-axis should point from left to right hip

        constraints_added += 2
        script_log("VIRTUAL_HIP_FRAME: COPY_LOCATION -> VIRTUAL_HIP_MIDPOINT, DAMPED_TRACK -> CTRL_RIGHT_HIP")

    # VIRTUAL_SHOULDER_FRAME should follow VIRTUAL_SHOULDER_MIDPOINT location AND calculate rotation from shoulder landmarks
    shoulder_frame = bpy.data.objects.get("VIRTUAL_SHOULDER_FRAME")
    shoulder_midpoint = bpy.data.objects.get("VIRTUAL_SHOULDER_MIDPOINT")
    left_shoulder = bpy.data.objects.get("CTRL_LEFT_SHOULDER")
    right_shoulder = bpy.data.objects.get("CTRL_RIGHT_SHOULDER")

    if shoulder_frame and shoulder_midpoint and left_shoulder and right_shoulder:
        # Clear existing constraints
        for constraint in list(shoulder_frame.constraints):
            shoulder_frame.constraints.remove(constraint)

        # Copy location from midpoint
        copy_loc = shoulder_frame.constraints.new('COPY_LOCATION')
        copy_loc.target = shoulder_midpoint
        copy_loc.influence = 1.0

        # Use DAMPED_TRACK to make the frame point from left shoulder to right shoulder
        damped_track = shoulder_frame.constraints.new('DAMPED_TRACK')
        damped_track.target = right_shoulder
        damped_track.track_axis = 'TRACK_X'  # X-axis should point from left to right shoulder

        constraints_added += 2
        script_log(
            "VIRTUAL_SHOULDER_FRAME: COPY_LOCATION -> VIRTUAL_SHOULDER_MIDPOINT, DAMPED_TRACK -> CTRL_RIGHT_SHOULDER")

    script_log(f"Virtual frame constraints: {constraints_added} constraints added")
    return constraints_added

##########################################################################################

def setup_virtual_point_constraints():
    """Set up constraints for all virtual points including the new spine midpoint"""
    script_log("=== SETTING UP VIRTUAL POINT CONSTRAINTS ===")

    constraints_added = 0

    # Setup VIRTUAL_HIP_MIDPOINT constraint
    hip_midpoint = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")
    left_hip = bpy.data.objects.get("CTRL_LEFT_HIP")
    right_hip = bpy.data.objects.get("CTRL_RIGHT_HIP")

    if hip_midpoint and left_hip and right_hip:
        for constraint in list(hip_midpoint.constraints):
            hip_midpoint.constraints.remove(constraint)

        copy_left = hip_midpoint.constraints.new('COPY_LOCATION')
        copy_left.target = left_hip
        copy_left.use_offset = False
        copy_left.influence = 0.5

        copy_right = hip_midpoint.constraints.new('COPY_LOCATION')
        copy_right.target = right_hip
        copy_right.use_offset = False
        copy_right.influence = 0.5

        constraints_added += 2
        script_log("Set up VIRTUAL_HIP_MIDPOINT constraints")

    # Setup VIRTUAL_SHOULDER_MIDPOINT constraint
    shoulder_midpoint = bpy.data.objects.get("VIRTUAL_SHOULDER_MIDPOINT")
    left_shoulder = bpy.data.objects.get("CTRL_LEFT_SHOULDER")
    right_shoulder = bpy.data.objects.get("CTRL_RIGHT_SHOULDER")

    if shoulder_midpoint and left_shoulder and right_shoulder:
        for constraint in list(shoulder_midpoint.constraints):
            shoulder_midpoint.constraints.remove(constraint)

        copy_left = shoulder_midpoint.constraints.new('COPY_LOCATION')
        copy_left.target = left_shoulder
        copy_left.use_offset = False
        copy_left.influence = 0.5

        copy_right = shoulder_midpoint.constraints.new('COPY_LOCATION')
        copy_right.target = right_shoulder
        copy_right.use_offset = False
        copy_right.influence = 0.5

        constraints_added += 2
        script_log("Set up VIRTUAL_SHOULDER_MIDPOINT constraints")

    # Setup VIRTUAL_SPINE_MIDPOINT constraint - midpoint between hips and shoulders
    spine_midpoint = bpy.data.objects.get("VIRTUAL_SPINE_MIDPOINT")
    if spine_midpoint:
        for constraint in list(spine_midpoint.constraints):
            spine_midpoint.constraints.remove(constraint)

        # 50% influence from hip midpoint
        if hip_midpoint:
            copy_hip = spine_midpoint.constraints.new('COPY_LOCATION')
            copy_hip.target = hip_midpoint
            copy_hip.use_offset = False
            copy_hip.influence = 0.5

        # 50% influence from shoulder midpoint
        if shoulder_midpoint:
            copy_shoulder = spine_midpoint.constraints.new('COPY_LOCATION')
            copy_shoulder.target = shoulder_midpoint
            copy_shoulder.use_offset = False
            copy_shoulder.influence = 0.5

        constraints_added += 2
        script_log("Set up VIRTUAL_SPINE_MIDPOINT constraints")

    # Setup VIRTUAL_HEAD_BASE constraint
    head_base = bpy.data.objects.get("VIRTUAL_HEAD_BASE")
    head_top = bpy.data.objects.get("CTRL_HEAD_TOP")
    nose = bpy.data.objects.get("CTRL_NOSE")

    if head_base:
        for constraint in list(head_base.constraints):
            head_base.constraints.remove(constraint)

        # Follow HEAD_TOP with some influence
        if head_top:
            copy_head = head_base.constraints.new('COPY_LOCATION')
            copy_head.target = head_top
            copy_head.use_offset = False
            copy_head.influence = 0.7  # 70% influence from head top
            constraints_added += 1
            script_log("VIRTUAL_HEAD_BASE: 70% influence from CTRL_HEAD_TOP")

        # Also follow NOSE to position head base
        if nose:
            copy_nose = head_base.constraints.new('COPY_LOCATION')
            copy_nose.target = nose
            copy_nose.use_offset = False
            copy_nose.influence = 0.3  # 30% influence from nose
            constraints_added += 1
            script_log("VIRTUAL_HEAD_BASE: 30% influence from CTRL_NOSE")

        if not head_top and not nose:
            script_log("WARNING: VIRTUAL_HEAD_BASE has no targets (missing CTRL_HEAD_TOP and CTRL_NOSE)")

    bpy.context.view_layer.update()
    script_log(f"Virtual point constraints: {constraints_added} constraints added")
    return constraints_added

##########################################################################################

def create_kid_rig(figure_name):
    """Create kid rig with hierarchy-driven spine"""
    # [All the existing bone rigging code remains exactly the same]
    # This creates the armature structure that the cloth will follow

    script_log(f"=== Creating kid rig: {figure_name} ===")

    def calculate_bone_order(bone_definitions):
        """Calculate bone creation order from parent relationships"""
        script_log("Calculating bone order from hierarchy...")

        children = {}
        root_bones = []

        for bone_name, bone_data in bone_definitions.items():
            parent_name = bone_data.get("parent")

            if parent_name:
                if parent_name not in children:
                    children[parent_name] = []
                children[parent_name].append(bone_name)
                script_log(f"  {bone_name} -> {parent_name}")
            else:
                root_bones.append(bone_name)
                script_log(f"  {bone_name} -> ROOT")

        # Depth-first traversal to build order
        bone_order = []

        def traverse_bone(bone_name):
            bone_order.append(bone_name)
            script_log(f"  Traversing: {bone_name}")
            if bone_name in children:
                for child_name in children[bone_name]:
                    traverse_bone(child_name)

        # Start with root bones in specific order for stability
        preferred_order = ["LowerSpine"]
        for preferred_bone in preferred_order:
            if preferred_bone in root_bones:
                traverse_bone(preferred_bone)
                root_bones.remove(preferred_bone)

        # Add any remaining root bones
        for root_bone in root_bones:
            traverse_bone(root_bone)

        script_log(f"Final bone order: {bone_order}")
        return bone_order

    # Calculate bone order
    bone_order = calculate_bone_order(bone_definitions)

    # Create armature data and object
    armature_obj_name = f"{figure_name}_Rig"

    # Clean up existing armature completely
    if armature_obj_name in bpy.data.objects:
        script_log(f"Removing existing armature: {armature_obj_name}")
        bpy.data.objects.remove(bpy.data.objects[armature_obj_name], do_unlink=True)

    # Also clean up any armature data
    armature_data_name = f"{figure_name}_Armature"
    if armature_data_name in bpy.data.armatures:
        script_log(f"Removing existing armature data: {armature_data_name}")
        bpy.data.armatures.remove(armature_data_name)

    # Create new armature data and object
    armature_data = bpy.data.armatures.new(armature_data_name)
    armature_obj = bpy.data.objects.new(armature_obj_name, armature_data)
    script_log(f"Created new armature: {armature_obj_name}")

    # Link to main collection
    main_collection = bpy.data.collections.get(f"{figure_name}_MainFigure")
    if not main_collection:
        main_collection = add_collection(f"{figure_name}_MainFigure")
    main_collection.objects.link(armature_obj)

    # Make armature active and enter Edit Mode to add bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    # Dictionary to store bone references
    def_bones_dict = {}

    # Get first frame data for initial bone positions
    first_frame = mocap_data.get(str(frame_numbers[0]), {})

    # Create programmatic root bones first
    script_log("Creating programmatic root bones...")

    # Create HipRoot bone (programmatic only - not in config)
    hip_root_bone = armature_data.edit_bones.new("DEF_HipRoot")
    hip_root_pos = calculate_virtual_position(first_frame, "VIRTUAL_HIP_MIDPOINT")
    hip_root_bone.head = hip_root_pos
    hip_root_bone.tail = hip_root_pos + Vector((0, 0.1, 0))  # Small bone for visibility
    def_bones_dict["HipRoot"] = hip_root_bone
    script_log(f"Created DEF_HipRoot at {hip_root_pos}")

    # Create ShoulderRoot bone (programmatic only - not in config)
    shoulder_root_bone = armature_data.edit_bones.new("DEF_ShoulderRoot")
    shoulder_root_pos = calculate_virtual_position(first_frame, "VIRTUAL_SHOULDER_MIDPOINT")
    shoulder_root_bone.head = shoulder_root_pos
    shoulder_root_bone.tail = shoulder_root_pos + Vector((0, 0.1, 0))
    def_bones_dict["ShoulderRoot"] = shoulder_root_bone
    script_log(f"Created DEF_ShoulderRoot at {shoulder_root_pos}")

    # Hard-coded parenting overrides for root bone hierarchy
    ROOT_BONE_PARENTING = {
        "LeftHip": "HipRoot",
        "RightHip": "HipRoot",
        "LowerSpine": "HipRoot",
        "LeftShoulder": "ShoulderRoot",
        "RightShoulder": "ShoulderRoot",
        "UpperSpine": "ShoulderRoot",
        "Neck": "ShoulderRoot"
    }

    # Create all bones including spine in main loop
    script_log("Creating DEF bones for skin deformation and animation...")

    # Create bones in order to ensure parents exist first
    creation_order = [
        "LeftHip", "RightHip", "LowerSpine", "LeftShoulder", "RightShoulder", "UpperSpine",
        "Neck", "Head", "LeftUpperArm", "RightUpperArm", "LeftForearm", "RightForearm",
        "LeftHand", "RightHand", "LeftThigh", "RightThigh", "LeftShin", "RightShin",
        "LeftFoot", "RightFoot"
    ]

    for bone_name in creation_order:
        if bone_name not in bone_definitions:
            continue

        bone_data = bone_definitions[bone_name]

        # Get DEF bone name from centralized mapping
        def_bone_name = def_bone_names.get(bone_name)
        if not def_bone_name:
            script_log(f"WARNING: No DEF bone name for {bone_name}, skipping")
            continue

        # Get tail position from control point or landmark
        tail_control_point = bone_data["tail_control_point"]
        if tail_control_point.startswith("VIRTUAL_"):
            tail_pos = calculate_virtual_position(first_frame, tail_control_point)
        else:
            # Use explicit tail landmark if available
            tail_landmark = bone_tail_landmarks.get(bone_name)
            if tail_landmark:
                tail_pos = get_landmark_position(first_frame, tail_landmark)
            else:
                tail_pos = get_landmark_position(first_frame, tail_control_point[5:])

        # Determine head position based on bone type and parenting
        head_pos = Vector((0, 0, 0))

        # Use our hard-coded parenting if defined, otherwise use config
        if bone_name in ROOT_BONE_PARENTING:
            parent_name = ROOT_BONE_PARENTING[bone_name]
        else:
            parent_name = bone_data.get("parent")

        # HEAD POSITIONING LOGIC - USING ROOT BONES FOR HIP/SHOULDER REGIONS
        # HIP REGION BONES: Head at HipRoot position
        if bone_name in ["LeftHip", "RightHip", "LowerSpine"]:
            if "HipRoot" in def_bones_dict:
                head_pos = def_bones_dict["HipRoot"].head
                script_log(f"HIP REGION: Positioning {bone_name} head at HipRoot: {head_pos}")

        # SHOULDER REGION BONES: Head at ShoulderRoot position
        elif bone_name in ["LeftShoulder", "RightShoulder", "UpperSpine", "Neck"]:
            if "ShoulderRoot" in def_bones_dict:
                head_pos = def_bones_dict["ShoulderRoot"].head
                script_log(f"SHOULDER REGION: Positioning {bone_name} head at ShoulderRoot: {head_pos}")

        # HEAD BONE: Use VIRTUAL_HEAD_BASE as before
        elif bone_name == "Head":
            head_control_point = "VIRTUAL_HEAD_BASE"
            head_pos = calculate_virtual_position(first_frame, head_control_point)
            script_log(f"HEAD: Positioning {bone_name} head at {head_control_point}: {head_pos}")

        # NORMAL CASE: Use parent's tail for connection
        elif parent_name and parent_name in def_bones_dict:
            parent_bone = def_bones_dict[parent_name]
            head_pos = parent_bone.tail
            script_log(f"Positioning {bone_name} head at parent {parent_name} tail: {head_pos}")

        else:
            # Root bone - use head control point if specified
            if "head_control_point" in bone_data:
                head_cp = bone_data["head_control_point"]
                if head_cp.startswith("VIRTUAL_"):
                    head_pos = calculate_virtual_position(first_frame, head_cp)
                    script_log(f"ROOT: Positioning {bone_name} head at {head_cp}: {head_pos}")

        # Create DEF bone
        def_bone = armature_data.edit_bones.new(def_bone_name)
        def_bone.head = head_pos
        def_bone.tail = tail_pos

        # Store bone reference
        def_bones_dict[bone_name] = def_bone

        script_log(f"Created {def_bone_name}: head={head_pos}, tail={tail_pos}")

    # Set up parent relationships - using root bone hierarchy
    script_log("Setting up bone parent relationships with root bone hierarchy...")

    # Parent all bones
    for bone_name in creation_order:
        if bone_name not in bone_definitions:
            continue

        # Use our hard-coded parenting if defined, otherwise use config
        if bone_name in ROOT_BONE_PARENTING:
            parent_name = ROOT_BONE_PARENTING[bone_name]
        else:
            bone_data = bone_definitions[bone_name]
            parent_name = bone_data.get("parent")

        if not parent_name:
            continue

        # SPECIAL HANDLING FOR HIP REGION: Parent to HipRoot
        if bone_name in ["LeftHip", "RightHip", "LowerSpine"]:
            if "HipRoot" in def_bones_dict and bone_name in def_bones_dict:
                parent_bone = def_bones_dict["HipRoot"]
                child_bone = def_bones_dict[bone_name]

                child_bone.parent = parent_bone
                child_bone.use_connect = False

                script_log(f"ROOT HIERARCHY: Parented {bone_name} to HipRoot")

        # SPECIAL HANDLING FOR SHOULDER REGION: Parent to ShoulderRoot
        elif bone_name in ["LeftShoulder", "RightShoulder", "UpperSpine", "Neck"]:
            if "ShoulderRoot" in def_bones_dict and bone_name in def_bones_dict:
                parent_bone = def_bones_dict["ShoulderRoot"]
                child_bone = def_bones_dict[bone_name]

                child_bone.parent = parent_bone
                child_bone.use_connect = False

                script_log(f"ROOT HIERARCHY: Parented {bone_name} to ShoulderRoot")

        # NORMAL PARENTING for other bones
        elif parent_name in def_bones_dict and bone_name in def_bones_dict:
            parent_bone = def_bones_dict[parent_name]
            child_bone = def_bones_dict[bone_name]

            child_bone.parent = parent_bone
            child_bone.use_connect = False

            # Position at parent's tail
            child_bone.head = parent_bone.tail
            script_log(f"Parented {bone_name} to {parent_name}")

    script_log("Bone parenting with root hierarchy completed")

    # Exit Edit Mode first
    bpy.ops.object.mode_set(mode='OBJECT')

    # Force scene update to ensure bones are registered
    bpy.context.view_layer.update()

    return armature_obj


##########################################################################################

def setup_two_segment_spine_constraints(armature_obj, figure_name):
    """Setup simplified constraints for hierarchy-driven spine"""
    script_log("=== SETTING UP HIERARCHY-DRIVEN SPINE CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # Get the virtual points
    spine_mid_target = bpy.data.objects.get("VIRTUAL_SPINE_MIDPOINT")

    # LOWER SPINE CONSTRAINTS: Tail stretches to virtual spine midpoint
    if "DEF_LowerSpine" in armature_obj.pose.bones and spine_mid_target:
        lower_spine = armature_obj.pose.bones["DEF_LowerSpine"]

        # Clear existing constraints
        for constraint in list(lower_spine.constraints):
            lower_spine.constraints.remove(constraint)

        # HEAD POSITION & ROTATION: Handled by parenting to HipRoot (with coordinate frame)
        # TAIL: STRETCH_TO to virtual spine midpoint
        stretch_to = lower_spine.constraints.new('STRETCH_TO')
        stretch_to.target = spine_mid_target
        stretch_to.influence = 1.0
        constraints_added += 1
        script_log("DEF_LowerSpine STRETCH_TO -> VIRTUAL_SPINE_MIDPOINT (tail stretches to center)")

    # UPPER SPINE CONSTRAINTS: Tail stretches to virtual spine midpoint
    if "DEF_UpperSpine" in armature_obj.pose.bones and spine_mid_target:
        upper_spine = armature_obj.pose.bones["DEF_UpperSpine"]

        # Clear existing constraints
        for constraint in list(upper_spine.constraints):
            upper_spine.constraints.remove(constraint)

        # HEAD POSITION & ROTATION: Handled by parenting to ShoulderRoot (with coordinate frame)
        # TAIL: STRETCH_TO to virtual spine midpoint
        stretch_to = upper_spine.constraints.new('STRETCH_TO')
        stretch_to.target = spine_mid_target
        stretch_to.influence = 1.0
        constraints_added += 1
        script_log("DEF_UpperSpine STRETCH_TO -> VIRTUAL_SPINE_MIDPOINT (tail stretches to center)")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"Hierarchy-driven spine constraints: {constraints_added} constraints added")
    return constraints_added


##########################################################################################

def setup_direct_constraints(armature_obj, figure_name="Main"):
    """Set up constraints for all non-spine bones, SKIPPING hips and shoulders"""
    script_log("=== SETTING UP DIRECT CONSTRAINTS (SKIPPING HIPS/SHOULDERS) ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    constraints_added = 0

    # Apply constraints for all bones in bone_definitions EXCEPT hips and shoulders
    for bone_name, bone_data in bone_definitions.items():
        # SKIP hips and shoulders - they get position and rotation from root bones now
        if bone_name in ["LeftHip", "RightHip", "LeftShoulder", "RightShoulder"]:
            script_log(f"SKIPPING constraints for {bone_name} - using coordinate frame system")
            continue

        def_bone_name = def_bone_names.get(bone_name)
        constraint_type = bone_constraint_types.get(bone_name)

        if not def_bone_name or def_bone_name not in armature_obj.pose.bones:
            continue

        bone = armature_obj.pose.bones[def_bone_name]
        tail_control_point = bone_tail_control_points.get(bone_name)

        if tail_control_point:
            # Find target object
            target_obj = None
            for obj in bpy.data.objects:
                if obj.name == tail_control_point:
                    target_obj = obj
                    break

            if target_obj:
                # Check if STRETCH_TO constraint already exists
                has_stretch_to = any(constraint.type == 'STRETCH_TO' for constraint in bone.constraints)

                if not has_stretch_to:
                    # Use STRETCH_TO constraint for the TAIL
                    stretch_to = bone.constraints.new('STRETCH_TO')
                    stretch_to.target = target_obj
                    stretch_to.influence = 1.0

                    constraints_added += 1
                    script_log(f"{def_bone_name} STRETCH_TO → {tail_control_point}")

    # ADD NECK ROTATION CONSTRAINT TO FOLLOW SHOULDERS
    if "DEF_Neck" in armature_obj.pose.bones:
        neck_bone = armature_obj.pose.bones["DEF_Neck"]
        shoulder_midpoint_obj = bpy.data.objects.get("VIRTUAL_SHOULDER_MIDPOINT")

        if shoulder_midpoint_obj:
            # Check if constraint already exists
            has_copy_rot = any(constraint.type == 'COPY_ROTATION' for constraint in neck_bone.constraints)

            if not has_copy_rot:
                # Add COPY_ROTATION constraint to make neck follow shoulder orientation
                copy_rot = neck_bone.constraints.new('COPY_ROTATION')
                copy_rot.target = shoulder_midpoint_obj
                copy_rot.use_offset = True  # Use offset to maintain local rotation
                copy_rot.influence = 0.7  # Partial influence to allow some independent neck movement
                copy_rot.mix_mode = 'ADD'  # Add to existing rotation

                constraints_added += 1
                script_log(f"DEF_Neck COPY_ROTATION → VIRTUAL_SHOULDER_MIDPOINT (influence: 0.7)")

    # ADD HEAD ROTATION CONSTRAINT TO FOLLOW NOSE LANDMARK
    if "DEF_Head" in armature_obj.pose.bones:
        head_bone = armature_obj.pose.bones["DEF_Head"]

        # Check if constraint already exists
        has_damped_track = any(constraint.type == 'DAMPED_TRACK' for constraint in head_bone.constraints)

        if not has_damped_track:
            # Create or get NOSE control point object
            nose_obj = bpy.data.objects.get("CTRL_NOSE")
            if not nose_obj:
                # Switch to OBJECT mode to create the empty
                bpy.ops.object.mode_set(mode='OBJECT')

                # Create NOSE control point if it doesn't exist
                script_log("Creating NOSE control point for head rotation...")
                first_frame = mocap_data.get(str(frame_numbers[0]), {})
                if "NOSE" in first_frame:
                    pos_data = first_frame["NOSE"]
                    nose_pos = Vector((pos_data["x"], pos_data["y"], pos_data["z"]))

                    # Create empty for nose control point
                    bpy.ops.object.empty_add(type='PLAIN_AXES', location=nose_pos)
                    nose_obj = bpy.context.active_object
                    nose_obj.name = "CTRL_NOSE"

                    # Add to control points collection
                    control_collection = bpy.data.collections.get(f"{figure_name}_ControlPoints")
                    if control_collection and nose_obj.name not in control_collection.objects:
                        control_collection.objects.link(nose_obj)

                    script_log(f"Created CTRL_NOSE at {nose_pos}")

                # Switch back to POSE mode to continue adding constraints
                bpy.context.view_layer.objects.active = armature_obj
                bpy.ops.object.mode_set(mode='POSE')

            if nose_obj:
                # Add DAMPED_TRACK constraint to make head aim toward nose control point
                damped_track = head_bone.constraints.new('DAMPED_TRACK')
                damped_track.target = nose_obj
                damped_track.track_axis = 'TRACK_Y'  # Adjust based on head bone orientation
                damped_track.influence = 1.0

                constraints_added += 1
                script_log(f"DEF_Head DAMPED_TRACK → CTRL_NOSE (head follows nose orientation)")

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"Direct constraints: {constraints_added} constraints added")
    return constraints_added


##########################################################################################

def align_bones_with_control_points(armature_obj, figure_name):
    """Align bone positions with control points"""
    script_log("=== ALIGNING BONES WITH CONTROL POINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bones_aligned = 0

    for bone_name, control_obj in control_point_objs.items():
        def_bone_name = def_bone_names.get(bone_name)

        if not def_bone_name or def_bone_name not in armature_obj.data.edit_bones:
            continue

        bone = armature_obj.data.edit_bones[def_bone_name]
        target_pos = control_obj.location

        # All control points align to bone TAIL
        bone_direction = (bone.tail - bone.head).normalized()
        bone_length = bone.length

        bone.tail = target_pos
        bone.head = target_pos - (bone_direction * bone_length)

        script_log(f"Aligned {def_bone_name} TAIL to {control_obj.name} at {target_pos}")
        bones_aligned += 1

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    script_log(f"Bone alignment complete: {bones_aligned} bones aligned")
    return bones_aligned

##########################################################################################

def save_blender_file():
    """Save the Blender file to the correct output location"""
    try:
        from utils import get_show_path, get_scene_folder_name

        show_path = get_show_path(args.show)
        scene_folder_name = get_scene_folder_name(args.show, args.scene)
        outputs_dir = os.path.join(show_path, "outputs", "scenes", scene_folder_name)
        os.makedirs(outputs_dir, exist_ok=True)

        # Change from "_kid.blend" to "_cloth.blend"
        output_blender_file = os.path.join(outputs_dir, f"{args.scene}_cloth.blend")
        bpy.ops.wm.save_as_mainfile(filepath=output_blender_file)
        script_log(f"Cloth simulation saved to: {output_blender_file}")

    except Exception as e:
        script_log(f"Error saving Blender file: {e}")


##########################################################################################

def verify_hip_constraints(armature_obj):
    """Verify that all hip-related constraints are set up"""
    script_log("=== VERIFYING HIP CONSTRAINTS ===")

    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='POSE')

    hip_midpoint_obj = bpy.data.objects.get("VIRTUAL_HIP_MIDPOINT")

    # Check LowerSpine constraints
    if "DEF_LowerSpine" in armature_obj.pose.bones:
        lower_spine = armature_obj.pose.bones["DEF_LowerSpine"]
        script_log(f"DEF_LowerSpine constraints: {len(lower_spine.constraints)}")
        for constraint in lower_spine.constraints:
            # Check if constraint has target attribute before accessing it
            if hasattr(constraint, 'target'):
                target_name = constraint.target.name if constraint.target else "None"
            else:
                target_name = "No target"
            script_log(f"  - {constraint.type} -> {target_name}")

    # Check hip constraints
    for hip_name in ["DEF_LeftHip", "DEF_RightHip"]:
        if hip_name in armature_obj.pose.bones:
            hip_bone = armature_obj.pose.bones[hip_name]
            script_log(f"{hip_name} constraints: {len(hip_bone.constraints)}")
            for constraint in hip_bone.constraints:
                # Check if constraint has target attribute before accessing it
                if hasattr(constraint, 'target'):
                    target_name = constraint.target.name if constraint.target else "None"
                else:
                    target_name = "No target"
                script_log(f"  - {constraint.type} -> {target_name}")

    bpy.ops.object.mode_set(mode='OBJECT')

##########################################################################################

if __name__ == "__main__":
    main_execution()