import bpy
import bmesh
from mathutils import Vector


def setup_dynamic_vertex_weights(garment_obj, vertex_bundle_centers, joint_control_systems, script_log):
    """
    Set up dynamic vertex weights that update every frame based on control point positions
    """
    if not garment_obj or garment_obj.type != 'MESH':
        script_log(f"❌ Cannot setup dynamic weights for non-mesh object: {garment_obj}")
        return

    # Convert Vector objects to serializable lists for custom properties
    serializable_centers = {}
    for bundle_name, control_point_name in vertex_bundle_centers.items():
        serializable_centers[bundle_name] = control_point_name

    # Create a custom property to store vertex bundle configuration
    garment_obj["vertex_bundle_config"] = {
        "bundle_centers": serializable_centers,
        "last_positions": {},
        "update_frame": 0,
        "movement_threshold": 0.01,
        "update_frequency": 3
    }

    # Initialize last positions
    config = garment_obj["vertex_bundle_config"]
    for bundle_name, control_point_name in config["bundle_centers"].items():
        if control_point_name in joint_control_systems:
            rpy_empty = joint_control_systems[control_point_name]["rpy_empty"]
            config["last_positions"][bundle_name] = rpy_empty.location.copy()

    # Add frame change handler if not already present
    if update_garment_vertex_weights not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(update_garment_vertex_weights)

    script_log(f"✓ Set up dynamic vertex weights for {garment_obj.name} with {len(vertex_bundle_centers)} bundles")


def update_garment_vertex_weights(scene=None):
    """
    Update all garment vertex weights on frame change
    Called automatically by Blender on every frame change
    """
    # If called manually without scene parameter, get the current scene
    if scene is None:
        scene = bpy.context.scene

    frame = scene.frame_current

    for obj in bpy.data.objects:
        if "vertex_bundle_config" not in obj or obj.type != 'MESH':
            continue

        config = obj["vertex_bundle_config"]

        # Check update frequency
        if frame - config.get("update_frame", 0) < config.get("update_frequency", 3):
            continue

        # Check if control point positions have changed significantly
        positions_changed = check_positions_changed(config)

        if positions_changed:
            try:
                recalculate_vertex_weights(obj, config)
                config["update_frame"] = frame
                if frame % 30 == 0:  # Log every 30 frames for debugging
                    print(f"↻ Updated vertex weights for {obj.name} at frame {frame}")
            except Exception as e:
                print(f"❌ Error updating vertex weights for {obj.name}: {str(e)}")

def check_positions_changed(config):
    """
    Check if control point positions have changed significantly
    """
    current_positions = {}
    threshold = config.get("movement_threshold", 0.01)

    for bundle_name, control_point_name in config["bundle_centers"].items():
        # We need to access joint_control_systems through the scene
        joint_control_systems = get_joint_control_systems_from_scene()
        if not joint_control_systems:
            continue

        if control_point_name in joint_control_systems:
            rpy_empty = joint_control_systems[control_point_name]["rpy_empty"]
            current_positions[bundle_name] = rpy_empty.location.copy()
        else:
            print(f"⚠ Control point {control_point_name} not found in joint_control_systems")

    # Compare with last positions
    last_positions = config.get("last_positions", {})

    for bundle_name, current_pos in current_positions.items():
        last_pos = last_positions.get(bundle_name)
        if last_pos is None:
            config["last_positions"][bundle_name] = current_pos
            return True

        movement_distance = (current_pos - last_pos).length
        if movement_distance > threshold:
            config["last_positions"][bundle_name] = current_pos
            return True

    # Update last positions even if no change (for initial setup)
    config["last_positions"] = current_positions
    return False


def get_joint_control_systems_from_scene():
    """
    Retrieve joint_control_systems from scene custom properties
    """
    scene = bpy.context.scene
    if "joint_control_systems_data" in scene:
        # Reconstruct the joint_control_systems from scene data
        systems_data = scene["joint_control_systems_data"]
        joint_control_systems = {}

        for cp_name, system_data in systems_data.items():
            xyz_empty = bpy.data.objects.get(system_data["xyz_empty_name"])
            rpy_empty = bpy.data.objects.get(system_data["rpy_empty_name"])

            if xyz_empty and rpy_empty:
                joint_control_systems[cp_name] = {
                    "xyz_empty": xyz_empty,
                    "rpy_empty": rpy_empty
                }

        return joint_control_systems
    return None


def recalculate_vertex_weights(obj, config):
    """
    Recalculate vertex weights based on current control point positions
    """
    mesh = obj.data
    bundle_centers = config["bundle_centers"]

    joint_control_systems = get_joint_control_systems_from_scene()
    if not joint_control_systems:
        return

    # Use bmesh for efficient vertex operations
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    # Clear existing vertex groups (except those not in our bundle centers)
    groups_to_remove = []
    for vgroup in obj.vertex_groups:
        if vgroup.name in bundle_centers:
            groups_to_remove.append(vgroup)

    for vgroup in groups_to_remove:
        obj.vertex_groups.remove(vgroup)

    # Create new vertex groups with current positions
    for bundle_name, control_point_name in bundle_centers.items():
        if control_point_name not in joint_control_systems:
            continue

        rpy_empty = joint_control_systems[control_point_name]["rpy_empty"]
        center = rpy_empty.location
        radius = get_bundle_radius(control_point_name)

        if radius <= 0:
            print(f"⚠ Zero radius for {control_point_name}, using default 0.1")
            radius = 0.1

        # Create vertex group
        vgroup = obj.vertex_groups.new(name=bundle_name)

        # Calculate weights for all vertices
        for vert in bm.verts:
            # Convert to world coordinates
            vert_world = obj.matrix_world @ vert.co
            distance = (vert_world - center).length

            if distance <= radius:
                # Smooth weight falloff
                weight = 1.0 - (distance / radius)
                weight = max(0.0, min(1.0, weight))
                # Apply falloff curve (cubic for smoother transition)
                weight = weight * weight * (3 - 2 * weight)
                vgroup.add([vert.index], weight, 'REPLACE')

    # Clean up
    bm.free()
    mesh.update()


def get_bundle_radius(control_point_name):
    """
    Get the radius for a vertex bundle based on control point type
    """
    radius_configs = {
        "CTRL_LEFT_HIP": 0.15, "CTRL_RIGHT_HIP": 0.15,
        "CTRL_LEFT_KNEE": 0.12, "CTRL_RIGHT_KNEE": 0.12,
        "CTRL_LEFT_HEEL": 0.08, "CTRL_RIGHT_HEEL": 0.08,
        "CTRL_LEFT_SHOULDER": 0.12, "CTRL_RIGHT_SHOULDER": 0.12,
        "CTRL_LEFT_ELBOW": 0.10, "CTRL_RIGHT_ELBOW": 0.10,
        "CTRL_LEFT_WRIST": 0.06, "CTRL_RIGHT_WRIST": 0.06
    }

    return radius_configs.get(control_point_name, 0.1)


def setup_animated_empty_parents(joint_control_systems, script_log):
    """
    Make vertex bundle empties children of control point empties
    This ensures they move together in the hierarchy
    """
    for control_point_name, system in joint_control_systems.items():
        try:
            # Try to get the empties with flexible key names
            xyz_empty = None
            rpy_empty = None

            # Check for various possible key names
            if "xyz_empty" in system:
                xyz_empty = system["xyz_empty"]
            elif "control_empty" in system:
                xyz_empty = system["control_empty"]
            elif "empty" in system:
                xyz_empty = system["empty"]
            elif "obj" in system:
                xyz_empty = system["obj"]

            if "rpy_empty" in system:
                rpy_empty = system["rpy_empty"]
            elif "rotation_empty" in system:
                rpy_empty = system["rotation_empty"]
            elif "empty" in system and xyz_empty is None:
                rpy_empty = system["empty"]
            elif "obj" in system and xyz_empty is None:
                rpy_empty = system["obj"]

            # If we don't have both empties, skip this system
            if not xyz_empty or not rpy_empty:
                script_log(f"⚠ Skipping {control_point_name} - missing required empties")
                continue

            # Clear existing parenting and constraints
            rpy_empty.parent = None
            rpy_empty.constraints.clear()

            # Make RPY empty child of XYZ empty
            rpy_empty.parent = xyz_empty
            rpy_empty.matrix_parent_inverse = xyz_empty.matrix_world.inverted()

            # Set up copy location constraint (maintains separate rotation control)
            copy_loc = rpy_empty.constraints.new('COPY_LOCATION')
            copy_loc.target = xyz_empty
            copy_loc.use_x = True
            copy_loc.use_y = True
            copy_loc.use_z = True
            copy_loc.influence = 1.0

            script_log(f"✓ Set up animated empty parenting for {control_point_name}")

        except Exception as e:
            script_log(f"❌ Error setting up parenting for {control_point_name}: {str(e)}")

def get_opposite_control_point(control_point_name, joint_control_systems):
    """
    Get the opposite side control point for tracking
    """
    opposites = {
        "CTRL_LEFT_HIP": "CTRL_RIGHT_HIP",
        "CTRL_RIGHT_HIP": "CTRL_LEFT_HIP",
        "CTRL_LEFT_SHOULDER": "CTRL_RIGHT_SHOULDER",
        "CTRL_RIGHT_SHOULDER": "CTRL_LEFT_SHOULDER",
        "CTRL_LEFT_ELBOW": "CTRL_RIGHT_ELBOW",
        "CTRL_RIGHT_ELBOW": "CTRL_LEFT_ELBOW"
    }

    opposite_name = opposites.get(control_point_name, "CTRL_HEAD_TOP")
    if opposite_name in joint_control_systems:
        return joint_control_systems[opposite_name]["rpy_empty"]
    return None


def create_dynamic_vertex_bundle_empties(joint_control_systems, script_log):
    """
    Create vertex bundle empties that follow control points
    These are used for visual debugging and direct parenting
    """
    for control_point_name in joint_control_systems.keys():
        # Check if bundle empty already exists
        bundle_name = f"VB_{control_point_name}"
        if bundle_name in bpy.data.objects:
            continue

        # Create vertex bundle empty
        bundle_empty = bpy.data.objects.new(bundle_name, None)
        bpy.context.collection.objects.link(bundle_empty)
        bundle_empty.empty_display_size = 0.05
        bundle_empty.empty_display_type = 'SPHERE'
        bundle_empty.show_name = True
        bundle_empty.color = (0.0, 0.8, 0.2, 0.3)  # Green translucent

        # Get RPY empty and parent to it
        rpy_empty = joint_control_systems[control_point_name]["rpy_empty"]
        bundle_empty.parent = rpy_empty
        bundle_empty.matrix_parent_inverse = rpy_empty.matrix_world.inverted()

        # Copy location exactly
        bundle_empty.location = (0, 0, 0)  # Local to parent

        # Store in control system
        joint_control_systems[control_point_name]["bundle_empty"] = bundle_empty

        script_log(f"✓ Created dynamic vertex bundle empty for {control_point_name}")


def parent_garment_to_bundle_empties(garment_obj, bundle_mappings, joint_control_systems, script_log):
    """
    Parent garment to appropriate bundle empties for hierarchical movement
    """
    # Clear existing parenting
    garment_obj.parent = None

    # Create an empty specifically for this garment
    garment_parent = bpy.data.objects.new(f"Parent_{garment_obj.name}", None)
    bpy.context.collection.objects.link(garment_parent)
    garment_parent.empty_display_size = 0.1
    garment_parent.empty_display_type = 'CUBE'

    # Position garment parent at garment center
    garment_parent.location = garment_obj.location.copy()

    # Parent garment to its parent empty
    garment_obj.parent = garment_parent
    garment_obj.matrix_parent_inverse = garment_parent.matrix_world.inverted()

    # Set up constraints on garment parent to follow relevant bundle empties
    for bundle_type, control_point_name in bundle_mappings.items():
        if control_point_name in joint_control_systems:
            bundle_empty = joint_control_systems[control_point_name].get("bundle_empty")
            if bundle_empty:
                # Add copy location constraint with partial influence
                copy_loc = garment_parent.constraints.new('COPY_LOCATION')
                copy_loc.target = bundle_empty
                copy_loc.use_x = True
                copy_loc.use_y = True
                copy_loc.use_z = True
                copy_loc.influence = 0.3  # Partial influence for blending

    script_log(f"✓ Parented {garment_obj.name} to dynamic bundle empties")


def setup_driver_based_vertex_weights(garment_obj, vertex_bundle_config, joint_control_systems, script_log):
    """
    Set up Blender drivers for real-time vertex weight updates
    Most performant option for complex animations
    """
    if not garment_obj or garment_obj.type != 'MESH':
        return

    mesh = garment_obj.data

    # Ensure we have a vertex group modifier
    weight_modifier = None
    for mod in garment_obj.modifiers:
        if mod.type == 'VERTEX_WEIGHT_EDIT' and "Dynamic" in mod.name:
            weight_modifier = mod
            break

    if not weight_modifier:
        weight_modifier = garment_obj.modifiers.new("Dynamic_VertexWeights", 'VERTEX_WEIGHT_EDIT')
        weight_modifier.vertex_group = ""  # We'll handle groups individually

    # Set up drivers for each vertex bundle
    for bundle_name, control_point_name in vertex_bundle_config.items():
        if control_point_name not in joint_control_systems:
            continue

        # Create or get vertex group
        vgroup = garment_obj.vertex_groups.get(bundle_name)
        if not vgroup:
            vgroup = garment_obj.vertex_groups.new(name=bundle_name)

        # Set up drivers for this vertex group
        setup_vertex_group_drivers(garment_obj, vgroup, control_point_name, joint_control_systems, script_log)

    script_log(f"✓ Set up driver-based vertex weights for {garment_obj.name}")


def setup_vertex_group_drivers(obj, vertex_group, control_point_name, joint_control_systems, script_log):
    """
    Set up drivers for all vertices in a vertex group
    """
    mesh = obj.data
    rpy_empty = joint_control_systems[control_point_name]["rpy_empty"]
    radius = get_bundle_radius(control_point_name)

    # Use bmesh for efficient operations
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()

    # Clear existing weights
    vertex_group.remove(range(len(mesh.vertices)))

    # Set up drivers for each vertex
    driver_count = 0
    for vert in bm.verts:
        vert_world = obj.matrix_world @ vert.co
        initial_distance = (vert_world - rpy_empty.location).length

        if initial_distance <= radius:
            base_weight = max(0.0, 1.0 - (initial_distance / radius))
            base_weight = base_weight * base_weight  # Quadratic falloff

            if base_weight > 0.01:  # Only set up drivers for significant weights
                success = setup_single_vertex_driver(obj, vert.index, vertex_group.name,
                                                     control_point_name, base_weight, radius, joint_control_systems)
                if success:
                    driver_count += 1

    bm.free()
    script_log(f"  Set up {driver_count} vertex drivers for {vertex_group.name}")


def setup_single_vertex_driver(obj, vert_index, vgroup_name, control_point_name, base_weight, radius,
                               joint_control_systems):
    """
    Set up driver for individual vertex weight
    """
    try:
        # Get the vertex group
        vgroup = obj.vertex_groups.get(vgroup_name)
        if not vgroup:
            return False

        # Add driver to vertex group weight for this specific vertex
        driver = vgroup.id_data.animation_data_create().drivers
        fcurve = driver.find(f'vertex_groups["{vgroup_name}"].weight', index=vert_index)

        if not fcurve:
            fcurve = driver.driver_add(f'vertex_groups["{vgroup_name}"].weight', vert_index)

        # Configure driver
        fcurve.driver.type = 'SCRIPTED'
        fcurve.driver.use_self = True

        # Add control point position variable
        pos_var = fcurve.driver.variables.new()
        pos_var.name = "ctrl_pos"
        pos_var.type = 'TRANSFORMS'
        pos_var.targets[0].id = joint_control_systems[control_point_name]["rpy_empty"]
        pos_var.targets[0].transform_type = 'LOC_X'

        # Add vertex position variable (via custom property)
        vert_pos_prop = f"vert_{vert_index}_pos"
        if vert_pos_prop not in obj:
            mesh = obj.data
            obj[vert_pos_prop] = list(mesh.vertices[vert_index].co)

        vert_var = fcurve.driver.variables.new()
        vert_var.name = "vert_pos"
        vert_var.type = 'SINGLE_PROP'
        vert_var.targets[0].id = obj
        vert_var.targets[0].data_path = f'["{vert_pos_prop}"]'

        # Set driver expression
        fcurve.driver.expression = f"calculate_dynamic_weight(ctrl_pos, vert_pos, {base_weight}, {radius})"

        # Set driver namespace
        fcurve.driver.namespace = {
            'calculate_dynamic_weight': calculate_dynamic_weight,
            'Vector': Vector
        }

        return True

    except Exception as e:
        print(f"❌ Error setting up vertex driver: {str(e)}")
        return False


def calculate_dynamic_weight(ctrl_pos, vert_pos, base_weight, radius):
    """
    Custom function for driver expressions - calculates weight in real-time
    """
    try:
        # Convert to Vector objects
        if hasattr(ctrl_pos, '__len__') and len(ctrl_pos) >= 3:
            control_point = Vector((ctrl_pos[0], ctrl_pos[1], ctrl_pos[2]))
        else:
            control_point = Vector((0, 0, 0))

        if hasattr(vert_pos, '__len__') and len(vert_pos) >= 3:
            vertex_pos = Vector((vert_pos[0], vert_pos[1], vert_pos[2]))
        else:
            vertex_pos = Vector((0, 0, 0))

        # Calculate current distance
        current_distance = (vertex_pos - control_point).length

        # Calculate dynamic weight
        if current_distance <= radius:
            dynamic_weight = 1.0 - (current_distance / radius)
            dynamic_weight = max(0.0, min(1.0, dynamic_weight))
            # Apply smooth falloff
            dynamic_weight = dynamic_weight * dynamic_weight * (3 - 2 * dynamic_weight)
            return dynamic_weight * base_weight
        else:
            return 0.0

    except Exception as e:
        return base_weight  # Fallback to base weight


def setup_comprehensive_dynamic_system(joint_control_systems, script_log):
    """
    Set up all three systems for maximum robustness
    """
    script_log("=== SETTING UP COMPREHENSIVE DYNAMIC VERTEX SYSTEM ===")

    # 1. First ensure joint control systems are properly set up
    if not joint_control_systems:
        script_log("❌ joint_control_systems not initialized - cannot setup dynamic system")
        return False

    # 2. Store joint_control_systems in scene for frame handler access
    store_joint_control_systems_in_scene(joint_control_systems)

    # 3. Set up animated empty parents
    setup_animated_empty_parents(joint_control_systems, script_log)

    # 4. Create dynamic vertex bundle empties
    create_dynamic_vertex_bundle_empties(joint_control_systems, script_log)

    # 5. Register the frame change handler
    if update_garment_vertex_weights not in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.append(update_garment_vertex_weights)

    # 6. Set up performance optimization
    setup_performance_optimization(script_log)

    script_log("✓ Comprehensive dynamic vertex system initialized")
    return True


def store_joint_control_systems_in_scene(joint_control_systems):
    """
    Store joint_control_systems data in scene custom properties for frame handler access
    Handles different possible data structures
    """
    scene = bpy.context.scene
    systems_data = {}

    for cp_name, system in joint_control_systems.items():
        try:
            # Debug: Log what's actually in the system
            print(f"DEBUG: Processing {cp_name} - keys: {list(system.keys())}")

            # Try different possible key names for the empty objects
            xyz_empty = None
            rpy_empty = None

            # Check for various possible key names
            if "xyz_empty" in system:
                xyz_empty = system["xyz_empty"]
            elif "control_empty" in system:
                xyz_empty = system["control_empty"]
            elif "empty" in system:
                xyz_empty = system["empty"]
            elif "obj" in system:
                xyz_empty = system["obj"]

            if "rpy_empty" in system:
                rpy_empty = system["rpy_empty"]
            elif "rotation_empty" in system:
                rpy_empty = system["rotation_empty"]
            elif "empty" in system and xyz_empty is None:
                rpy_empty = system["empty"]
            elif "obj" in system and xyz_empty is None:
                rpy_empty = system["obj"]

            # If we found valid empties, store their names
            if xyz_empty and hasattr(xyz_empty, 'name'):
                systems_data[cp_name] = {
                    "xyz_empty_name": xyz_empty.name,
                    "rpy_empty_name": rpy_empty.name if rpy_empty and hasattr(rpy_empty, 'name') else xyz_empty.name
                }
            elif rpy_empty and hasattr(rpy_empty, 'name'):
                # If only rpy_empty exists, use it for both
                systems_data[cp_name] = {
                    "xyz_empty_name": rpy_empty.name,
                    "rpy_empty_name": rpy_empty.name
                }
            else:
                print(f"⚠ Could not find valid empties for {cp_name}")
                continue

        except Exception as e:
            print(f"❌ Error processing {cp_name}: {str(e)}")
            continue

    scene["joint_control_systems_data"] = systems_data
    print(f"✓ Stored {len(systems_data)} control systems in scene")


def setup_performance_optimization(script_log):
    """
    Configure performance settings for the dynamic system
    """
    # Set global performance settings
    bpy.context.scene.render.fps = 24  # Standard animation fps
    bpy.context.scene.frame_step = 1

    script_log("✓ Performance optimization configured")


def enhanced_garment_creation(garment_obj, vertex_bundle_mapping, joint_control_systems, script_log):
    """
    Enhanced garment creation with dynamic vertex systems
    """
    # 1. Set up dynamic vertex weights (frame-based updates)
    setup_dynamic_vertex_weights(garment_obj, vertex_bundle_mapping, joint_control_systems, script_log)

    # 2. Set up driver-based weights (real-time updates)
    setup_driver_based_vertex_weights(garment_obj, vertex_bundle_mapping, joint_control_systems, script_log)

    # 3. Parent to bundle empties (hierarchical movement)
    parent_garment_to_bundle_empties(garment_obj, vertex_bundle_mapping, joint_control_systems, script_log)

    return garment_obj


def cleanup_dynamic_systems(script_log):
    """
    Clean up dynamic systems when done
    """
    # Remove frame change handlers
    if update_garment_vertex_weights in bpy.app.handlers.frame_change_pre:
        bpy.app.handlers.frame_change_pre.remove(update_garment_vertex_weights)

    # Remove custom properties
    for obj in bpy.data.objects:
        for key in list(obj.keys()):
            if "vertex_bundle_config" in key or "vert_" in key:
                del obj[key]

    # Remove scene custom properties
    scene = bpy.context.scene
    if "joint_control_systems_data" in scene:
        del scene["joint_control_systems_data"]

    # Remove driver-based empties
    for obj in bpy.data.objects:
        if obj.name.startswith("VB_") or obj.name.startswith("Parent_"):
            bpy.data.objects.remove(obj, do_unlink=True)

    script_log("✓ Cleaned up dynamic vertex systems")


def get_dynamic_bundle_center(control_point_name, joint_control_systems, script_log):
    """
    Replacement for get_bundle_center that returns current dynamic position
    """
    if control_point_name in joint_control_systems:
        rpy_empty = joint_control_systems[control_point_name]["rpy_empty"]
        return rpy_empty.location.copy()
    else:
        script_log(f"⚠ Control point {control_point_name} not found for dynamic bundle center")
        return Vector((0, 0, 0))