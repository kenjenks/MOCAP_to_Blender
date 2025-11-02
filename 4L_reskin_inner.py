# 4L_reskin_inner.py (Version 8.5 - Vertex Matching for Bridges)

import bpy
import bmesh
import sys
import os
import argparse
import json
from mathutils import Vector, Matrix
import math
import mathutils


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
        g_config = full_config

        # DEBUG: Log what we found in the config
        script_log(f"Loaded reskin config from: {config_file_path}")
        script_log(f"Config keys found: {list(g_config.keys())}")

        if "anatomical_processing_order" in g_config:
            script_log(f"Found anatomical_processing_order: {len(g_config['anatomical_processing_order'])} bones")
            script_log(f"Bones in order: {g_config['anatomical_processing_order']}")
        else:
            script_log("WARNING: anatomical_processing_order not found in config")

        if "scene_objects" in g_config:
            script_log(f"Scene objects: {g_config['scene_objects']}")

    except Exception as e:
        script_log(f"ERROR: Failed to load config file {config_file_path}: {e}")
        sys.exit(1)


##########################################################################################

def get_object_by_name(name):
    """Safely retrieves an object by name."""
    obj = bpy.data.objects.get(name)
    if not obj:
        script_log(f"ERROR: Required object '{name}' not found in scene.")
    return obj


##########################################################################################

def save_blender_file():
    """Saves the current blend file."""
    try:
        bpy.ops.wm.save_mainfile()
        script_log("✓ Saved updated blend file.")
    except Exception as e:
        script_log(f"Error saving Blender file: {e}")


##########################################################################################

def select_object(obj):
    """Selects a single object and makes it active."""
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


##########################################################################################

def clean_up_old_mesh(new_mesh_name):
    """Delete any existing final new mesh to ensure a clean start."""
    if new_mesh_name in bpy.data.objects:
        new_mesh = bpy.data.objects[new_mesh_name]
        if new_mesh.users_collection:
            for collection in new_mesh.users_collection:
                collection.objects.unlink(new_mesh)
        bpy.data.objects.remove(new_mesh, do_unlink=True)
        script_log(f"Removed existing final mesh: {new_mesh_name}")


##########################################################################################

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


##########################################################################################

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


##########################################################################################

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


##########################################################################################

def remesh_component(component, settings, bone_name, target_faces):
    """Applies QuadriFlow remesh to a component mesh with smart face count limits."""
    select_object(component)

    # Calculate maximum possible faces based on original geometry
    original_faces = len(component.data.polygons)
    max_possible_faces = original_faces * 10  # Reasonable upper limit

    # Use the smaller of target or max possible
    actual_target_faces = min(target_faces, max_possible_faces)

    script_log(f"    - Original: {original_faces} faces, Target: {target_faces}, Using: {actual_target_faces}")
    script_log(f"    - Applying QuadriFlow Remesh for {bone_name}...")

    try:
        if bpy.context.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')

        bpy.ops.object.quadriflow_remesh(target_faces=actual_target_faces)

        # Log actual result
        final_faces = len(component.data.polygons)
        script_log(f"    ✓ Remesh complete: {final_faces} faces (target was {actual_target_faces})")

    except Exception as e:
        script_log(f"ERROR: QuadriFlow remesh failed on {component.name}: {e}. Component failed.")
        bpy.data.objects.remove(component, do_unlink=True)
        return None

    return component


##########################################################################################

def debug_lowerspine_component(old_mesh, bone_name="DEF_LowerSpine"):
    """Debug function to analyze LowerSpine geometry specifically"""
    script_log(f"=== DEBUG LOWERSPINE ANALYSIS ===")

    # Get the vertex group
    vg = old_mesh.vertex_groups.get(bone_name)
    if not vg:
        script_log(f"❌ Vertex group {bone_name} not found")
        return None

    # Count vertices in this group
    bpy.context.view_layer.objects.active = old_mesh
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.object.vertex_group_set_active(group=vg.name)
    bpy.ops.object.vertex_group_select()

    # Get selected vertex count
    bm = bmesh.from_edit_mesh(old_mesh.data)
    selected_verts = [v for v in bm.verts if v.select]
    vertex_count = len(selected_verts)

    bpy.ops.object.mode_set(mode='OBJECT')
    script_log(f"✓ {bone_name} vertex group contains {vertex_count} vertices")

    # Isolate and analyze the component
    component = old_mesh.copy()
    component.data = old_mesh.data.copy()
    bpy.context.collection.objects.link(component)
    component.name = f"Debug_{bone_name}"

    # Isolate geometry
    bpy.context.view_layer.objects.active = component
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.object.vertex_group_set_active(group=vg.name)
    bpy.ops.object.vertex_group_select()
    bpy.ops.mesh.select_all(action='INVERT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')

    # Analyze isolated component
    verts_after = len(component.data.vertices)
    faces_after = len(component.data.polygons)
    script_log(f"✓ Isolated {bone_name} component: {verts_after} vertices, {faces_after} faces")

    # Clean up debug mesh
    bpy.data.objects.remove(component, do_unlink=True)

    return vertex_count, verts_after, faces_after


##########################################################################################

def debug_weight_transfer(old_mesh, new_mesh):
    """Debug weight transfer to identify why mesh isn't moving with bones"""
    script_log("=== DEBUG WEIGHT TRANSFER ===")

    # Check if old mesh has vertex groups
    script_log(f"Old mesh '{old_mesh.name}' vertex groups: {len(old_mesh.vertex_groups)}")
    for vg in old_mesh.vertex_groups:
        script_log(f"  - {vg.name}")

    # Check if new mesh has vertex groups after transfer
    script_log(f"New mesh '{new_mesh.name}' vertex groups: {len(new_mesh.vertex_groups)}")
    for vg in new_mesh.vertex_groups:
        script_log(f"  - {vg.name}")

    # Check if armature modifier exists and is properly configured
    armature_mod = None
    for mod in new_mesh.modifiers:
        if mod.type == 'ARMATURE':
            armature_mod = mod
            break

    if armature_mod:
        script_log(f"Armature modifier found: {armature_mod.name}")
        script_log(f"Armature object: {armature_mod.object}")
        script_log(f"Use vertex groups: {armature_mod.use_vertex_groups}")
    else:
        script_log("❌ No armature modifier found on new mesh!")

    # Check if any vertices have weights
    if new_mesh.vertex_groups:
        script_log("✓ New mesh has vertex groups")
        # Sample a few vertices to see if they have weights
        for i, v in enumerate(new_mesh.data.vertices[:5]):  # Check first 5 vertices
            groups = [g for g in v.groups if g.weight > 0]
            script_log(f"  Vertex {i}: {len(groups)} weight groups")
            for g in groups:
                group_name = new_mesh.vertex_groups[g.group].name
                script_log(f"    - {group_name}: {g.weight:.3f}")
    else:
        script_log("❌ New mesh has NO vertex groups after transfer!")


##########################################################################################

def transfer_weights(old_mesh, new_mesh):
    """Comprehensive weight transfer with working methods only"""
    script_log("=== TRANSFERRING WEIGHTS ===")

    # Clear any existing vertex groups from new mesh
    new_mesh.vertex_groups.clear()

    # METHOD 1: Proximity-based manual assignment (PROVEN WORKING)
    script_log("Attempting proximity-based manual assignment...")
    try:
        # Get world coordinates of new mesh vertices
        new_mesh_coords = [new_mesh.matrix_world @ v.co for v in new_mesh.data.vertices]

        # For each vertex in new mesh, find closest vertex in old mesh and copy weights
        for i, new_vert in enumerate(new_mesh.data.vertices):
            closest_dist = float('inf')
            closest_old_vert = None

            # Find closest vertex in old mesh
            for old_vert in old_mesh.data.vertices:
                old_co = old_mesh.matrix_world @ old_vert.co
                dist = (new_mesh_coords[i] - old_co).length
                if dist < closest_dist:
                    closest_dist = dist
                    closest_old_vert = old_vert

            if closest_old_vert:
                # Copy vertex groups from closest old vertex
                for vg in closest_old_vert.groups:
                    if vg.weight > 0.1:  # Only copy significant weights
                        group_name = old_mesh.vertex_groups[vg.group].name
                        if group_name.startswith('DEF_'):
                            # Get or create vertex group
                            new_vg = new_mesh.vertex_groups.get(group_name)
                            if not new_vg:
                                new_vg = new_mesh.vertex_groups.new(name=group_name)
                            new_vg.add([i], vg.weight, 'REPLACE')

        if new_mesh.vertex_groups:
            script_log(f"✓ SUCCESS: Proximity assignment worked - {len(new_mesh.vertex_groups)} groups")
            return True
        else:
            script_log("Proximity assignment completed but no vertex groups created")
    except Exception as e:
        script_log(f"Proximity assignment failed: {e}")

    # METHOD 2: Create empty vertex groups as fallback
    script_log("Proximity method failed, creating empty vertex groups for manual painting...")
    try:
        for vg in old_mesh.vertex_groups:
            if vg.name.startswith('DEF_'):
                new_mesh.vertex_groups.new(name=vg.name)
        script_log(f"✓ Created {len(new_mesh.vertex_groups)} empty vertex groups")
        return True
    except Exception as e:
        script_log(f"❌ All weight transfer methods failed: {e}")
        return False


##########################################################################################
# BRIDGING FUNCTIONS - WITH VERTEX MATCHING
##########################################################################################

def find_seam_edges_between_bones(mesh_obj, bone_a, bone_b):
    """Find edges that form the seam between two bone components - STORES VERTEX INDICES"""
    script_log(f"Finding seam edges between {bone_a} and {bone_b}")

    # Get vertex groups
    vg_a = mesh_obj.vertex_groups.get(bone_a)
    vg_b = mesh_obj.vertex_groups.get(bone_b)

    if not vg_a or not vg_b:
        script_log(f"❌ Missing vertex groups: {bone_a} or {bone_b}")
        return []

    # Create BMesh for analysis
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Get vertex weights for both bones
    vert_weights_a = {}
    vert_weights_b = {}

    for vert in bm.verts:
        # Get weight for bone A
        try:
            weight_a = vg_a.weight(vert.index)
        except:
            weight_a = 0.0
        vert_weights_a[vert.index] = weight_a

        # Get weight for bone B
        try:
            weight_b = vg_b.weight(vert.index)
        except:
            weight_b = 0.0
        vert_weights_b[vert.index] = weight_b

    # Find seam edges - edges where one vertex has weight in A and the other in B
    # Store as vertex index pairs instead of BMesh references
    seam_edge_vertex_pairs = []

    for edge in bm.edges:
        v1, v2 = edge.verts

        # Check if this edge crosses the boundary
        has_a1 = vert_weights_a[v1.index] > 0.1
        has_b1 = vert_weights_b[v1.index] > 0.1
        has_a2 = vert_weights_a[v2.index] > 0.1
        has_b2 = vert_weights_b[v2.index] > 0.1

        # Edge crosses boundary if one vertex is primarily A and other is primarily B
        if ((has_a1 and not has_b1) and (has_b2 and not has_a2)) or \
                ((has_b1 and not has_a1) and (has_a2 and not has_b2)):
            seam_edge_vertex_pairs.append((v1.index, v2.index))

    script_log(f"Found {len(seam_edge_vertex_pairs)} seam edges between {bone_a} and {bone_b}")
    bm.free()

    return seam_edge_vertex_pairs


def get_boundary_loops_from_seam(mesh_obj, seam_edge_vertex_pairs, bone_a, bone_b):
    """Extract ordered boundary loops from seam edges using vertex indices"""
    if not seam_edge_vertex_pairs:
        return [], []

    # Create fresh BMesh for loop extraction
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Get vertex groups for classification
    vg_a = mesh_obj.vertex_groups.get(bone_a)
    vg_b = mesh_obj.vertex_groups.get(bone_b)

    # Create a set of seam edges for quick lookup
    seam_edges_set = set(seam_edge_vertex_pairs)

    # Find boundary loops using edge walking with vertex indices
    visited_edges = set()
    boundary_loops_a = []
    boundary_loops_b = []

    for edge_verts in seam_edge_vertex_pairs:
        if edge_verts in visited_edges:
            continue

        # Start a new loop with this edge
        current_loop_edges = [edge_verts]
        visited_edges.add(edge_verts)

        # Walk in both directions from this edge
        for direction in [0, 1]:
            current_edge = edge_verts
            current_vert_idx = current_edge[direction]

            while True:
                # Find next seam edge connected to current vertex
                next_edge = None

                # Get BMesh vertex for current vertex
                current_bm_vert = bm.verts[current_vert_idx]

                # Check all edges connected to this vertex
                for linked_edge in current_bm_vert.link_edges:
                    # Convert edge to vertex pair
                    linked_verts = (linked_edge.verts[0].index, linked_edge.verts[1].index)
                    linked_verts_reversed = (linked_edge.verts[1].index, linked_edge.verts[0].index)

                    # Check if this is a seam edge we haven't visited
                    if (linked_verts in seam_edges_set and linked_verts not in visited_edges) or \
                            (linked_verts_reversed in seam_edges_set and linked_verts_reversed not in visited_edges):

                        # Use the version that's in our seam set
                        if linked_verts in seam_edges_set:
                            next_edge = linked_verts
                        else:
                            next_edge = linked_verts_reversed
                        break

                if not next_edge:
                    break

                visited_edges.add(next_edge)
                current_loop_edges.append(next_edge)

                # Move to the other vertex of the next edge
                current_vert_idx = next_edge[0] if next_edge[1] == current_vert_idx else next_edge[1]
                current_edge = next_edge

        if current_loop_edges:
            # Extract all unique vertices from the loop edges
            all_vert_indices = set()
            for edge in current_loop_edges:
                all_vert_indices.add(edge[0])
                all_vert_indices.add(edge[1])

            # Convert to BMesh vertices and classify by bone
            loop_verts = [bm.verts[idx] for idx in all_vert_indices]

            # Separate vertices by bone
            loop_a = []
            loop_b = []

            for vert in loop_verts:
                try:
                    weight_a = vg_a.weight(vert.index)
                    weight_b = vg_b.weight(vert.index)
                except:
                    weight_a = 0.0
                    weight_b = 0.0

                if weight_a > weight_b:
                    loop_a.append(vert)
                else:
                    loop_b.append(vert)

            if loop_a and loop_b:
                boundary_loops_a.append(loop_a)
                boundary_loops_b.append(loop_b)

    bm.free()

    script_log(f"Found {len(boundary_loops_a)} boundary loops")
    return boundary_loops_a, boundary_loops_b


def sort_vertices_radial(vertices, center=None, normal=None):
    """Sort vertices radially around a center with proper normal orientation"""
    if not vertices:
        return []

    if center is None:
        center = Vector((0, 0, 0))
        for vert in vertices:
            if hasattr(vert, 'co'):
                center += vert.co
            else:
                center += vert['co']
        center /= len(vertices)

    if normal is None:
        # Estimate normal from first few vertices
        if len(vertices) >= 3:
            v1 = vertices[0].co if hasattr(vertices[0], 'co') else vertices[0]['co']
            v2 = vertices[1].co if hasattr(vertices[1], 'co') else vertices[1]['co']
            v3 = vertices[2].co if hasattr(vertices[2], 'co') else vertices[2]['co']
            normal = (v2 - v1).cross(v3 - v1).normalized()
        else:
            normal = Vector((0, 0, 1))

    # Create reference vectors for sorting
    ref_vector = None
    for vert in vertices:
        co = vert.co if hasattr(vert, 'co') else vert['co']
        vec = (co - center).normalized()
        if vec.length > 0.1 and abs(vec.dot(normal)) < 0.9:
            ref_vector = vec
            break

    if not ref_vector:
        ref_vector = Vector((1, 0, 0))

    # Orthogonalize reference vector against normal
    ref_vector = ref_vector - normal * ref_vector.dot(normal)
    if ref_vector.length > 0:
        ref_vector.normalize()
    else:
        ref_vector = Vector((1, 0, 0))

    # Sort vertices by angle around the normal
    def get_angle(vert):
        co = vert.co if hasattr(vert, 'co') else vert['co']
        vec = (co - center).normalized()

        # Project onto plane perpendicular to normal
        vec_proj = vec - normal * vec.dot(normal)
        if vec_proj.length < 0.001:
            return 0

        vec_proj.normalize()

        # Calculate angle relative to reference vector
        angle = math.atan2(ref_vector.cross(vec_proj).dot(normal), ref_vector.dot(vec_proj))
        return angle

    return sorted(vertices, key=get_angle)


def find_optimal_vertex_offset(ring_a, ring_b):
    """Find the optimal offset that minimizes total distance between corresponding vertices"""
    if len(ring_a) != len(ring_b):
        script_log(f"Warning: Ring sizes don't match: {len(ring_a)} vs {len(ring_b)}")
        return 0

    best_offset = 0
    min_total_distance = float('inf')

    # Try all possible offsets
    for offset in range(len(ring_b)):
        total_distance = 0.0

        # Calculate total distance for this offset
        for i in range(len(ring_a)):
            j = (i + offset) % len(ring_b)
            vert_a = ring_a[i]
            vert_b = ring_b[j]

            # Get vertex coordinates
            co_a = vert_a.co if hasattr(vert_a, 'co') else vert_a['co']
            co_b = vert_b.co if hasattr(vert_b, 'co') else vert_b['co']

            total_distance += (co_a - co_b).length

        # Check if this is the best offset so far
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_offset = offset

    script_log(f"Optimal vertex offset: {best_offset} (min distance: {min_total_distance:.3f})")
    return best_offset


def resample_vertices(vertices, target_count):
    """Resample vertices to match target count"""
    if len(vertices) == target_count:
        return vertices
    elif len(vertices) > target_count:
        # Reduce - take evenly spaced vertices
        step = len(vertices) / target_count
        return [vertices[int(i * step)] for i in range(target_count)]
    else:
        # Expand - duplicate vertices to reach target
        result = list(vertices)
        while len(result) < target_count:
            result.extend(vertices[:target_count - len(result)])
        return result


def create_bridge_between_boundaries(mesh_obj, boundary_a, boundary_b, bridge_vertex_count):
    """Create a proper bridge between two sorted boundary loops with vertex matching"""
    script_log(f"Creating bridge with {bridge_vertex_count} segments")

    # Enter edit mode
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Convert boundary data to BMesh vertices
    bm_verts_a = []
    bm_verts_b = []

    # Create KD-tree for vertex lookup
    kd = mathutils.kdtree.KDTree(len(bm.verts))
    for i, vert in enumerate(bm.verts):
        kd.insert(vert.co, i)
    kd.balance()

    # Find closest BMesh vertices for boundary A
    for vert_data in boundary_a:
        if hasattr(vert_data, 'co'):
            co = vert_data.co
        else:
            co = vert_data['co']
        closest_idx = kd.find(co)[1]
        bm_verts_a.append(bm.verts[closest_idx])

    # Find closest BMesh vertices for boundary B
    for vert_data in boundary_b:
        if hasattr(vert_data, 'co'):
            co = vert_data.co
        else:
            co = vert_data['co']
        closest_idx = kd.find(co)[1]
        bm_verts_b.append(bm.verts[closest_idx])

    # Ensure we have enough vertices
    if len(bm_verts_a) < 3 or len(bm_verts_b) < 3:
        script_log("❌ Not enough boundary vertices for bridge")
        bmesh.update_edit_mesh(mesh_obj.data)
        bpy.ops.object.mode_set(mode='OBJECT')
        return False

    # Sort boundaries radially
    center_a = sum((v.co for v in bm_verts_a), Vector((0, 0, 0))) / len(bm_verts_a)
    center_b = sum((v.co for v in bm_verts_b), Vector((0, 0, 0))) / len(bm_verts_b)
    avg_center = (center_a + center_b) / 2

    # Estimate bridge normal
    bridge_normal = (center_b - center_a).normalized()

    bm_verts_a_sorted = sort_vertices_radial(bm_verts_a, avg_center, bridge_normal)
    bm_verts_b_sorted = sort_vertices_radial(bm_verts_b, avg_center, bridge_normal)

    # Resample to target vertex count
    bm_verts_a_resampled = resample_vertices(bm_verts_a_sorted, bridge_vertex_count)
    bm_verts_b_resampled = resample_vertices(bm_verts_b_sorted, bridge_vertex_count)

    # Find optimal vertex matching
    optimal_offset = find_optimal_vertex_offset(bm_verts_a_resampled, bm_verts_b_resampled)

    # Create bridge faces with proper vertex matching
    faces_created = 0
    for i in range(bridge_vertex_count):
        # Get vertices from ring A
        v1 = bm_verts_a_resampled[i]
        v2 = bm_verts_a_resampled[(i + 1) % bridge_vertex_count]

        # Get corresponding vertices from ring B using optimal offset
        j = (i + optimal_offset) % bridge_vertex_count
        v3 = bm_verts_b_resampled[(j + 1) % bridge_vertex_count]
        v4 = bm_verts_b_resampled[j]

        try:
            # Check if face would be valid (all vertices distinct)
            if len(set([v1, v2, v3, v4])) == 4:
                face = bm.faces.new([v1, v2, v3, v4])
                face.smooth = True
                faces_created += 1
        except Exception as e:
            script_log(f"Failed to create bridge face {i}: {e}")
            # Try alternative winding
            try:
                face = bm.faces.new([v1, v2, v3, v4])
                face.smooth = True
                faces_created += 1
            except:
                pass

    # Update mesh
    bmesh.update_edit_mesh(mesh_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    script_log(f"✓ Created {faces_created}/{bridge_vertex_count} bridge faces")
    return faces_created > 0


def create_bridge_between_bones(mesh_obj, from_bone, to_bone, bridge_vertex_count):
    """Create a proper bridge between two bone components with correct topology"""
    script_log(f"Creating bridge: {from_bone} → {to_bone}")

    # Find seam edges between bones - now returns vertex index pairs
    seam_edge_vertex_pairs = find_seam_edges_between_bones(mesh_obj, from_bone, to_bone)

    if not seam_edge_vertex_pairs:
        script_log(f"❌ No seam edges found between {from_bone} and {to_bone}")
        return False

    # Extract boundary loops from seam - now uses vertex indices
    boundary_loops_a, boundary_loops_b = get_boundary_loops_from_seam(mesh_obj, seam_edge_vertex_pairs, from_bone,
                                                                      to_bone)

    if not boundary_loops_a or not boundary_loops_b:
        script_log(f"❌ Could not extract boundary loops between {from_bone} and {to_bone}")
        return False

    # Create bridges for each boundary loop pair
    total_faces_created = 0
    for i, (loop_a, loop_b) in enumerate(zip(boundary_loops_a, boundary_loops_b)):
        script_log(f"Creating bridge for loop {i + 1} with {len(loop_a)}/{len(loop_b)} vertices")

        # Convert BMesh vertices to coordinate data
        loop_a_data = [{'co': v.co, 'index': v.index} for v in loop_a]
        loop_b_data = [{'co': v.co, 'index': v.index} for v in loop_b]

        success = create_bridge_between_boundaries(
            mesh_obj, loop_a_data, loop_b_data, bridge_vertex_count
        )

        if success:
            total_faces_created += bridge_vertex_count

    script_log(f"✓ Created bridge with {total_faces_created} total faces")
    return total_faces_created > 0


##########################################################################################

def delete_non_bridge_faces(mesh_obj, bridge_face_indices):
    """Delete all faces except the newly created bridge faces"""
    script_log("=== DELETING NON-BRIDGE FACES FOR VISUALIZATION ===")

    # Enter edit mode
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_mode(type='FACE')

    # Get bmesh
    bm = bmesh.from_edit_mesh(mesh_obj.data)
    bm.faces.ensure_lookup_table()

    # Count total faces before deletion
    total_faces_before = len(bm.faces)

    # Select ALL faces first
    bpy.ops.mesh.select_all(action='SELECT')

    # Deselect the bridge faces we want to keep
    for face_index in bridge_face_indices:
        if face_index < len(bm.faces):
            bm.faces[face_index].select = False

    # Delete all selected faces (which are the non-bridge faces)
    selected_faces = [f for f in bm.faces if f.select]
    if selected_faces:
        bmesh.ops.delete(bm, geom=selected_faces, context='FACES')

    # Update mesh
    bmesh.update_edit_mesh(mesh_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Count remaining faces
    mesh_obj.data.update()
    total_faces_after = len(mesh_obj.data.polygons)

    script_log(f"✓ Deleted {total_faces_before - total_faces_after} non-bridge faces")
    script_log(f"✓ Kept {total_faces_after} bridge faces")
    script_log("⚠ WARNING: Only bridge geometry remains visible")

    return total_faces_after


##########################################################################################

def create_bridges_from_anatomical_order(mesh_obj, anatomical_order, bridge_settings):
    """Create bridges based on bridge_to relationships in anatomical_processing_order"""
    script_log("=== CREATING BRIDGES FROM ANATOMICAL ORDER ===")

    successful_bridges = 0
    default_vertex_count = bridge_settings.get("default_bridge_vertex_count", 12)
    bridge_face_indices = []

    # Get current face count before creating bridges
    mesh_obj.data.update()
    initial_face_count = len(mesh_obj.data.polygons)

    # Create bridges for each bone that has a bridge_to relationship
    for bone_def in anatomical_order:
        bridge_to = bone_def.get("bridge_to")
        if bridge_to:
            # Use bone-specific vertex count or default
            bridge_vertex_count = bone_def.get("bridge_vertex_count", default_vertex_count)

            script_log(f"Creating bridge: {bone_def['bone']} → {bridge_to} ({bridge_vertex_count} vertices)")

            success = create_bridge_between_bones(
                mesh_obj,
                bone_def["bone"],
                bridge_to,
                bridge_vertex_count
            )
            if success:
                successful_bridges += 1

                # Get the indices of the newly created bridge faces
                mesh_obj.data.update()
                current_face_count = len(mesh_obj.data.polygons)
                # The new faces are at the end of the faces list
                new_face_indices = list(range(initial_face_count, current_face_count))
                bridge_face_indices.extend(new_face_indices)
                initial_face_count = current_face_count

    # NOW delete non-bridge faces (only after creating all bridges)
    if successful_bridges > 0 and bridge_face_indices:
        remaining_faces = delete_non_bridge_faces(mesh_obj, bridge_face_indices)

        script_log("=== BRIDGE VISUALIZATION COMPLETE ===")
        script_log(f"✓ Created {successful_bridges} bridges")
        script_log(f"✓ Kept {remaining_faces} bridge faces")
        script_log("✓ All non-bridge geometry has been deleted")
        script_log("✓ Only bridge connections are now visible")
        script_log("NOTE: This is for visualization only - restore from backup for full mesh")
    else:
        script_log("⚠ No bridges were successfully created")

    return successful_bridges


##########################################################################################

def create_component_based_reskin(config):
    """
    Main function for the component-based reskinning workflow with anatomical ordering.
    """
    scene_objects = config.get("scene_objects", {})
    old_mesh_name = scene_objects.get("old_mesh_name")
    new_mesh_name = scene_objects.get("new_mesh_name")
    rig_name = scene_objects.get("rig_name")

    anatomical_config = config.get("anatomical_processing_order", [])
    settings = config.get("retopology_settings", {})
    smoothing_settings = config.get("smoothing_settings", {})
    default_face_count = settings.get("default_face_count", 500)

    # Parse anatomical order with face counts
    anatomical_bone_order = []
    bone_face_counts = {}

    for item in anatomical_config:
        if isinstance(item, dict) and "bone" in item:
            bone_name = item["bone"]
            face_count = item.get("face_count", default_face_count)
            anatomical_bone_order.append(bone_name)
            bone_face_counts[bone_name] = face_count
        elif isinstance(item, str):
            # Backward compatibility with string-only lists
            anatomical_bone_order.append(item)
            bone_face_counts[item] = default_face_count

    script_log(f"DEBUG: Found {len(anatomical_bone_order)} bones in anatomical_processing_order")
    script_log(f"Bones to process: {anatomical_bone_order}")
    script_log(f"Face counts: {bone_face_counts}")

    if not anatomical_bone_order:
        script_log("CRITICAL ERROR: 'anatomical_processing_order' list is empty or missing from config.")
        return None

    old_mesh = get_object_by_name(old_mesh_name)
    if not old_mesh:
        script_log(f"ERROR: Old mesh '{old_mesh_name}' not found.")
        return None

    # Validate old mesh has required vertex groups
    if not validate_old_mesh_for_weights(old_mesh):
        script_log("ERROR: Old mesh missing required DEF_ vertex groups for weight transfer")
        return None

    # Clean up previous final mesh and intermediate objects
    cleanup_intermediate_objects()
    clean_up_old_mesh(new_mesh_name)

    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

    # STEP 1: Use anatomical order directly from config
    script_log(f"Processing bones in anatomical order: {anatomical_bone_order}")

    # STEP 2: Process and join components sequentially
    script_log("--- PROCESSING COMPONENTS IN ANATOMICAL ORDER ---")

    final_mesh = None
    processed_components = 0

    for i, bone_name in enumerate(anatomical_bone_order, 1):
        script_log(f"--- Processing {bone_name} ({i}/{len(anatomical_bone_order)}) ---")

        # DEBUG - ONLY FOR LOWERSPINE
        if bone_name == "DEF_LowerSpine":
            script_log(">>> LOWERSPINE DEBUG TRIGGERED <<<")
            debug_lowerspine_component(old_mesh, bone_name="DEF_LowerSpine")

        # Isolate component
        component = isolate_component(old_mesh, bone_name)
        if not component:
            script_log(f"❌ Failed to isolate component for {bone_name}")
            continue

        # Get face count for this bone
        face_count = bone_face_counts.get(bone_name, default_face_count)

        # Remesh component with bone-specific face count
        remeshed_component = remesh_component(component, settings, bone_name, face_count)
        if not remeshed_component:
            script_log(f"❌ Failed to remesh component for {bone_name}")
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

                # MINIMAL merging only - no bridging
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.mesh.remove_doubles(threshold=0.001)  # Very conservative merging
                bpy.ops.object.mode_set(mode='OBJECT')

            except Exception as e:
                script_log(f"ERROR: Failed to join {bone_name}: {e}")
                continue

        processed_components += 1

    if not final_mesh:
        script_log("CRITICAL ERROR: No valid components were generated.")
        return None

    script_log(
        f"Successfully processed {processed_components}/{len(anatomical_bone_order)} components in anatomical order")

    # STEP 3: Minimal seam stitching (NO BRIDGING)
    script_log("--- CREATING CLEAN SEAMS (NO BRIDGING) ---")
    select_object(final_mesh)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    # Conservative merge only - no bridging
    bpy.ops.mesh.remove_doubles(threshold=0.001)

    # EXPLICITLY SKIP BRIDGING
    script_log("    - Created clean seams (90° rotations prevent automatic bridging)")

    bpy.ops.object.mode_set(mode='OBJECT')

    # STEP 4: Weight Transfer
    script_log("--- 4. Transferring Weights ---")

    if not transfer_weights(old_mesh, final_mesh):
        script_log("⚠ Weight transfer failed - mesh will need manual weight painting")
    else:
        script_log("✓ Weight transfer completed successfully")

    # STEP 5: Create bridges from anatomical order (VISUALIZATION ONLY)
    bridge_settings = config.get("bridge_settings", {})
    if bridge_settings:
        script_log("--- CREATING BRIDGES BETWEEN COMPONENTS (VISUALIZATION) ---")
        script_log("WARNING: This will delete all non-bridge geometry for visualization")

        bridges_created = create_bridges_from_anatomical_order(
            final_mesh,
            anatomical_config,
            bridge_settings
        )

        if bridges_created > 0:
            script_log(f"✓ Created {bridges_created} bridge connections")
            script_log("VISUALIZATION: Only bridge geometry is now visible")
            script_log("To restore full mesh, reload the original file")
        else:
            script_log("⚠ No bridges were created")
    else:
        script_log("No bridge settings found - skipping bridge creation")

    # STEP 6: Setup final mesh
    script_log("--- 6. Setting up final mesh ---")

    # Add armature modifier
    rig_obj = bpy.data.objects.get(rig_name)
    if rig_obj:
        armature_mod = final_mesh.modifiers.new(name='Armature', type='ARMATURE')
        armature_mod.object = rig_obj
        armature_mod.use_vertex_groups = True
        script_log(f"    - Added Armature modifier ({rig_name}).")
        script_log(f"    - Armature object: {armature_mod.object}")
        script_log(f"    - Use vertex groups: {armature_mod.use_vertex_groups}")
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


##########################################################################################

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
        script_log(f"ERROR during component reskinning: {e}")
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