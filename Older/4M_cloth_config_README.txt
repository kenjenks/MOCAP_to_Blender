4M CLOTH SYSTEM - ARTIST CONFIGURATION GUIDE
=============================================

TABLE OF CONTENTS
=================
1. QUICK START - Basic Setup
2. FIGURE & BONE SYSTEM - Character Foundation  
3. GARMENT CREATION - Clothing Setup
  3.1 Upper Body Garments
  3.2 Lower Body Garments 
  3.3 Accessories
4. MATERIALS & APPEARANCE - Visual Properties
5. CLOTH PHYSICS - Simulation Behavior
6. UV MAPPING & TEXTURING
7. ADVANCED SETTINGS - Fine-Tuning
8. ARTIST NOTES: REPLACING THE HEAD MODULE

INTRODUCTION
============
Welcome to the 4M Cloth System! This configuration file controls everything about 
your character's clothing and appearance. Each section below corresponds to a 
section in the 4M_cloth_config.json file.

---

1. QUICK START - BASIC SETUP
============================

SYSTEM SETTINGS:
- auto_generate_uvs: true/false - Automatically create UV maps for garments
- auto_seam_detection: true/false - Automatically detect optimal seam locations

UV SETTINGS:
- uv_scale_factor: 1.0 - Global scale for all UV maps
- uv_padding: 0.02 - Space between UV islands to prevent bleeding

FIGURE SCALING:
- x_squish_fraction: 1.0 - Scale character width (0.8 = 80% width)
- y_squish_fraction: 1.0 - Scale character depth  
- z_squish_fraction: 1.0 - Scale character height

---

2. FIGURE & BONE SYSTEM - CHARACTER FOUNDATION
==============================================

GEOMETRY SETTINGS:
- character_height: 1.6 - Overall character height in meters
- default_segments: 8 - Default cylinder segments for garments
- head_segments: 16 - Head mesh resolution
- head_ring_count: 12 - Head ring count for smoother geometry
- torso_height: 0.8 - Torso length
- torso_radius: 0.17 - Torso base radius

BASE RADII (Body Part Sizes):
- ankle: 0.08 - Ankle circumference base
- wrist: 0.06 - Wrist circumference base  
- shoulder: 0.12 - Shoulder joint size
- hip: 0.14 - Hip joint size
- head: 0.35 - Head base size

BONE DEFINITIONS:
Each bone has:
- parent: Parent bone name or null for root bones
- head_control_point: Virtual point for bone start position
- tail_control_point: Virtual point for bone end position  
- def_bone: Actual bone name used for deformation
- constraint: Type of constraint ("COPY_TO", "COPY_TAIL_ONLY")

---

3. GARMENT CREATION - CLOTHING SETUP
====================================

3.1 UPPER BODY GARMENTS:
------------------------

GARMENT_HEAD:
- enabled: true/false - Create head geometry
- type: "OBLATE_SPHEROID" - Head shape type
- scale: [0.35, 0.38, 0.4] - Head proportions (width, height, depth)
- neck_connection_radius: 0.12 - Where neck connects
- subdivision_levels: 2 - Mesh smoothness
- human_proportions: Fine-tune head shape (forehead, cheeks, jaw, etc.)
- facial_features: Add eye sockets, nose bridge

GARMENT_NECK:
- enabled: true/false - Create neck clothing
- type: "CYLINDRICAL" - Neck garment shape
- neck_height: 0.08 - Height of neck garment
- neck_diameter: 0.14 - Width of neck garment
- collar_style: "turtleneck" - Style of collar

COAT_TORSO:
- enabled: true/false - Create coat/jacket
- type: "CYLINDER" - Basic coat shape
- coat_length: "long" - "short" or "long" coat style
- torso_radius: 0.17 - Coat base size
- armhole_radius: 0.075 - Sleeve connection size
- vertical_segments: 8 - Coat vertical resolution
- radial_segments: 12 - Coat horizontal resolution

SLEEVES (left_sleeve / right_sleeve):
- enabled: true/false - Create sleeves
- type: "TAPERED_CYLINDER" - Sleeve shape that narrows
- bone_chain: ["DEF_LeftShoulder", "DEF_LeftUpperArm", "DEF_LeftForearm"] - Bones to follow
- diameter_start: 0.15 - Shoulder end diameter
- diameter_elbow: 0.12 - Elbow diameter  
- diameter_end: 0.08 - Wrist diameter
- segments: 32 - Sleeve mesh resolution

3.2 LOWER BODY GARMENTS:
------------------------

PANTS (left_pants / right_pants):
- enabled: true/false - Create pants
- type: "TAPERED_CYLINDER" - Pants shape
- bone_chain: ["DEF_LeftHip", "DEF_LeftThigh", "DEF_LeftShin"] - Leg bones to follow
- diameter_hip: 0.18 - Hip width
- diameter_knee: 0.14 - Knee width
- diameter_ankle: 0.12 - Ankle width
- segments: 8 - Pants mesh resolution

BOOTS (left_boot / right_boot):
- enabled: true/false - Create boots
- type: "COMPOUND" - Multi-part footwear
- bones: ["DEF_LeftShin", "DEF_LeftFoot"] - Bones for boot parts
- shaft_height: 0.15 - Boot shaft height
- foot_length: 0.12 - Foot coverage length
- foot_height: 0.06 - Foot coverage height
- segments: 8 - Boot mesh resolution

3.3 ACCESSORIES:
----------------

MITTENS (left_mitten / right_mitten):
- enabled: true/false - Create mittens
- type: "BOX_MITTEN" - Mitten shape
- hand_size: [0.1, 0.08, 0.04] - Hand coverage (length, width, height)
- thumb_size: [0.04, 0.03, 0.03] - Thumb size
- wrist_connection: true/false - Connect to sleeve

---

4. MATERIALS & APPEARANCE - VISUAL PROPERTIES
=============================================

Each garment has a MATERIAL section:
- color: [R, G, B, A] - Base color (0-1 values)
- roughness: 0.8 - Surface roughness (0.0 = mirror, 1.0 = matte)
- metallic: 0.0 - Metallic appearance (0.0 = non-metal, 1.0 = metal)
- specular: 0.3 - Specular intensity
- subsurface: 0.2 - Subsurface scattering for skin
- subsurface_color: [1.0, 0.8, 0.7, 1.0] - Subsurface color

HEAD_PRESETS:
Pre-defined head types with scale and skin tone:
- child: Small head with light skin
- adult: Medium head with medium skin  
- cartoon: Large head with stylized skin

COLLAR_PRESETS:
Pre-defined collar styles:
- simple_ring: Basic neck ring
- hood: Hood-style collar
- furry_cylinder: Fur-trimmed collar

---

5. CLOTH PHYSICS - SIMULATION BEHAVIOR
======================================

Each garment has CLOTH_SETTINGS:
- enabled: true/false - Enable cloth simulation
- quality: 6 - Simulation accuracy (higher = better, slower)
- mass: 0.3 - Fabric weight (affects drape and movement)
- tension_stiffness: 5.0 - Resistance to stretching
- compression_stiffness: 4.0 - Resistance to compression
- shear_stiffness: 3.0 - Resistance to shearing
- bending_stiffness: 0.5 - Resistance to bending
- air_damping: 1.0 - Air resistance effect
- time_scale: 0.8 - Simulation speed relative to animation

**JOINT VERTEX BUNDLES (Attachment Pinning):**
- **joint_bundle_falloff**: "quadratic" - Falloff curve for vertex weight from the joint (e.g., shoulder, wrist)
- **min_weight_threshold**: 0.05 - Minimum weight to be assigned to a vertex group
- **bundle_influence_scale**: 1.0 - Scale factor for the size of the joint influence area

SUBDIVISION SETTINGS (for sleeves):
- manual_cuts: 2 - Additional edge loops for detail
- subdiv_levels: 2 - Subdivision surface levels
- min_rings: 24 - Minimum rings around sleeve
- rings_per_meter: 50 - Ring density based on size

---

6. UV MAPPING & TEXTURING
=========================

Each garment has UV_PRESET:
- method: "SPHERICAL"/"CYLINDRICAL"/"BOX_PROJECT" - UV mapping technique
- seams: ["neck_opening", "back_head_seam"] - Where to place seams
- island_margin: 0.01 - Space between UV islands
- scale_to_bounds: true - Automatically scale to 0-1 UV space

---

7. ADVANCED SETTINGS - FINE-TUNING
==================================

GEOMETRY CALCULATION METHODS:
- knee_radius_calculation: "average" - How to calculate knee size
- elbow_radius_calculation: "average" - How to calculate elbow size

COAT LENGTH SETTINGS:
- short coats: Elastic waist with hip constraints
- long coats: Quarter separation for leg movement

HUMAN PROPORTIONS (Head):
- forehead_scale: 0.95 - Forehead size modifier
- cheek_scale: 1.1 - Cheek fullness
- jaw_scale: 0.9 - Jaw width
- chin_scale: 0.8 - Chin prominence
- back_head_scale: 1.05 - Back of head size
- top_head_scale: 0.95 - Top of head flatness

FACIAL FEATURES:
- eye_socket_depth: 0.08 - Depth of eye sockets
- nose_bridge_height: 0.05 - Nose bridge height
- enable_eye_sockets: true/false - Add eye socket detail
- enable_nose_bridge: true/false - Add nose bridge detail

---

8. ARTIST NOTES: REPLACING THE HEAD MODULE
==========================================

The 4M Cloth system is modular, offering the flexibility to replace the procedurally generated head with a custom, high-detail asset.

* **Disabling Generation (The Config):** To prevent the system from generating the default head mesh, you **must** set the `enabled` flag to **`false`** under the `GARMENT_HEAD` section in the `4M_cloth_config.json` file. All other parameters within the `GARMENT_HEAD` section are ignored when generation is disabled.

* **Asset Integration (The Blender File):**
    1. Import your custom head mesh into the generated `.blend` file.
    2. Position it accurately to match the character's neck attachment point.
    3. Ensure your custom head has an **Armature Modifier** pointing to the main character armature (`Armature_4K_kid`).
    4. You may then hide or delete the procedurally generated head mesh, which is named **`Head_Main_Mesh`**.

* **Critical Naming Convention (Do Not Change!):** The rest of the pipeline and other garments (neck, coat) rely on specific object and data names for attachment and deformation. To maintain full compatibility and functionality, you **must not** change the name of the following critical elements in the `.blend` file:
    * The main **Armature** object (e.g., `Armature_4K_kid`).
    * The **`garment_head` vertex groups**, as these are used by the neck garment for smooth blending.
    * Any **Nose landmark tracking objects** (empty objects or points) if they exist, as they are vital for facial rigging compatibility.
    * The generated head object name (`Head_Main_Mesh`), if you decide to keep it in the scene for backup or reference.

---

TROUBLESHOOTING
===============

COMMON ISSUES:

1. Garments not appearing:
   - Check "enabled": true for each garment
   - Verify bone names match your rig

2. Cloth simulation too stiff/loose:
   - Adjust mass (0.1-1.0 for light-heavy fabrics)
   - Modify stiffness values (1.0-30.0 range)

3. Garments clipping through body:
   - Increase collision_quality in cloth settings
   - Adjust self_distance_min for tighter fit

4. Poor deformation at joints (Pinning is too weak/small):
   - Increase **bundle_influence_scale** (size of the pinning zone)
   - Change **joint_bundle_falloff** to "smooth" for a gentler transition

5. UV mapping issues:
   - Change UV method based on garment shape
   - Adjust seam locations to hide seams

PERFORMANCE TIPS:
- Lower quality settings for faster simulation
- Reduce segments for less complex garments
- Disable subdivision when not needed
- Use lower resolution for distant characters

EXAMPLES
========

T-SHIRT AND JEANS SETUP:
"garment_head": {"enabled": true, ...},
"garment_neck": {"enabled": false, ...},
"left_sleeve": {"enabled": true, "material": {"color": [1,1,1,1]}},
"right_sleeve": {"enabled": true, "material": {"color": [1,1,1,1]}},
"left_pants": {"enabled": true, "material": {"color": [0.1,0.1,0.3,1]}},
"right_pants": {"enabled": true, "material": {"color": [0.1,0.1,0.3,1]}},

WINTER OUTFIT:
"garment_neck": {"enabled": true, "collar_style": "turtleneck"},
"coat_torso": {"enabled": true, "coat_length": "long"},
"left_mitten": {"enabled": true},
"right_mitten": {"enabled": true},
"left_boot": {"enabled": true},
"right_boot": {"enabled": true}

Remember to save your config file and restart the cloth generation to see changes!
