4M Cloth Garment System (4M_cloth and 4M_cloth_inner)

Introduction

The 4M_cloth program is part of the larger animation pipeline that creates and animates cloth garments on top of the 4K_kid human child model. This system generates modular, physics-enabled clothing that follows the character's skeleton and responds naturally to motion capture data. The system implements a sophisticated garment creation system that generates complete clothing items with proper vertex group assignment, armature deformation, and Blender cloth simulation. Each garment is carefully weighted to follow specific bones while maintaining realistic fabric behavior through physics simulation.

Technical Architecture

The 4M Cloth system uses a bone-driven garment system with realistic cloth physics:

- **Modular Garment System**: Individual garments for head, neck, sleeves, pants, boots, mittens, and coat torso
- **Bone-Driven Deformation**: Each garment follows specific bone chains with proper vertex weighting
- **Cloth Simulation**: Blender cloth modifiers with configurable fabric properties (stiffness, mass, damping)
- **Continuous Geometry**: Single-mesh sleeves and pants with smooth transitions between bone influences
- **Pinning Zone Weighting**: Advanced vertex weighting using spherical influence zones at the shoulder and wrist for firm cloth anchoring
- **Coordinate Frame Integration**: Leverages the same virtual control points and frame system as 4K_kid

Core functionality includes:

- Reading processed motion data from the apply_physics pipeline step
- Generating complete garment sets with configurable dimensions and materials
- Setting up bone-based vertex groups for proper armature deformation
- Applying cloth simulation with artist-controllable fabric properties
- Creating human-like head with facial features and nose landmark tracking
- Saving complete clothed character as .blend files ready for animation

IMPLEMENTATION

The 4M_cloth_inner.py script creates a complete garment system:

- **Human Head Generation**: Creates realistic head with facial features (eye sockets, nose bridge) and skin material
- **Continuous Sleeves**: Single-mesh sleeves weighted to upper arm (shoulder) and forearm (wrist) bones using **Pinning Zone Weighting**
- **Modular Pants**: Tapered cylindrical pants with hip, thigh, and shin segments
- **Compound Boots**: Multi-part boots with ankle spheres, shaft cylinders, and foot cylinders
- **Mittens**: Hand and thumb cylinders with configurable thumb angles
- **Coat Torso**: Cylindrical coat with optional quarter separation and shoulder coordination
- **Cloth Simulation**: Realistic fabric behavior with tension, compression, shear, and bending stiffness

The 4M_cloth.py script handles the pipeline integration:

- Reading configuration from 4M_cloth_config.json
- Setting up file paths using project utilities
- Copying base Blender file to output location
- Executing Blender in background mode with 4M_cloth_inner.py
- Launching Blender GUI with the final clothed character

Workflow

1. **Garment Creation** (4M_cloth.py):
   - Loads processed motion data from apply_physics step
   - Creates armature with coordinate frame system (same as 4K_kid)
   - Generates all enabled garments from configuration
   - Sets up cloth simulation and vertex groups
   - Saves as Scene-XXXX_cloth.blend

2. **Garment Configuration**:
   - Artists can enable/disable specific garments in config
   - Adjust dimensions, materials, and fabric properties
   - Modify subdivision levels and vertex weighting
   - Configure UV mapping presets

3. **Animation Ready**:
   - Clothed character maintains all 4K_kid animation capabilities
   - Garments deform naturally with character movement
   - Cloth simulation responds to character motion
   - Compatible with existing animation pipeline

Key Benefits:
- Modular garment system - mix and match clothing items
- Realistic cloth physics with artist-controllable parameters
- Proper deformation at joints using **Pinning Zone Weighting** for cloth anchoring
- Human-like head with facial features and landmark tracking
- Continuous geometry prevents separation during animation
- Full integration with existing 4K_kid animation pipeline

File Descriptions

=== 4M_cloth.py ===

This is the main entry point for clothed character creation. It is a standalone Python script that runs from the command line. Primary responsibilities:
* Reading configuration from 4M_cloth_config.json
* Setting up necessary file paths using utils.py
* Copying a base Blender file to the output location
* Executing Blender in background mode with 4M_cloth_inner.py
* Launching Blender GUI with the final clothed character

=== 4M_cloth_inner.py ===

This is the heart of the garment creation logic, executed by Blender's Python interpreter. Key functions include:
* Parsing command-line arguments passed from 4M_cloth.py
* Loading motion data from apply_physics JSON output
* Creating complete armature system (same as 4K_kid)
* Generating all configured garments:
  - create_head(): Human-like head with facial features
  - create_sleeve(): Continuous sleeves with **Pinning Zone Weighting** at shoulder and wrist
  - create_pants(): Modular pants with tapered cylinders
  - create_boot(): Compound boots with multiple parts
  - create_mitten(): Hand coverings with thumb articulation
  - create_coat(): Torso covering with cloth simulation
* Setting up vertex groups for bone deformation
* Applying cloth simulation modifiers
* Configuring materials and subdivision
* Saving the complete clothed character

=== 4M_cloth_config.json ===

This JSON file is the single point of truth for customizing garment appearance and behavior. Contains settings for:
* cloth_garments: Configuration for each garment type with enable/disable flags
* geometry_settings: Global geometry parameters (segments, radii, proportions)
* base_radii: Fundamental body part dimensions
* bone_definitions: Bone structure and control point mappings
* Material and cloth simulation properties for each garment

Garment Types Available:

- **garment_head**: Human-like head with skin material and facial features
- **garment_neck**: Stretchy neck covering with turtleneck style
- **left_sleeve/right_sleeve**: Continuous sleeves with **Pinning Zone Weighting** at shoulder and wrist
- **left_pants/right_pants**: Tapered pants with hip-to-ankle coverage
- **left_boot/right_boot**: Modular boots with shaft and foot components
- **left_mitten/right_mitten**: Hand coverings with articulated thumbs
- **coat_torso**: Outer coat with configurable length and quarter separation

See 4M_cloth_config_README.txt for more details.

=== Other Dependencies ===
* **utils.py**: Project utilities for file paths and logging
* **4K_kid_base.blend**: Base Blender file template

---

## Artist Notes: Replacing the Head Module

The 4M Cloth system is modular, and artists have the option to replace the ugly, lumpy procedurally generated head with a custom asset.

* **Location:** The generated head mesh object is named `Head_Main_Mesh` within the Blender file. This .blend file is created by the `create_head()` function when you have this config file setting:

  "cloth_garments": {
    "garment_head": {
      "replaceable_head": {
        "export_template": true,
        "export_path": "assets/hero_head_template.blend",
        "blend_file": "assets/hero_head.blend",

After you create a head, you should set "export_template": false so your newly created head doesn't get overwritten.

* **Procedure:** To replace the head, import your custom head mesh, position it correctly, and then edit or delete the `Head_Main_Mesh` object. Ensure your custom head has an **Armature Modifier** pointing to the main character armature.

* **Warning (Do Not Change):** The rest of the pipeline and the generated coat/neck garments rely on specific object and data names in the head object. To maintain full compatibility and functionality, you **must not** change the name of the following critical elements in the head's `.blend` file:
    * **The main Armature object** (e.g., `Armature_4K_kid`).
    * **The generated head object name** (`Head_Main_Mesh`) if you plan to keep it as a backup or reference.
    * **The nose landmark tracking objects** (if present), which are vital for facial rigging compatibility.
    * **The `garment_head` vertex groups**, as these are used by the neck garment for smooth transitions.

---

Technical Notes

- **Head System**: Generates human-like head with facial features, skin material, and nose landmark tracking
- **joint_vertex_bundles**: Uses quadratic falloff for creating high-weight areas at sleeve/pant attachment points
- **Continuous Geometry**: Single-mesh garments prevent separation during animation
- **Cloth Physics**: Configurable mass, stiffness, and damping for realistic fabric behavior
- **Modular Design**: Individual garments can be enabled/disabled independently
- **Bone Coordination**: Shoulder sphere groups enable coat-sleeve interaction
- **UV Presets**: Pre-configured UV mapping for each garment type

Garment Specific Features:

- **Sleeves**: Continuous mesh with **Pinning Zone Weighting** at shoulder/wrist, configurable taper
- **Pants**: Modular system with hip bridges, thigh cylinders, knee bridges, and shin cylinders
- **Boots**: Compound construction with ankle sphere, shaft cylinder, and elliptical foot
- **Mittens**: Hand and thumb cylinders with 45-degree thumb angle
- **Head**: Human proportions with eye sockets, nose bridge, and skin subsurface scattering
- **Coat**: Configurable length (short/long), quarter separation, and shoulder coordination

PROGRAM STATUS: PRODUCTION READY

The 4M Cloth pipeline successfully creates fully clothed characters with realistic garment deformation and cloth simulation. The modular architecture supports flexible clothing combinations while maintaining full compatibility with the existing 4K_kid animation system. The system handles complete garment generation, proper cloth simulation setup, and reliable deformation, making it suitable for production animation pipelines requiring clothed characters.
