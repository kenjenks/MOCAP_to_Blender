3_APPLYPHYSICS.PY - PHYSICS-BASED MOTION CAPTURE TRANSFORMATION (Version 3.4)
---

OVERVIEW
---
3_ApplyPhysics.py transforms MediaPipe pose data by applying physics-based constraints, coordinate system transformations, and biomechanical corrections.

You can edit many parameters in apply_physics_config.json that affect the program. 

KEY FEATURES
---
* Coordinate System Transformation
* Depth Adjustment System (New in Version 3.1+)
* Motion State Detection  
* Height Adjustment & Grounding
* Biomechanical Constraints
* Shoulder Stabilization (New in Version 3.4)
* Head Stabilization (New in Version 3.4)

COORDINATE SYSTEM TRANSFORMATION
---
Converts MediaPipe coordinate system to Blender coordinate system:
* MediaPipe X (horizontal right) -> Blender X (horizontal right)
* MediaPipe Y (vertical down) -> Blender Z (vertical up) - INVERTED
* MediaPipe Z (depth forward) -> Blender Y (depth forward)

Transformation formula: (X, Y, Z) -> (X, Z, -Y)

MOTION STATE DETECTION
---
Classifies movement into states:
* STANDING
* WALKING
* RUNNING
* JUMPING
* LANDING
* FALLING
* FEET_OFF_FLOOR (for handstands, rolls)
* TRANSITION

HEIGHT ADJUSTMENT & GROUNDING
---
* Ensures feet are properly grounded at Z=0
* Smooths jump trajectories using polynomial fitting
* Maintains consistent height across frames

BIOMECHANICAL CONSTRAINTS
---
* Applies realistic human joint limits
* Maintains proportional limb lengths
* Prevents physically impossible poses

DEPTH ADJUSTMENT SYSTEM
---
Automatically detects and corrects exaggerated depth estimates from 2D-to-3D inference.

Detection Strategies:
* Biomechanical Ratios
* Depth Variation Analysis  
* Planar Motion Detection

SHOULDER STABILIZATION SYSTEM (NEW IN V3.4)
---
Detects and corrects upper/lower body scaling mismatches by maintaining proper proportions between shoulders and hips.

Detection Triggers:
* Shoulder-Hip Width Ratio Mismatch - shoulders disproportionately wide/narrow vs hips
* Torso Aspect Ratio Mismatch - unnatural torso proportions (too squat or elongated)

Correction Mechanism:
* Smooth vertical corrections applied to shoulder positions
* Connected joint propagation (neck, head, elbows) to maintain limb proportions
* Configurable correction strength and tolerance thresholds

HEAD STABILIZATION SYSTEM (NEW IN V3.4)
---
Corrects unnatural head positioning using body-relative coordinate system in Blender space (+Z=up, +Y=forward).

Detection Triggers:
* Head Too Low - head too close to shoulders vertically
* Head Too Far Back - head behind shoulder plane  
* Head Squished - insufficient distance between HEAD_TOP and NOSE
* Nose Not Forward Enough - face not properly oriented forward

Body-Relative Coordinate System:
* Body "Up" direction calculated from shoulder_mid - hip_mid
* Body "Forward" direction calculated from cross(shoulder_line, body_up)
* Adapts to bending, leaning, or twisting poses

CONFIGURATION
---
Complete configuration structure in apply_physics_config.json:

===
{
    "general": {
        "epsilon": 1e-6,
        "log_file_path": "Log3.txt",
        "desired_up_vector": [0, 0, 1],
        "desired_forward_vector": [0, 1, 0]
    },
    "debug_flags": {
        "turn_off_z_up_transformation": false,
        "turn_off_forward_transformation": true,
        "turn_off_shoulder_shift": true,
        "turn_off_hip_shift": true,
        "turn_off_biomechanical_constraints": true,
        "turn_off_hip_heel_floor_shift": true,
        "turn_off_depth_adjustment": false,
        "turn_off_shoulder_stabilization": true,
        "turn_off_head_stabilization": false
    },
    "shoulder_stabilization": {
        "expected_shoulder_hip_ratio": 1.2,
        "expected_torso_aspect_ratio": 1.4,
        "target_vertical_offset": 0.4,
        "correction_factor": 0.3,
        "proportion_tolerance": 0.3,
        "enable_shoulder_stabilization": true
    },
    "head_stabilization": {
        "enable_head_stabilization": true,
        "distance_controls": {
            "min_head_shoulder_distance": 0.15,
            "target_head_shoulder_distance": 0.25,
            "distance_correction_strength": 0.3
        },
        "height_controls": {
            "min_head_height": 0.12,
            "target_head_height": 0.18,
            "height_correction_strength": 0.3
        },
        "forward_position_controls": {
            "min_head_forward_offset": 0.06,
            "target_head_forward_offset": 0.10,
            "min_nose_forward_offset": 0.10,
            "target_nose_forward_offset": 0.15,
            "nose_forward_advantage": 0.02,
            "forward_correction_strength": 0.3,
            "nose_forward_priority": 0.8
        },
        "body_plane_alignment": {
            "max_natural_plane_distance": 0.08,
            "plane_correction_strength": 0.2,
            "allow_natural_nodding": true,
            "nodding_tolerance_factor": 1.5,
            "side_to_side_tolerance_factor": 0.8
        },
        "proportional_corrections": {
            "nose_correction_ratio": 0.7,
            "other_landmarks_correction_ratio": 0.6,
            "ear_correction_ratio": 0.5,
            "eye_correction_ratio": 0.8
        },
        "smoothing_controls": {
            "correction_smoothing_factor": 0.4,
            "history_window_size": 15,
            "min_frames_for_trend": 5
        }
    },
    "motion_detection": {
        "feet_off_ground_threshold": 0.1,
        "upward_velocity_threshold": 0.05,
        "downward_velocity_threshold": -0.05,
        "running_velocity_threshold": 0.2,
        "walking_velocity_threshold": 0.05,
        "state_transition_window_size": 5,
        "jump_detection_min_frames": 10
    },
    "height_adjustment": {
        "desired_height": 1.6,
        "landing_blend_factor": 0.7,
        "jump_trajectory_window": 5,
        "ground_history_size": 5,
        "z_offset_history_size": 30,
        "polynomial_degree": 2
    },
    "depth_adjustment": {
        "max_allowed_depth_ratio": 0.5,
        "force_minimal_depth_variation": false,
        "manual_depth_scale_factor": null,
        "sensitivity": {
            "depth_variation_trigger_ratio": 1.2,
            "planar_motion_variance_ratio": 0.3,
            "biomechanical_discrepancy_threshold": 1.1,
            "min_scale_factor": 0.05,
            "max_scale_factor": 1.0
        }
    },
    "biomechanical_constraints": {
        "head_roll_limit_degrees": 45.0,
        "head_nod_limit_degrees": 45.0,
        "head_y_translation_limit_percent": 2.0,
        "head_roll_sensitivity": 0.5,
        "head_nod_sensitivity": 5.0
    }
}
===

Depth Adjustment Configuration Examples:

For Normal Motion:
===
"depth_adjustment": {
    "max_allowed_depth_ratio": 0.5,
    "force_minimal_depth_variation": false,
    "manual_depth_scale_factor": null,
    "sensitivity": {
        "depth_variation_trigger_ratio": 1.2,
        "planar_motion_variance_ratio": 0.3,
        "biomechanical_discrepancy_threshold": 1.1,
        "min_scale_factor": 0.05,
        "max_scale_factor": 1.0
    }
}
===

For Planar Motion:
===
"depth_adjustment": {
    "max_allowed_depth_ratio": 0.5,
    "force_minimal_depth_variation": false,
    "manual_depth_scale_factor": null,
    "sensitivity": {
        "depth_variation_trigger_ratio": 1.0,
        "planar_motion_variance_ratio": 0.1,
        "biomechanical_discrepancy_threshold": 1.05,
        "min_scale_factor": 0.05,
        "max_scale_factor": 0.5
    }
}
===

CONFIGURATION FLAGS
---
All transformations can be individually enabled/disabled via debug_flags in config:

===
"debug_flags": {
    "turn_off_z_up_transformation": false,
    "turn_off_forward_transformation": true,
    "turn_off_shoulder_shift": true,
    "turn_off_hip_shift": true,
    "turn_off_biomechanical_constraints": true,
    "turn_off_hip_heel_floor_shift": true,
    "turn_off_depth_adjustment": false,
    "turn_off_shoulder_stabilization": true,
    "turn_off_head_stabilization": false
}
===

PROCESSING PIPELINE
---
1. Z-Up Transformation
2. Shoulder Stabilization (if enabled)
3. Depth Adjustment (if enabled)
4. Head Stabilization (if enabled)
5. Motion State Detection
6. Height Grounding
7. Global Scaling
8. Centering
9. Biomechanical Constraints
10. Forward Orientation (if enabled)

SHOULDER STABILIZATION DETAILS
---

## When Shoulder Stabilization is Triggered

Shoulder stabilization activates when **ANY** of these biomechanical proportion violations are detected:

### 1. **Shoulder-Hip Width Ratio Mismatch**
- **Trigger**: Shoulders are disproportionately wide or narrow compared to hips
- **Detection**: `abs(shoulder_hip_ratio - EXPECTED_SHOULDER_HIP_RATIO) > TOLERANCE`
- **Expected Ratio**: 1.2 (shoulders should be ~20% wider than hips)
- **Tolerance**: 0.3 (30% deviation allowed)
- **What it means**: Upper body scaling doesn't match lower body scaling
- **Correction**: Adjusts shoulder vertical position to restore proportions

### 2. **Torso Aspect Ratio Mismatch**
- **Trigger**: Torso proportions are unnatural (too squat or elongated)
- **Detection**: `abs(torso_aspect_ratio - EXPECTED_TORSO_ASPECT) > TOLERANCE`
- **Expected Ratio**: 1.4 (shoulder width should be ~40% larger than torso height)
- **Tolerance**: 0.3 (30% deviation allowed)
- **What it means**: Torso shape is biomechanically improbable
- **Correction**: Adjusts shoulder height to restore natural torso shape

## Key Metrics Calculated

### Shoulder-Hip Relationship Analysis
For each frame, the system calculates:
- **Shoulder midpoint**: Average of LEFT_SHOULDER and RIGHT_SHOULDER
- **Hip midpoint**: Average of LEFT_HIP and RIGHT_HIP  
- **Shoulder width**: 3D distance between left and right shoulders
- **Hip width**: 3D distance between left and right hips
- **Torso height**: Vertical distance between shoulder and hip midpoints
- **Vertical offset**: Shoulder height relative to hips

### Critical Ratios
- **Shoulder-Hip Ratio** = `shoulder_width / hip_width`
- **Torso Aspect Ratio** = `shoulder_width / torso_height`

## Correction Mechanism

When issues are detected, the system applies **smooth vertical corrections**:

### Vertical Offset Correction
- **Target**: `TARGET_VERTICAL_OFFSET` (default: 0.4m)
- **Strength**: `CORRECTION_FACTOR` (default: 0.3 = 30% per frame)
- **Applied to**: Both LEFT_SHOULDER and RIGHT_SHOULDER landmarks
- **Connected joints**: Neck, head, and elbows receive proportional corrections

### Connected Joint Propagation
To maintain natural body proportions, corrections propagate to:
- **Neck/Head joints** (NOSE, EYES, EARS): 80% of shoulder correction
- **Elbow joints**: 50% of shoulder correction  
- This prevents unnatural stretching between connected body parts

HEAD STABILIZATION DETAILS
---

## When Head Stabilization is Triggered

Head stabilization activates when **ANY** of these conditions are detected:

### 1. **Head Too Low** (Vertical Position)
- **Trigger**: Head is too close to shoulders vertically
- **Detection**: `head_vertical < MIN_HEAD_VERTICAL` (default: 0.10m)
- **What it means**: Head is slouching or compressed downward relative to shoulders
- **Correction**: Moves head upward along body's "up" direction

### 2. **Head Too Far Back** (Forward Position)  
- **Trigger**: Head is behind the shoulder plane
- **Detection**: `head_forward < MIN_HEAD_FORWARD` (default: 0.08m)
- **What it means**: Head is retracted backward instead of being in front of shoulders
- **Correction**: Moves head forward along body's "forward" direction

### 3. **Head Squished** (Head Height)
- **Trigger**: Distance between HEAD_TOP and NOSE is too small
- **Detection**: `head_height < MIN_HEAD_HEIGHT` (default: 0.15m)
- **What it means**: Head landmarks are compressed vertically (unnatural head shape)
- **Correction**: Increases vertical distance between head_top and nose

### 4. **Nose Not Forward Enough**
- **Trigger**: Nose is not sufficiently in front of head_top
- **Detection**: `nose_forward < head_forward + NOSE_FORWARD_ADVANTAGE` (default: 0.02m advantage)
- **What it means**: Face is not properly oriented forward relative to head
- **Correction**: Moves nose further forward than head_top

## Body-Relative Coordinate System

The key innovation is using **body-relative directions** instead of global axes:

### Body "Up" Direction
- Calculated from: `shoulder_mid - hip_mid` 
- Represents the body's actual upright orientation
- Adapts to bending, leaning, or twisting poses

### Body "Forward" Direction  
- Calculated from: `cross(shoulder_line, body_up)`
- Always perpendicular to shoulder line and body up
- Ensures "forward" means "in front of the chest" regardless of body rotation

USAGE
---
Command line:
python 3_ApplyPhysics.py --show SHOW_NAME --scene SCENE_NAME

Arguments:
* --show: Override default show name
* --scene: Override current scene name

Input/Output files:
* Input: step_3_input.json
* Output: step_4_input.json
* Biometrics: step_3_output_biometrics.json

DEBUGGING
---
Debug logging in Log3.txt:
* Coordinate values before/after transformations
* Motion state classifications
* Depth adjustment calculations
* Scaling factors
* Shoulder and head stabilization corrections

TROUBLESHOOTING
---
Common Issues:

* Depth Still Exaggerated
- Lower depth_variation_trigger_ratio to 1.0 or below
- Set planar_motion_variance_ratio to 0.1
- Use manual_depth_scale_factor for direct control
- Enable force_minimal_depth_variation for aggressive correction

* Figure Too Flat
- Increase min_scale_factor to 0.3 or higher
- Raise depth_variation_trigger_ratio to 1.5
- Disable depth adjustment entirely

* Unrealistic Limb Proportions
- Lower biomechanical_discrepancy_threshold to 1.05
- Enable biomechanical constraints
- Enable shoulder stabilization for upper/lower body consistency

* Head Positioning Issues
- Enable head_stabilization in config
- Adjust min_head_vertical and min_head_forward thresholds
- Increase correction strengths for more aggressive fixes

* Shoulder Scaling Artifacts
- Enable shoulder_stabilization in config
- Adjust expected_shoulder_hip_ratio and expected_torso_aspect_ratio
- Modify correction_factor for faster/slower corrections

DEPENDENCIES
---
* numpy
* scikit-learn
* scipy
* Project utilities from utils.py

VERSION HISTORY
---
* v3.4 - Added comprehensive shoulder and head stabilization systems
* v3.1+ - Added configurable depth adjustment system
* v3.0 - Initial physics application system