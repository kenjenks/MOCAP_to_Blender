# _4D_magic_utils.py (Version 1.2 - Simplified debug_log)

import os
from datetime import datetime
import sys # Import sys for stdout flushing


# Global variable to store debug settings, set by the main script (4D_magic_inner.py)
_GLOBAL_DEBUG_SETTINGS = {}

def clear_log():
    """
        Clears the log file for script status.
        This function is universal and does not depend on Blender's bpy or mathutils.
    """
    try:
        log_dir = os.path.dirname(os.path.abspath(__file__))
        log_file_path = os.path.join(log_dir, "Log4D.txt")

        with open(log_file_path, 'w') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === START ===\n")
            f.flush()

    except Exception as e:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error clearing log file")
        sys.stdout.flush()


def script_log(message, mp_joint_name=None, debug_joint_settings=None, force_log=False):
    """
    Prints a message to the console and a separate log file for script status.
    Conditionally logs based on a joint's debug setting in debug_joint_settings (from debug_joints.json).
    This function is universal and does not depend on Blender's bpy or mathutils.

    Args:
        message (str): The log message.
        mp_joint_name (str, optional): The MediaPipe joint name (e.g., "LEFT_WRIST").
                                       If provided, the message will only be logged
                                       if this joint's 'debug' flag is True in debug_joint_settings.
        debug_joint_settings (dict, optional): The content of debug_joints.json.
                                               Used to determine if a specific joint's debug is enabled.
        force_log (bool, optional): If True, the message will be logged regardless of
                                    joint-specific debug settings. Useful for critical errors.
    """
    should_log = force_log

    if not should_log and debug_joint_settings:
        if mp_joint_name:
            # Check specific joint's debug flag
            joint_debug_info = debug_joint_settings.get(mp_joint_name)
            if joint_debug_info and joint_debug_info.get("debug", False):
                should_log = True
        
        # Also check global default debug flag from debug_joints.json metadata
        if not should_log and debug_joint_settings.get("metadata", {}).get("default_debug", False):
            should_log = True
    
    # If no specific joint or debug settings are provided, and it's not forced,
    # it's a general message, which should typically be logged unless it's a "debug_log" type
    # which is handled by the debug_log function itself.
    # This condition ensures general script_log calls (e.g., "Starting script...") are always logged.
    if not should_log and mp_joint_name is None and debug_joint_settings is None and not force_log:
        should_log = True


    if should_log:
        try:
            log_dir = os.path.dirname(os.path.abspath(__file__))
            log_file_path = os.path.join(log_dir, "Log4D.txt")

            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")
            sys.stdout.flush()

            with open(log_file_path, 'a') as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
                f.flush()

        except Exception as e:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error writing to log file: {e} - Message: {message}")
            sys.stdout.flush()


def debug_log(message):
    """
    Logs a message only if the 'DETAILED_DEBUGGING' flag in _GLOBAL_DEBUG_SETTINGS is True.
    This function simplifies logging of detailed debug information without needing to pass
    debug_joint_settings to every call.
    """
    global _GLOBAL_DEBUG_SETTINGS

    # Check the global DETAILED_DEBUGGING flag
    detailed_debugging_enabled = _GLOBAL_DEBUG_SETTINGS.get("metadata", {}).get("debug_flags", {}).get("DETAILED_DEBUGGING", False)
    
    if detailed_debugging_enabled:
        # Call script_log with force_log=True to ensure it's written
        # Pass _GLOBAL_DEBUG_SETTINGS so script_log can still use per-joint debug if it needs to,
        # though for debug_log, the primary gate is DETAILED_DEBUGGING.
        script_log(message, debug_joint_settings=_GLOBAL_DEBUG_SETTINGS, force_log=True)

