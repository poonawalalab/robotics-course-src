"""
This file is based on code in https://github.com/i2rt-robotics/i2rt/tree/main/i2rt/robots
It generates an xml that combines the YAM with the linear_3507 gripper 
Only this combination is supported because I did not copy the config reading code, but hard-coded it instead, reading from https://github.com/i2rt-robotics/i2rt/blob/main/i2rt/robots/config/linear_3507.yml
It assumes you have downloaded https://github.com/i2rt-robotics/i2rt/tree/main/i2rt/robot_models locally
I had to manually add gripper actuators to result (hardcoded output file) and play with parameters
To work with yam_sim and fsm, also need to add "cube_main" and mocap "target" to output xml
"""
import xml.etree.ElementTree as ET
import mujoco
import os
import numpy as np
import yaml
from copy import deepcopy
import tempfile


def _find_deepest_body(element: ET.Element) -> ET.Element:
    """Return the deepest (leaf) body in a kinematic chain.

    Walks into the first child ``<body>`` at each level until no more child
    bodies exist, then returns that leaf body.  This is used to locate the
    tip of the arm chain where the gripper should be appended.
    """
    current = element
    while True:
        child_bodies = [c for c in current if c.tag == "body"]
        if not child_bodies:
            return current
        current = child_bodies[0]

def combine_arm_and_gripper_xml() -> str:
    """Combine arm and gripper XML files into a single XML string.

    Appends the ``<body name="gripper">`` subtree from the gripper XML as a
    child of the deepest body in the arm's kinematic chain.  The last body
    in the arm chain is located dynamically via ``_find_deepest_body``, and
    its ``pos``, ``quat``, and first joint's ``axis`` are set from the
    gripper type's per-arm YAML config.

    Args:
        arm_type: ArmType enum value. Determines arm XML path and selects the
            correct per-arm mounting transform from the gripper's YAML config.
        gripper_type: GripperType enum value. Determines gripper XML path and
            mounting geometry from YAML config.
        ee_mass: Optional end-effector mass (kg) to override in gripper's inertial.
        ee_inertia: Optional end-effector inertia array. Expected as a flat array of
            10 elements: [ipos(3), quat(4), diaginertia(3)].

    Returns:
        Path to the combined XML file written to /tmp/.
    """
    arm_path = "robot_models/arm/yam/yam.xml"
    gripper_path = "robot_models/gripper/linear_3507/linear_3507.xml"

    arm_tree = ET.parse(arm_path)
    arm_root = arm_tree.getroot()

    # Set last-joint mounting geometry from gripper config (per-arm)
    worldbody = arm_root.find("worldbody")
    ## Deleted some config-setting code here
    if worldbody is not None:
        last_link = _find_deepest_body(worldbody)
        last_link.set("pos", "2.39858e-07 -0.0419481 0.0404996")
        last_link.set("quat", "0.499998 -0.5 -0.5 -0.500002")
        last_joint = last_link.find("joint")
        if last_joint is not None:
            last_joint.set("axis", "0 0 -1")

    # Resolve arm mesh paths to absolute
    arm_dir = os.path.dirname(os.path.abspath(arm_path))
    arm_compiler = arm_root.find("compiler")
    arm_meshdir = arm_compiler.get("meshdir", "") if arm_compiler is not None else ""
    arm_asset = arm_root.find("asset")
    if arm_asset is not None:
        for child in arm_asset:
            if child.get("file") and not os.path.isabs(child.get("file")):
                abs_file = os.path.join(arm_dir, arm_meshdir, child.get("file"))
                child.set("file", os.path.abspath(abs_file))

    # Remove meshdir from compiler (all paths now absolute)
    if arm_compiler is not None and arm_compiler.get("meshdir"):
        del arm_compiler.attrib["meshdir"]

    # attempt to load gripper and attach gripper body if available
    if gripper_path:
        try:
            grip_tree = ET.parse(gripper_path)
            grip_root = grip_tree.getroot()
            grip_body = grip_root.find(".//body[@name='gripper']")
        except Exception:
            grip_root = None
            grip_body = None

        # merge assets (avoid duplicates), resolving gripper mesh paths to absolute
        if grip_root is not None:
            grip_dir = os.path.dirname(os.path.abspath(gripper_path))
            grip_compiler = grip_root.find("compiler")
            grip_meshdir = grip_compiler.get("meshdir", "") if grip_compiler is not None else ""

            grip_asset = grip_root.find("asset")
            if grip_asset is not None:
                if arm_asset is None:
                    arm_asset = ET.Element("asset")
                    worldbody = arm_root.find("worldbody")
                    if worldbody is not None:
                        arm_root.insert(list(arm_root).index(worldbody), arm_asset)
                    else:
                        arm_root.append(arm_asset)
                existing = {(c.tag, c.get("name")) for c in arm_asset}
                for child in grip_asset:
                    key = (child.tag, child.get("name"))
                    if key not in existing:
                        elem = deepcopy(child)
                        if elem.get("file") and not os.path.isabs(elem.get("file")):
                            abs_file = os.path.join(grip_dir, grip_meshdir, elem.get("file"))
                            elem.set("file", os.path.abspath(abs_file))
                        arm_asset.append(elem)
                        existing.add(key)

        # Attach gripper body to the arm.
        # If the arm still has a legacy <body name="gripper"> placeholder, replace it.
        # Otherwise append the gripper body as a child of the deepest arm body.
        if grip_body is not None:
            existing_gripper = arm_root.find(".//body[@name='gripper']")
            if existing_gripper is not None:
                # Legacy path: replace the placeholder
                for parent in arm_root.iter():
                    children = list(parent)
                    for idx, child in enumerate(children):
                        if child.tag == "body" and child.get("name") == "gripper":
                            parent.remove(child)
                            parent.insert(idx, deepcopy(grip_body))
                            break
                    else:
                        continue
                    break
            else:
                # New path: append gripper body to the deepest body in the arm chain
                worldbody = arm_root.find("worldbody")
                if worldbody is not None:
                    tip_body = _find_deepest_body(worldbody)
                    tip_body.append(deepcopy(grip_body))

        # merge optional top-level sections (equality, contact) from gripper
        if grip_root is not None:
            for section_tag in ("equality", "contact"):
                grip_section = grip_root.find(section_tag)
                if grip_section is None:
                    continue
                arm_section = arm_root.find(section_tag)
                if arm_section is None:
                    arm_section = ET.SubElement(arm_root, section_tag)
                for child in grip_section:
                    arm_section.append(deepcopy(child))


    # write combined xml to /tmp/ and return filepath
    out_path = tempfile.NamedTemporaryFile(
        suffix=".xml", prefix=f"i2rt__", delete=False, dir="/tmp"
    ).name
    arm_tree.write("test_yam_raw.xml", encoding="utf-8", xml_declaration=True)
    return out_path

def merge_mjcf(file1, file2):
    tree1 = ET.parse(file1)
    tree2 = ET.parse(file2)
    root1 = tree1.getroot()  # <mujoco>
    root2 = tree2.getroot()  # <mujoco>

    # Merge each top-level section
    for elem2 in root2:
        # Find matching section in root1 (e.g. <worldbody>, <actuator>, etc.)
        existing = root1.find(elem2.tag)
        if existing is None:
            root1.append(elem2)
        else:
            # Append children into the existing section
            for child in elem2:
                existing.append(child)

    return ET.tostring(root1, encoding='unicode')

xml_string = combine_arm_and_gripper_xml()
