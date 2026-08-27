"""Shared constants for the capture_* modules.

These are literal values referenced from more than one of the capture_*.py
modules (paths, image size, the self-body exclusion distance). Kept in their
own leaf module so no two capture_*.py files need to import each other just
to share a constant.
"""

MOUNT_PATH = (
    "/World/ArmWithHandOnly/Robotiq_2F_85/Robotiq_2F_85/base_link/d455_camera"
)
CAMERA_PATHS = {
    "left": MOUNT_PATH + "/RSD455/Camera_OmniVision_OV9782_Left",
    "right": MOUNT_PATH + "/RSD455/Camera_OmniVision_OV9782_Right",
    "color": MOUNT_PATH + "/RSD455/Camera_OmniVision_OV9782_Color",
}
IK_TARGET_PATH = "/World/IKTarget"
# Prim type Xform. If missing, stop and report -- do not guess another path.
ARM_BASE_PATH = "/World/ArmWithHandOnly/NHC12_A00/base_link"
WIDTH = 640
HEIGHT = 480
REGISTRY_NAME = "_nhc12_d455_capture_registry"

# Not a free parameter: measured from the Robotiq 2F-85's own AABB envelope
# (0.167 m from the camera at its furthest) and the depth gap observed from
# 0.18-0.45 m. Do not change this value.
SELF_BODY_MAX_CAMERA_DISTANCE_M = 0.180
