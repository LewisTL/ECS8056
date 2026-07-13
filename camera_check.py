#!/usr/bin/env python3
"""
camera_check.py — minimal camera/scene sanity test.
Loads a Franka + table, aims the camera with the built-in viewport helper,
steps enough frames to settle, and saves ONE png. If the arm is visible here,
the fix is confirmed and we port this camera setup into the real scripts.

    cd /opt/IsaacSim
    ./python.sh ~/camera_check.py --out ~/cam_test.png
"""
import argparse, os
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="./cam_test.png")
args = ap.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import numpy as np

# --- version-tolerant imports ---
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.franka import Franka
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.viewports import set_camera_view

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.scene.add(FixedCuboid(prim_path="/World/table", name="table",
    position=np.array([0.45, 0.0, 0.10]),
    scale=np.array([0.50, 0.90, 0.20]),
    color=np.array([0.85, 0.85, 0.85])))
world.scene.add(Franka(prim_path="/World/franka", name="franka"))

# Key fix 1: aim the viewport camera with the built-in helper (correct every time)
EYE = [2.2, 2.2, 1.8]
TARGET = [0.45, 0.0, 0.30]
set_camera_view(eye=EYE, target=TARGET)

# A capture camera that inherits the viewport is fiddly across versions, so we
# render the *viewport* the helper just aimed. Camera() here just gives us a
# render product tied to the default viewport camera path.
try:
    cam = Camera(prim_path="/OmniverseKit_Persp", resolution=(1280, 720))
except Exception:
    cam = Camera(prim_path="/World/cap", position=np.array(EYE),
                 resolution=(1280, 720))

world.reset()
cam.initialize()

# Key fix 2: step enough frames for the transform + render to settle
for i in range(60):
    world.step(render=True)

rgba = cam.get_rgba()
print("frame shape:", None if rgba is None else rgba.shape,
      "| nonzero pixels:", 0 if rgba is None else int(np.count_nonzero(rgba[..., :3])))

from PIL import Image
Image.fromarray(rgba[..., :3].astype(np.uint8)).save(os.path.expanduser(args.out))
print("saved", args.out)
simulation_app.close()
