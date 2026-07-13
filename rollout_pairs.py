#!/usr/bin/env python3
"""
rollout_pairs.py
----------------
Watch a Franka arm execute OpenVLA's predicted end-effector deltas for each
contrastive pair: run instruction A (arm reaches), reset to home, then B.
Renders live in the Isaac Sim GUI (viewed over DCV) and records one MP4 per pair.

Run from a terminal INSIDE the DCV desktop (the GUI needs a display):
    cd /opt/IsaacSim
    ./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --out ~/rollouts

Scale caveat: one OpenVLA delta is ~3 mm (a single 5 Hz control step),
imperceptible as a one-shot reach. --scale exaggerates it for visibility. The
DIRECTION of motion is faithful to the model; absolute distance is not.

Version note: the robot/controller imports and the end-effector pose accessor
are the version-sensitive lines. If one fails, run `cat /opt/IsaacSim/VERSION`.
"""

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", default="./rollouts")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--scale", type=float, default=40.0,
                help="exaggerate delta for visibility (direction preserved)")
ap.add_argument("--steps", type=int, default=120, help="sim steps per reach")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--headless", action="store_true", help="no GUI (still records)")
args = ap.parse_args()

# ---- 1. App before any omni/isaacsim import --------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

# ---- 2. Version-tolerant imports -------------------------------------------
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController)
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_camera_view
    NS = "isaacsim"
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.franka import Franka
    from omni.isaac.franka.controllers import RMPFlowController
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.viewports import set_camera_view
    NS = "omni.isaac"

print(f"[rollout] using {NS} namespace")

CAM_EYE = [1.9, 1.9, 1.5]
CAM_TARGET = [0.45, 0.0, 0.35]


def write_video(frames, path, fps):
    if not frames:
        return
    try:
        import imageio.v2 as imageio
        imageio.mimsave(path, frames, fps=fps, macro_block_size=None)
        print(f"    saved {path} ({len(frames)} frames)")
    except Exception as e:
        seq_dir = path.rsplit(".", 1)[0] + "_frames"
        os.makedirs(seq_dir, exist_ok=True)
        from PIL import Image
        for k, fr in enumerate(frames):
            Image.fromarray(fr).save(os.path.join(seq_dir, f"{k:04d}.png"))
        print(f"    imageio unavailable ({e}); wrote PNGs to {seq_dir}")
        print(f"    stitch: ffmpeg -framerate {fps} -i {seq_dir}/%04d.png "
              f"-pix_fmt yuv420p {path}")


def main():
    with open(os.path.expanduser(args.pairs)) as f:
        pairs = json.load(f)
    if args.limit:
        pairs = pairs[: args.limit]
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    world.scene.add(FixedCuboid(
        prim_path="/World/table", name="table",
        position=np.array([0.45, 0.0, 0.10]),
        scale=np.array([0.50, 0.90, 0.20]),
        color=np.array([0.85, 0.85, 0.85])))
    franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))

    # Camera: aim the persp viewport (proven by camera_check.py).
    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
    try:
        camera = Camera(prim_path="/OmniverseKit_Persp", resolution=(1280, 720))
    except Exception:
        camera = Camera(prim_path="/World/cap_cam", position=np.array(CAM_EYE),
                        resolution=(1280, 720))

    controller = RMPFlowController(name="rmpflow", robot_articulation=franka)

    world.reset()
    camera.initialize()
    controller.reset()
    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)   # re-aim after reset
    for _ in range(60):                               # settle + warm-up
        world.step(render=True)

    home_joints = franka.get_joint_positions()
    ee_home_pos, ee_home_quat = franka.end_effector.get_world_pose()
    print(f"[rollout] EE home pose: {np.round(ee_home_pos, 3)}")

    def grab(frames):
        frames.append(camera.get_rgba()[..., :3].astype(np.uint8))

    def go_home():
        franka.set_joint_positions(home_joints)
        controller.reset()
        for _ in range(20):
            world.step(render=True)

    for i, p in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {p['pair_id']}")
        frames = []
        for role in ("a", "b"):
            go_home()
            delta = np.asarray(p[f"action_{role}"][:3], dtype=float) * args.scale
            target = ee_home_pos + delta
            print(f"    {role}: '{p['instr_' + role]}' -> "
                  f"delta*scale={np.round(delta, 3)}")
            for _ in range(int(args.fps * 0.5)):      # hold at home
                world.step(render=True); grab(frames)
            for _ in range(args.steps):               # reach
                action = controller.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=ee_home_quat)
                franka.apply_action(action)
                world.step(render=True); grab(frames)
            for _ in range(int(args.fps * 0.7)):      # hold at target
                world.step(render=True); grab(frames)

        write_video(frames, os.path.join(out_dir, f"{p['pair_id']}.mp4"), args.fps)

    print(f"[rollout] done -> {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()