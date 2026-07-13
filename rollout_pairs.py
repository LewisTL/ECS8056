#!/usr/bin/env python3
"""
rollout_pairs.py
----------------
Watch a Franka arm execute OpenVLA's predicted end-effector deltas for each
contrastive pair: run instruction A (arm reaches in the predicted direction),
reset to home, then run instruction B. Renders live in the Isaac Sim GUI
(viewed over DCV) and records one MP4 per pair.

Run from a terminal INSIDE the DCV desktop session (the GUI needs a display):
    ISAAC=$(dirname $(find / -name python.sh -path '*isaac*' 2>/dev/null | head -1))
    cd "$ISAAC"
    ./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --out ~/rollouts

Scale caveat: one OpenVLA action delta is ~3 mm (a single 5 Hz control step),
imperceptible as a one-shot reach. --scale exaggerates it for visibility. The
DIRECTION of motion is faithful to the model; absolute distance is not. State
this in any caption.

Version note: the robot/controller import block and the end-effector pose
accessor are the parts most sensitive to the exact Isaac Sim version. If an
import fails, run `cat "$ISAAC/VERSION"` and adjust that block.
"""

import argparse
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True, help="path to pairs.json")
ap.add_argument("--out", default="./rollouts", help="output dir for videos")
ap.add_argument("--limit", type=int, default=0, help="only first N pairs")
ap.add_argument("--scale", type=float, default=40.0,
                help="exaggerate delta for visibility (direction preserved)")
ap.add_argument("--steps", type=int, default=120, help="sim steps per reach")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--headless", action="store_true",
                help="no GUI (still records video)")
args = ap.parse_args()

# ---- 1. App before any omni/isaacsim import --------------------------------
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

# ---- 2. Version-tolerant imports (5.x first, 4.x fallback) ------------------
try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.utils.nucleus import get_assets_root_path
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController)
    from isaacsim.sensors.camera import Camera
    NS = "isaacsim"
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.core.utils.nucleus import get_assets_root_path
    from omni.isaac.franka import Franka
    from omni.isaac.franka.controllers import RMPFlowController
    from omni.isaac.sensor import Camera
    NS = "omni.isaac"

print(f"[rollout] using {NS} namespace")


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = eye - target
    z /= (np.linalg.norm(z) + 1e-12)
    x = np.cross(np.asarray(up, float), z)
    x /= (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    m = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2.0
    if w < 1e-8:
        return np.array([1.0, 0, 0, 0])
    return np.array([w, (m[2, 1] - m[1, 2]) / (4 * w),
                     (m[0, 2] - m[2, 0]) / (4 * w),
                     (m[1, 0] - m[0, 1]) / (4 * w)])


def write_video(frames, path, fps):
    """MP4 if imageio+ffmpeg are present, else a PNG sequence + ffmpeg hint."""
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
        print(f"    stitch with: ffmpeg -framerate {fps} "
              f"-i {seq_dir}/%04d.png -pix_fmt yuv420p {path}")


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

    cam_eye = np.array([1.6, 1.6, 1.2])
    cam_target = np.array([0.45, 0.0, 0.35])
    camera = Camera(prim_path="/World/cam", position=cam_eye,
                    orientation=look_at_quat(cam_eye, cam_target),
                    resolution=(1280, 720))

    controller = RMPFlowController(name="rmpflow", robot_articulation=franka)

    world.reset()
    camera.initialize()
    controller.reset()
    for _ in range(30):
        world.step(render=True)

    # Home configuration captured after settling.
    home_joints = franka.get_joint_positions()
    ee_home_pos, ee_home_quat = franka.end_effector.get_world_pose()
    print(f"[rollout] EE home pose: {np.round(ee_home_pos, 3)}")

    def reach(target_pos, target_quat, frames):
        for _ in range(args.steps):
            action = controller.forward(
                target_end_effector_position=target_pos,
                target_end_effector_orientation=target_quat)
            franka.apply_action(action)
            world.step(render=True)
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
            # brief hold at home so the video shows the start, then reach
            for _ in range(int(args.fps * 0.5)):
                world.step(render=True)
                frames.append(camera.get_rgba()[..., :3].astype(np.uint8))
            reach(target, ee_home_quat, frames)
            for _ in range(int(args.fps * 0.7)):   # hold at target
                world.step(render=True)
                frames.append(camera.get_rgba()[..., :3].astype(np.uint8))

        write_video(frames, os.path.join(out_dir, f"{p['pair_id']}.mp4"),
                    args.fps)

    print(f"[rollout] done -> {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
