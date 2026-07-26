#!/usr/bin/env python3
"""
rollout_pairs.py
----------------
Franka executes OpenVLA's predicted deltas per contrastive pair (A, home, B),
recorded as one self-explanatory MP4 per pair:

  * LEFT PANEL:  the real BridgeData frame the model saw, plus the instruction
    currently executing (blue = A, orange = B).
  * RIGHT PANEL: the Isaac Sim rollout, with a persistent end-effector trail
    per role (blue/orange spheres) so both trajectories remain visible for
    direct comparison by the end of the clip.

Usage (terminal inside the DCV desktop for live view, or --headless):
    cd /opt/IsaacSim
    ./python.sh ~/rollout_pairs.py --pairs ~/pairs.json --out ~/rollouts \
        --frames-root ~/bridge_frames

--frames-root points at the cached BridgeData frames (copy once with e.g.
`rclone copy gdrive:openvla_cache/bridge_multiobj/frames ~/bridge_frames`).
Pairs whose image can't be found still render, without the photo.

Scale caveat: one OpenVLA delta is ~3 mm; --scale exaggerates it for
visibility. Direction is faithful, distance is not, so state this in captions.
"""

import argparse
import json
import os
import textwrap

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True)
ap.add_argument("--out", default="./rollouts")
ap.add_argument("--frames-root", default=None,
                help="dir containing the cached BridgeData PNGs")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--scale", type=float, default=40.0)
ap.add_argument("--steps", type=int, default=120)
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--trail-every", type=int, default=4,
                help="drop a trail sphere every N sim steps")
ap.add_argument("--headless", action="store_true")
args = ap.parse_args()

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": args.headless})

import numpy as np

try:
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid, VisualSphere
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController)
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.utils.viewports import set_camera_view
except ImportError:  # Isaac Sim 4.x
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid, VisualSphere
    from omni.isaac.franka import Franka
    from omni.isaac.franka.controllers import RMPFlowController
    from omni.isaac.sensor import Camera
    from omni.isaac.core.utils.viewports import set_camera_view

from PIL import Image, ImageDraw, ImageFont

CAM_EYE = [1.9, 1.9, 1.5]
CAM_TARGET = [0.45, 0.0, 0.35]
COLOR_A_RGB = (26, 115, 255)
COLOR_B_RGB = (255, 140, 26)
COLOR_A = np.array(COLOR_A_RGB) / 255.0
COLOR_B = np.array(COLOR_B_RGB) / 255.0
PANEL_W = 640            # left panel width; sim frame is 1280x720 -> 1920x720
SIM_W, SIM_H = 1280, 720
TRAIL_POOL = 80          # spheres per role
PARK = np.array([0.0, 0.0, -10.0])


class Trail:
    """Pre-created sphere pool per role; spheres are parked below ground and
    moved onto the end-effector path as it executes."""

    def __init__(self, world, name, color):
        self.spheres = [world.scene.add(VisualSphere(
            prim_path=f"/World/trail_{name}_{k}", name=f"trail_{name}_{k}",
            position=PARK + np.array([0.05 * k, 0.0, 0.0]),
            radius=0.008, color=color)) for k in range(TRAIL_POOL)]
        self.i = 0

    def drop(self, pos):
        if self.i < len(self.spheres):
            self.spheres[self.i].set_world_pose(np.asarray(pos, dtype=float),
                                                np.array([1.0, 0, 0, 0]))
            self.i += 1

    def clear(self):
        for k, s in enumerate(self.spheres):
            s.set_world_pose(PARK + np.array([0.05 * k, 0.0, 0.0]),
                             np.array([1.0, 0, 0, 0]))
        self.i = 0


def load_scene_image(p):
    """Locate and load the pair's BridgeData frame, or None."""
    if not args.frames_root or "image_path" not in p:
        return None
    root = os.path.expanduser(args.frames_root)
    for cand in (os.path.join(root, p["image_path"]),
                 os.path.join(root, os.path.basename(p["image_path"]))):
        if os.path.exists(cand):
            return Image.open(cand).convert("RGB")
    return None


def make_panel(scene_img, instr, role, pair_id):
    """Left panel: scene photo + instruction banner coloured by role."""
    panel = Image.new("RGB", (PANEL_W, SIM_H), (24, 24, 24))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()

    if scene_img is not None:
        img = scene_img.copy()
        img.thumbnail((PANEL_W - 20, 440))
        panel.paste(img, ((PANEL_W - img.width) // 2, 10))
        y = 10 + img.height + 14
        draw.text((10, y), "model input (BridgeData V2)", fill=(150, 150, 150),
                  font=font)
        y += 22
    else:
        draw.text((10, 10), "(scene image unavailable)", fill=(150, 150, 150),
                  font=font)
        y = 40

    color = COLOR_A_RGB if role == "a" else COLOR_B_RGB
    draw.rectangle([0, y, PANEL_W, y + 26], fill=color)
    draw.text((10, y + 6), f"executing instruction {role.upper()}",
              fill=(0, 0, 0), font=font)
    y += 34
    for line in textwrap.wrap(instr, width=70)[:4]:
        draw.text((10, y), line, fill=(235, 235, 235), font=font)
        y += 16
    draw.text((10, SIM_H - 22), pair_id, fill=(120, 120, 120), font=font)
    return panel


def compose(sim_rgb, panel):
    canvas = Image.new("RGB", (PANEL_W + SIM_W, SIM_H))
    canvas.paste(panel, (0, 0))
    canvas.paste(Image.fromarray(sim_rgb), (PANEL_W, 0))
    return np.asarray(canvas)


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

    trails = {"a": Trail(world, "a", COLOR_A), "b": Trail(world, "b", COLOR_B)}

    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
    try:
        camera = Camera(prim_path="/OmniverseKit_Persp", resolution=(SIM_W, SIM_H))
    except Exception:
        camera = Camera(prim_path="/World/cap_cam", position=np.array(CAM_EYE),
                        resolution=(SIM_W, SIM_H))

    controller = RMPFlowController(name="rmpflow", robot_articulation=franka)
    world.reset()
    camera.initialize()
    controller.reset()
    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)
    for _ in range(60):
        world.step(render=True)

    home_joints = franka.get_joint_positions()
    ee_home_pos, ee_home_quat = franka.end_effector.get_world_pose()
    print(f"[rollout] EE home pose: {np.round(ee_home_pos, 3)}")

    def go_home():
        franka.set_joint_positions(home_joints)
        controller.reset()
        for _ in range(20):
            world.step(render=True)

    for i, p in enumerate(pairs):
        print(f"[{i + 1}/{len(pairs)}] {p['pair_id']}")
        scene_img = load_scene_image(p)
        if scene_img is None and args.frames_root:
            print("    note: scene image not found; rendering without photo")
        trails["a"].clear()
        trails["b"].clear()
        frames = []

        for role in ("a", "b"):
            go_home()
            panel = make_panel(scene_img, p[f"instr_{role}"], role, p["pair_id"])
            delta = np.asarray(p[f"action_{role}"][:3], dtype=float) * args.scale
            target = ee_home_pos + delta
            print(f"    {role}: '{p['instr_' + role]}' -> "
                  f"delta*scale={np.round(delta, 3)}")

            def grab():
                frames.append(compose(
                    camera.get_rgba()[..., :3].astype(np.uint8), panel))

            for _ in range(int(args.fps * 0.5)):
                world.step(render=True); grab()
            for s in range(args.steps):
                action = controller.forward(
                    target_end_effector_position=target,
                    target_end_effector_orientation=ee_home_quat)
                franka.apply_action(action)
                world.step(render=True)
                if s % args.trail_every == 0:
                    ee_pos, _ = franka.end_effector.get_world_pose()
                    trails[role].drop(ee_pos)
                grab()
            for _ in range(int(args.fps * 0.7)):
                world.step(render=True); grab()

        write_video(frames, os.path.join(out_dir, f"{p['pair_id']}.mp4"), args.fps)

    print(f"[rollout] done -> {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()