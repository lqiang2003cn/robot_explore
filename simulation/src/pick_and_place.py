"""Franka Panda pick-and-place task with video recording (Isaac Sim 6.0.0).

Usage:
    python -m src.pick_and_place                          # defaults (headless, saves to output/pick_and_place.mp4)
    python -m src.pick_and_place --output my_video.mp4    # custom output path
    python -m src.pick_and_place --no-headless             # render to screen as well
    python -m src.pick_and_place --width 1920 --height 1080
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import argparse
import numpy as np
from pathlib import Path

import isaacsim
from isaacsim.simulation_app import SimulationApp


PHYSICS_DT = 1 / 120
RENDERING_DT = 1 / 60
VIDEO_FPS = 30
CAPTURE_INTERVAL = 2  # capture every N world steps → 60Hz / 2 = 30fps


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Franka Panda pick-and-place with video recording")
    p.add_argument("--output", type=str, default="output/pick_and_place.mp4", help="Output video path")
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--max-steps", type=int, default=6000, help="Max simulation steps before aborting")
    return p


def main() -> None:
    args = build_parser().parse_args()

    app = SimulationApp({
        "headless": args.headless,
        "width": args.width,
        "height": args.height,
        "renderer": "RaytracedLighting",
        "anti_aliasing": 0,
        "multi_gpu": False,
        "active_gpu": 0,
    })

    # ── Deferred imports (require a running SimulationApp) ────────────────
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
    from isaacsim.core.utils.viewports import set_camera_view
    import omni.replicator.core as rep

    # ── Scene ─────────────────────────────────────────────────────────────
    world = World(
        physics_dt=PHYSICS_DT,
        rendering_dt=RENDERING_DT,
        stage_units_in_meters=1.0,
    )
    world.scene.add_default_ground_plane()

    franka = world.scene.add(
        Franka(prim_path="/World/Franka", name="franka")
    )

    world.scene.add(
        FixedCuboid(
            prim_path="/World/Table",
            name="table",
            position=np.array([0.45, 0.0, 0.2]),
            scale=np.array([0.6, 0.9, 0.4]),
            color=np.array([0.40, 0.26, 0.13]),
        )
    )

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/Cube",
            name="target_cube",
            position=np.array([0.4, 0.12, 0.45]),
            scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([0.92, 0.10, 0.14]),
            mass=0.02,
        )
    )

    place_position = np.array([0.4, -0.25, 0.45])

    # ── Camera for recording ─────────────────────────────────────────────
    set_camera_view(
        eye=np.array([1.3, -1.0, 1.0]),
        target=np.array([0.35, 0.0, 0.45]),
    )

    render_product = rep.create.render_product(
        "/OmniverseKit_Persp", (args.width, args.height)
    )
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])

    # ── Init ──────────────────────────────────────────────────────────────
    world.reset()

    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
    )
    controller.reset()

    frames: list[np.ndarray] = []

    # ── Main loop ─────────────────────────────────────────────────────────
    print("Running Franka pick-and-place task …")

    for step in range(args.max_steps):
        world.step(render=True)

        cube_pos, _ = cube.get_world_pose()

        actions = controller.forward(
            picking_position=cube_pos,
            placing_position=place_position,
            current_joint_positions=franka.get_joint_positions(),
            end_effector_offset=np.array([0.0, 0.005, 0.0]),
        )
        franka.apply_action(actions)

        if step % CAPTURE_INTERVAL == 0:
            data = rgb_annotator.get_data()
            if data is not None and data.size > 0:
                frames.append(data[:, :, :3].copy())

        if controller.is_done():
            print(f"Pick-and-place completed at step {step}")
            for _ in range(120):
                world.step(render=True)
                data = rgb_annotator.get_data()
                if data is not None and data.size > 0:
                    frames.append(data[:, :, :3].copy())
            break
    else:
        print(f"Reached max steps ({args.max_steps}) without task completion")

    # ── Save video ────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames:
        import imageio

        print(f"Writing {len(frames)} frames → {output_path} ({VIDEO_FPS} fps) …")
        writer = imageio.get_writer(str(output_path), fps=VIDEO_FPS, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved: {output_path}")
    else:
        print("Warning: no frames were captured.")

    app.close()


if __name__ == "__main__":
    main()
