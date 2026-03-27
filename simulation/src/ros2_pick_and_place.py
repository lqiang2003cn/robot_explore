"""Isaac Sim scene for ROS2-driven Franka pick-and-place with video recording.

This script sets up the Franka Panda scene and the ROS2 bridge. A separate
ROS2 process (ros2_stack) sends joint commands via /joint_command and this
script publishes joint states on /joint_states and the ground-truth cube
pose on /cube_pose.

Usage (from the simulation/ directory with the roboex-simulation conda env):
    python -m src.ros2_pick_and_place
    python -m src.ros2_pick_and_place --output my_video.mp4
    python -m src.ros2_pick_and_place --no-headless
"""

import os
import sys
import functools

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# Force unbuffered output so print() shows up immediately in Isaac Sim
print = functools.partial(print, flush=True)  # type: ignore[assignment]

import argparse
import threading
import time
import numpy as np
from pathlib import Path

import isaacsim
from isaacsim.simulation_app import SimulationApp


PHYSICS_DT = 1 / 120
RENDERING_DT = 1 / 60
VIDEO_FPS = 30
CAPTURE_INTERVAL = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Isaac Sim scene for ROS2-driven Franka pick-and-place"
    )
    p.add_argument(
        "--output", type=str, default="output/ros2_pick_and_place.mp4",
        help="Output video path",
    )
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--max-steps", type=int, default=18000,
        help="Max simulation steps before aborting (longer for ROS2 planning latency)",
    )
    p.add_argument(
        "--timeout", type=float, default=300.0,
        help="Max wall-clock seconds to wait for ROS2 task completion",
    )
    return p


def setup_ros2_bridge(franka_prim_path: str) -> None:
    """Create an OmniGraph Action Graph for joint command subscription only.

    Joint state publishing is handled from Python (with wall-clock timestamps)
    so that MoveIt2 recognizes the timestamps as current.
    """
    import omni.graph.core as og

    og.Controller.edit(
        {"graph_path": "/ROS2Bridge", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("ArticulationController.inputs:robotPath", franka_prim_path),
            ],
        },
    )


class IsaacSimPublisher:
    """Publishes joint states, cube pose, and target pose with wall-clock timestamps."""

    def __init__(self, cube_getter, joint_getter, place_position: np.ndarray):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("isaac_sim_publisher")
        self._joint_pub = self._node.create_publisher(JointState, "/joint_states", 10)
        self._cube_pub = self._node.create_publisher(PoseStamped, "/cube_pose", 10)
        self._target_pub = self._node.create_publisher(PoseStamped, "/target_place_pose", 10)
        self._done_sub = self._node.create_subscription(
            Bool, "/task_complete", self._on_task_complete, 10,
        )
        self._cube_getter = cube_getter
        self._joint_getter = joint_getter
        self._place_position = place_position
        self._task_done = False
        self._spin_thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._spin_thread.start()

    def _on_task_complete(self, msg):
        if msg.data:
            self._task_done = True

    @property
    def task_done(self) -> bool:
        return self._task_done

    def publish(self) -> None:
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        now = self._node.get_clock().now().to_msg()

        # Publish joint states with wall-clock timestamp
        joint_names, joint_positions, joint_velocities = self._joint_getter()
        if joint_names:
            js_msg = JointState()
            js_msg.header.stamp = now
            js_msg.name = joint_names
            js_msg.position = joint_positions
            js_msg.velocity = joint_velocities
            self._joint_pub.publish(js_msg)

        # Publish cube pose
        cube_pos, cube_rot = self._cube_getter()

        cube_msg = PoseStamped()
        cube_msg.header.stamp = now
        cube_msg.header.frame_id = "world"
        cube_msg.pose.position.x = float(cube_pos[0])
        cube_msg.pose.position.y = float(cube_pos[1])
        cube_msg.pose.position.z = float(cube_pos[2])
        cube_msg.pose.orientation.w = float(cube_rot[0])
        cube_msg.pose.orientation.x = float(cube_rot[1])
        cube_msg.pose.orientation.y = float(cube_rot[2])
        cube_msg.pose.orientation.z = float(cube_rot[3])
        self._cube_pub.publish(cube_msg)

        target_msg = PoseStamped()
        target_msg.header.stamp = now
        target_msg.header.frame_id = "world"
        target_msg.pose.position.x = float(self._place_position[0])
        target_msg.pose.position.y = float(self._place_position[1])
        target_msg.pose.position.z = float(self._place_position[2])
        target_msg.pose.orientation.w = 1.0
        self._target_pub.publish(target_msg)

    def destroy(self) -> None:
        import rclpy
        self._node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


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

    # Enable ROS2 bridge extension
    import omni.kit.app
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

    # Allow extension to initialise
    app.update()
    app.update()

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
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

    # ── ROS2 bridge ───────────────────────────────────────────────────────
    setup_ros2_bridge("/World/Franka")

    # ── Camera for recording ──────────────────────────────────────────────
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

    # Joint name list matching the Franka articulation order
    _joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
        "panda_finger_joint1", "panda_finger_joint2",
    ]

    def _get_joints():
        try:
            pos = franka.get_joint_positions()
            vel = franka.get_joint_velocities()
            if pos is not None and len(pos) >= 9:
                return (
                    list(_joint_names),
                    [float(p) for p in pos[:9]],
                    [float(v) for v in vel[:9]] if vel is not None else [0.0] * 9,
                )
        except Exception:
            pass
        return ([], [], [])

    pose_pub = IsaacSimPublisher(
        cube_getter=lambda: cube.get_world_pose(),
        joint_getter=_get_joints,
        place_position=place_position,
    )

    frames: list[np.ndarray] = []

    # ── Main loop ─────────────────────────────────────────────────────────
    print("Isaac Sim ROS2 scene running — waiting for ROS2 controller …")
    print(f"  Publishing: /joint_states, /cube_pose, /target_place_pose")
    print(f"  Subscribing: /joint_command")
    print(f"  Listening for: /task_complete")

    start_wall = time.time()

    for step in range(args.max_steps):
        world.step(render=True)
        pose_pub.publish()

        if step % CAPTURE_INTERVAL == 0:
            data = rgb_annotator.get_data()
            if data is not None and data.size > 0:
                frames.append(data[:, :, :3].copy())

        if pose_pub.task_done:
            print(f"ROS2 controller signalled task complete at step {step}")
            for _ in range(120):
                world.step(render=True)
                data = rgb_annotator.get_data()
                if data is not None and data.size > 0:
                    frames.append(data[:, :, :3].copy())
            break

        elapsed = time.time() - start_wall
        if elapsed > args.timeout:
            print(f"Timeout ({args.timeout}s) reached at step {step}")
            break
    else:
        print(f"Reached max steps ({args.max_steps}) without task completion")

    # ── Save video ────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if frames:
        import imageio

        print(f"Writing {len(frames)} frames -> {output_path} ({VIDEO_FPS} fps) …")
        writer = imageio.get_writer(str(output_path), fps=VIDEO_FPS, codec="libx264")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"Video saved: {output_path}")
    else:
        print("Warning: no frames were captured.")

    pose_pub.destroy()
    app.close()


if __name__ == "__main__":
    main()
