"""ROS2-to-MuJoCo bridge for the X3Plus arm.

A pure translator between ROS2 topics and the MuJoCo simulation. Receives
absolute joint position commands on /joint_command and drives the MuJoCo
position actuators directly. Publishes joint states, cube pose, target
place pose, and front-camera images.

Usage:
    source /opt/ros/jazzy/setup.bash
    source activate_env.sh vla_x3plus
    python -m src.mujoco_bridge_node
    python -m src.mujoco_bridge_node --place-pos 0.15 -0.15 0.39
    python -m src.mujoco_bridge_node --record-video output/bt_pick_place.mp4
"""

from __future__ import annotations

import argparse
import os
import pathlib
import threading

os.environ.setdefault("MUJOCO_GL", "osmesa")

import mujoco
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent

ACTUATOR_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "grip_joint",
]

ALL_JOINT_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5",
    "grip_joint",
    "rlink_joint2", "rlink_joint3",
    "llink_joint1", "llink_joint2", "llink_joint3",
]


class MuJoCoBridgeNode(Node):
    def __init__(
        self,
        model_path: str,
        place_position: list[float],
        camera_name: str = "front_cam",
        resolution: tuple[int, int] = (256, 256),
        record_video: str | None = None,
        video_fps: int = 30,
    ):
        super().__init__("mujoco_bridge")

        xml_path = COMPONENT_DIR / model_path
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        h, w = resolution
        self._renderer = mujoco.Renderer(self._model, height=h, width=w)
        self._camera_name = camera_name
        self._resolution = resolution
        self._place_position = np.array(place_position, dtype=np.float64)

        self._record_video_path = record_video
        self._video_fps = video_fps
        self._video_frames: list[np.ndarray] = []
        self._recording = record_video is not None
        self._video_saved = False
        self._saving_video = threading.Lock()
        self._qpos_history: list[np.ndarray] = []

        self._actuator_ctrl_idx = {
            name: i for i, name in enumerate(ACTUATOR_JOINT_NAMES)
        }
        self._joint_qpos_adr = {
            name: self._model.joint(name).qposadr[0]
            for name in ALL_JOINT_NAMES
        }
        self._joint_qvel_adr = {
            name: self._model.joint(name).dofadr[0]
            for name in ALL_JOINT_NAMES
        }

        mujoco.mj_forward(self._model, self._data)

        self._base_link_pos = self._data.body("base_link").xpos.copy()
        self._place_in_base = self._place_position - self._base_link_pos

        weld_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_EQUALITY, "grip_weld",
        )
        self._weld_eq_id = weld_id
        self._grip_attached = False
        self._grip_ctrl_idx = self._actuator_ctrl_idx["grip_joint"]
        self._cube_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "target_cube",
        )
        self._arm5_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "arm_link5",
        )
        self._attach_dist = 0.12
        self._detach_ctrl = -0.5

        self._lock = threading.Lock()
        self._latest_rgb: np.ndarray | None = None
        self._render_counter = 0

        cb = ReentrantCallbackGroup()

        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._cube_pose_pub = self.create_publisher(PoseStamped, "/cube_pose", 10)
        self._target_pose_pub = self.create_publisher(PoseStamped, "/target_place_pose", 10)
        self._image_pub = self.create_publisher(Image, "/front_cam/image_raw", 10)

        self.create_subscription(
            JointState, "/joint_command", self._on_joint_command, 10,
            callback_group=cb,
        )
        self.create_subscription(
            Bool, "/task_complete", self._on_task_complete, 10,
            callback_group=cb,
        )

        self._sim_timer = self.create_timer(1.0 / 60.0, self._step_sim)
        self._pub_timer = self.create_timer(1.0 / 20.0, self._publish_state)

        if self._recording:
            self.get_logger().info(f"Video recording enabled -> {record_video}")

        self.get_logger().info(
            f"MuJoCo bridge ready — model={xml_path.name}  "
            f"base_link@world={self._base_link_pos.tolist()}  "
            f"place_world={self._place_position.tolist()}  "
            f"place_base={self._place_in_base.tolist()}"
        )

    def _on_joint_command(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                idx = self._actuator_ctrl_idx.get(name)
                if idx is not None:
                    self._data.ctrl[idx] = pos

    def _on_task_complete(self, msg: Bool) -> None:
        if not msg.data or self._video_saved:
            return
        if not self._saving_video.acquire(blocking=False):
            return
        try:
            if self._video_saved:
                return
            self.get_logger().info("Received task_complete — processing")
            if self._recording:
                self._save_video()
            else:
                with self._lock:
                    mujoco.mj_resetData(self._model, self._data)
                    self._data.ctrl[:] = 0.0
                    mujoco.mj_forward(self._model, self._data)
        finally:
            self._saving_video.release()

    def _save_video(self) -> None:
        import imageio

        self._video_saved = True
        self._sim_timer.cancel()
        self._pub_timer.cancel()

        path = pathlib.Path(self._record_video_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        n_snapshots = len(self._qpos_history)
        if n_snapshots == 0:
            self.get_logger().warn("No qpos snapshots recorded, skipping video")
            return

        snap_dt = self._model.opt.timestep * self._SIM_SUBSTEPS
        sim_duration = n_snapshots * snap_dt
        n_frames = max(1, int(sim_duration * self._video_fps))
        self.get_logger().info(
            f"Rendering {n_frames} video frames from {n_snapshots} sim snapshots "
            f"({sim_duration:.1f}s sim time) -> {path}"
        )

        import time as _time

        frames: list[np.ndarray] = []
        t0 = _time.monotonic()
        for i in range(n_frames):
            t = i / self._video_fps
            idx = min(int(t / snap_dt), n_snapshots - 1)
            self._data.qpos[:] = self._qpos_history[idx]
            mujoco.mj_forward(self._model, self._data)
            self._renderer.update_scene(self._data, camera=self._camera_name)
            frames.append(self._renderer.render().copy())
            if (i + 1) % 20 == 0 or i == n_frames - 1:
                elapsed = _time.monotonic() - t0
                self.get_logger().info(
                    f"  rendered {i+1}/{n_frames} frames ({elapsed:.1f}s)"
                )

        self.get_logger().info(f"Writing {len(frames)} frames at {self._video_fps} fps")
        imageio.mimwrite(
            str(path),
            frames,
            fps=self._video_fps,
            quality=10,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-preset", "slow", "-crf", "17"],
        )
        self._qpos_history.clear()
        self.get_logger().info(f"Video saved: {path}")
        raise SystemExit(0)

    _SIM_SUBSTEPS = 16

    def _update_grip_weld(self) -> None:
        """Magnetic gripper: activate weld when gripper closes near cube."""
        grip_ctrl = self._data.ctrl[self._grip_ctrl_idx]
        cube_pos = self._data.body(self._cube_body_id).xpos.copy()
        cube_quat = self._data.body(self._cube_body_id).xquat.copy()
        arm5_pos = self._data.body(self._arm5_body_id).xpos.copy()
        arm5_quat = self._data.body(self._arm5_body_id).xquat.copy()
        dist = np.linalg.norm(cube_pos - arm5_pos)

        if not self._grip_attached:
            if grip_ctrl > self._detach_ctrl and dist < self._attach_dist:
                inv_arm5_quat = np.zeros(4)
                mujoco.mju_negQuat(inv_arm5_quat, arm5_quat)

                rel_pos = np.zeros(3)
                diff = cube_pos - arm5_pos
                mujoco.mju_rotVecQuat(rel_pos, diff, inv_arm5_quat)

                rel_quat = np.zeros(4)
                mujoco.mju_mulQuat(rel_quat, inv_arm5_quat, cube_quat)

                eq_data = self._model.eq_data[self._weld_eq_id]
                eq_data[0:3] = 0.0
                eq_data[3:6] = rel_pos
                eq_data[6:10] = rel_quat

                self._data.eq_active[self._weld_eq_id] = 1
                self._grip_attached = True
                self.get_logger().info(
                    f"Grip ATTACHED (dist={dist:.3f}m, ctrl={grip_ctrl:.2f}, "
                    f"rel_pos=[{rel_pos[0]:.3f},{rel_pos[1]:.3f},{rel_pos[2]:.3f}])"
                )
        else:
            if grip_ctrl <= self._detach_ctrl:
                self._data.eq_active[self._weld_eq_id] = 0
                self._grip_attached = False
                self.get_logger().info(
                    f"Grip RELEASED (ctrl={grip_ctrl:.2f})"
                )

    def _step_sim(self) -> None:
        with self._lock:
            self._update_grip_weld()
            for _ in range(self._SIM_SUBSTEPS):
                mujoco.mj_step(self._model, self._data)
            if self._recording and not self._video_saved:
                self._qpos_history.append(self._data.qpos.copy())
                return
            self._render_counter += 1
            if self._render_counter % 3 == 0:
                self._renderer.update_scene(self._data, camera=self._camera_name)
                self._latest_rgb = self._renderer.render().copy()

    def _publish_state(self) -> None:
        now = self.get_clock().now().to_msg()

        with self._lock:
            qpos = self._data.qpos.copy()
            qvel = self._data.qvel.copy()
            cube_pos = self._data.body("target_cube").xpos.copy()
            cube_quat = self._data.body("target_cube").xquat.copy()

        js_msg = JointState()
        js_msg.header.stamp = now
        js_msg.name = list(ALL_JOINT_NAMES)
        js_msg.position = [float(qpos[self._joint_qpos_adr[n]]) for n in ALL_JOINT_NAMES]
        js_msg.velocity = [float(qvel[self._joint_qvel_adr[n]]) for n in ALL_JOINT_NAMES]
        self._joint_state_pub.publish(js_msg)

        cube_base = cube_pos - self._base_link_pos
        cube_msg = PoseStamped()
        cube_msg.header.stamp = now
        cube_msg.header.frame_id = "base_link"
        cube_msg.pose.position.x = float(cube_base[0])
        cube_msg.pose.position.y = float(cube_base[1])
        cube_msg.pose.position.z = float(cube_base[2])
        cube_msg.pose.orientation.w = float(cube_quat[0])
        cube_msg.pose.orientation.x = float(cube_quat[1])
        cube_msg.pose.orientation.y = float(cube_quat[2])
        cube_msg.pose.orientation.z = float(cube_quat[3])
        self._cube_pose_pub.publish(cube_msg)

        target_msg = PoseStamped()
        target_msg.header.stamp = now
        target_msg.header.frame_id = "base_link"
        target_msg.pose.position.x = float(self._place_in_base[0])
        target_msg.pose.position.y = float(self._place_in_base[1])
        target_msg.pose.position.z = float(self._place_in_base[2])
        target_msg.pose.orientation.w = 1.0
        self._target_pose_pub.publish(target_msg)

        rgb = self._latest_rgb
        if rgb is not None:
            img_msg = Image()
            img_msg.header.stamp = now
            img_msg.header.frame_id = "front_cam"
            img_msg.height = rgb.shape[0]
            img_msg.width = rgb.shape[1]
            img_msg.encoding = "rgb8"
            img_msg.is_bigendian = False
            img_msg.step = rgb.shape[1] * 3
            img_msg.data = rgb.tobytes()
            self._image_pub.publish(img_msg)

    def destroy_node(self):
        self._renderer.close()
        super().destroy_node()


def _load_config() -> dict:
    with open(COMPONENT_DIR / "config.yml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = _load_config()
    sim_cfg = config["simulation"]
    ds_cfg = config.get("dataset", {})

    parser = argparse.ArgumentParser(description="MuJoCo ROS2 bridge for X3Plus")
    parser.add_argument("--model", default=sim_cfg["model_path"])
    parser.add_argument("--camera", default=sim_cfg.get("camera_name", "front_cam"))
    parser.add_argument(
        "--place-pos", nargs=3, type=float,
        default=ds_cfg.get("place_position", [0.15, -0.15, 0.39]),
    )
    parser.add_argument(
        "--resolution", nargs=2, type=int,
        default=sim_cfg.get("camera_resolution", [256, 256]),
    )
    parser.add_argument("--record-video", type=str, default=None,
                        help="Path to save MP4 video on task_complete")
    parser.add_argument("--video-fps", type=int,
                        default=sim_cfg.get("video_fps", 30))
    args = parser.parse_args()

    rclpy.init()
    node = MuJoCoBridgeNode(
        model_path=args.model,
        place_position=args.place_pos,
        camera_name=args.camera,
        resolution=tuple(args.resolution),
        record_video=args.record_video,
        video_fps=args.video_fps,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
