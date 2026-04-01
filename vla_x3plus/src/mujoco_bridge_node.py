"""ROS2-to-MuJoCo bridge for the X3Plus arm (two-block pick-and-place).

A pure translator between ROS2 topics and the MuJoCo simulation. Receives
absolute joint position commands on /joint_command and drives the MuJoCo
position actuators directly. Publishes joint states, block poses (yellow
and red with orientation), and front-camera images.

On startup, both blocks are randomly placed on the table within configurable
bounds and with random yaw orientations.

Usage:
    source /opt/ros/jazzy/setup.bash
    source activate_env.sh vla_x3plus
    python -m src.mujoco_bridge_node
    python -m src.mujoco_bridge_node --record-video output/bt_pick_place.mp4
    python -m src.mujoco_bridge_node --seed 42
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import threading

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = COMPONENT_DIR.parent
PICK_PLACE_CONFIG_PATH = (
    REPO_ROOT
    / "ros2_stack/ws/src/x3plus_pick_place/config/pick_place_tree.yaml"
)

try:
    from x3plus_pick_place.ik_solver import is_in_orth_workspace
except ImportError:
    is_in_orth_workspace = None

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

TABLE_TOP_Z = 0.37
BLOCK_HALF_H = 0.014
BLOCK_ON_TABLE_Z = TABLE_TOP_Z + BLOCK_HALF_H

# Keep randomized placements away from the base where the current
# X3Plus MoveIt collision model becomes overly conservative.
BLOCK_X_RANGE = (0.14, 0.22)
BLOCK_Y_RANGE = (-0.12, 0.12)
BLOCK_MIN_SEPARATION = 0.06
ORTH_WORKSPACE_DEMO_YELLOW_BASE = np.array([0.237, -0.070, -0.067])
ORTH_WORKSPACE_DEMO_RED_BASE = np.array([0.238, 0.072, -0.067])
ORTH_WORKSPACE_DEMO_YAWS = (1.23, -2.70)


def _yaw_to_quat(yaw: float) -> np.ndarray:
    """Convert a yaw angle (rad) to MuJoCo quaternion [w, x, y, z]."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _load_pick_place_config() -> dict:
    with open(PICK_PLACE_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _compute_ee_z(
    block_z: float,
    height_offset: float,
    config: dict,
    stack_on_top: bool = False,
    pick_block_half_h: float = 0.0,
    place_block_half_h: float = 0.0,
) -> float:
    finger_offset = float(config["robot"]["finger_ee_offset_z"])
    if stack_on_top:
        target_surface_z = block_z + place_block_half_h + pick_block_half_h
        return target_surface_z - finger_offset + height_offset
    return block_z - finger_offset + height_offset


def _yellow_pick_targets(block_base_pos: np.ndarray, config: dict) -> list[np.ndarray]:
    movement = config["movement"]
    z = float(block_base_pos[2])
    height_offsets = [0.0, float(movement["approach_height"]), float(movement["lift_height"])]
    return [
        np.array([
            float(block_base_pos[0]),
            float(block_base_pos[1]),
            _compute_ee_z(z, height_offset, config),
        ])
        for height_offset in height_offsets
    ]


def _red_place_targets(block_base_pos: np.ndarray, config: dict) -> list[np.ndarray]:
    movement = config["movement"]
    blocks = config["blocks"]
    z = float(block_base_pos[2])
    height_offsets = [0.0, float(movement["approach_height"])]
    yellow_hh = float(blocks["yellow"]["half_height"])
    red_hh = float(blocks["red"]["half_height"])
    return [
        np.array([
            float(block_base_pos[0]),
            float(block_base_pos[1]),
            _compute_ee_z(
                z,
                height_offset,
                config,
                stack_on_top=True,
                pick_block_half_h=yellow_hh,
                place_block_half_h=red_hh,
            ),
        ])
        for height_offset in height_offsets
    ]


def _targets_in_orth_workspace(targets: list[np.ndarray]) -> bool:
    if is_in_orth_workspace is None:
        raise RuntimeError(
            "orth_workspace filtering requires the x3plus_pick_place package. "
            "Source /opt/ros/jazzy/setup.bash and ros2_stack/ws/install/setup.bash first."
        )
    return all(is_in_orth_workspace(target) for target in targets)


def _randomize_block_poses(
    rng: np.random.Generator,
    x_range: tuple[float, float] = BLOCK_X_RANGE,
    y_range: tuple[float, float] = BLOCK_Y_RANGE,
    min_sep: float = BLOCK_MIN_SEPARATION,
    base_link_pos: np.ndarray | None = None,
    pick_place_config: dict | None = None,
    orth_workspace_only: bool = False,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Return (yellow_pos, yellow_yaw, red_pos, red_yaw) in world frame."""
    if orth_workspace_only:
        if base_link_pos is None or pick_place_config is None:
            raise ValueError(
                "orth_workspace_only requires base_link_pos and pick_place_config"
            )
        yellow_base = ORTH_WORKSPACE_DEMO_YELLOW_BASE.copy()
        red_base = ORTH_WORKSPACE_DEMO_RED_BASE.copy()
        if (
            math.hypot(
                float(yellow_base[0] - red_base[0]),
                float(yellow_base[1] - red_base[1]),
            ) >= min_sep
            and _targets_in_orth_workspace(
                _yellow_pick_targets(yellow_base, pick_place_config)
            )
            and _targets_in_orth_workspace(
                _red_place_targets(red_base, pick_place_config)
            )
        ):
            yellow_pos = base_link_pos + yellow_base
            red_pos = base_link_pos + red_base
            return (
                yellow_pos,
                ORTH_WORKSPACE_DEMO_YAWS[0],
                red_pos,
                ORTH_WORKSPACE_DEMO_YAWS[1],
            )

    for _ in range(1000):
        yx = rng.uniform(x_range[0], x_range[1])
        yy = rng.uniform(y_range[0], y_range[1])
        rx = rng.uniform(x_range[0], x_range[1])
        ry = rng.uniform(y_range[0], y_range[1])
        if math.hypot(yx - rx, yy - ry) >= min_sep:
            yellow_pos = np.array([yx, yy, BLOCK_ON_TABLE_Z])
            red_pos = np.array([rx, ry, BLOCK_ON_TABLE_Z])
            if orth_workspace_only:
                yellow_base = yellow_pos - base_link_pos
                red_base = red_pos - base_link_pos
                if not _targets_in_orth_workspace(
                    _yellow_pick_targets(yellow_base, pick_place_config)
                ):
                    continue
                if not _targets_in_orth_workspace(
                    _red_place_targets(red_base, pick_place_config)
                ):
                    continue
            y_yaw = rng.uniform(-math.pi, math.pi)
            r_yaw = rng.uniform(-math.pi, math.pi)
            return (
                yellow_pos,
                y_yaw,
                red_pos,
                r_yaw,
            )
    raise RuntimeError("Could not place two blocks with sufficient separation")


class MuJoCoBridgeNode(Node):
    def __init__(
        self,
        model_path: str,
        camera_name: str = "front_cam",
        resolution: tuple[int, int] = (256, 256),
        record_video: str | None = None,
        video_fps: int = 30,
        seed: int | None = None,
        orth_workspace_only: bool = False,
    ):
        super().__init__("mujoco_bridge")

        xml_path = COMPONENT_DIR / model_path
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        h, w = resolution
        self._renderer = mujoco.Renderer(self._model, height=h, width=w)
        self._camera_name = camera_name
        self._resolution = resolution

        self._record_video_path = record_video
        self._video_fps = video_fps
        self._video_frames: list[np.ndarray] = []
        self._recording = record_video is not None
        self._video_saved = False
        self._saving_video = threading.Lock()

        snap_dt = self._model.opt.timestep * self._SIM_SUBSTEPS
        self._snap_dt = snap_dt
        self._video_frame_interval = max(1, round(1.0 / (video_fps * snap_dt)))
        self._sim_step_count = 0

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

        self._yellow_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "yellow_block",
        )
        self._red_body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, "red_block",
        )

        self._yellow_qpos_adr = self._model.joint("yellow_joint").qposadr[0]
        self._red_qpos_adr = self._model.joint("red_joint").qposadr[0]

        init_key_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_KEY, "init",
        )
        if init_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, init_key_id)

        mujoco.mj_forward(self._model, self._data)
        self._base_link_pos = self._data.body("base_link").xpos.copy()

        rng = np.random.default_rng(seed)
        pick_place_config = _load_pick_place_config() if orth_workspace_only else None
        y_pos, y_yaw, r_pos, r_yaw = _randomize_block_poses(
            rng,
            base_link_pos=self._base_link_pos,
            pick_place_config=pick_place_config,
            orth_workspace_only=orth_workspace_only,
        )
        self._set_block_qpos(self._yellow_qpos_adr, y_pos, y_yaw)
        self._set_block_qpos(self._red_qpos_adr, r_pos, r_yaw)

        _GRIPPER_OPEN_CTRL = -1.54
        self._data.ctrl[self._actuator_ctrl_idx["grip_joint"]] = _GRIPPER_OPEN_CTRL

        mujoco.mj_forward(self._model, self._data)

        self._lock = threading.Lock()
        self._latest_rgb: np.ndarray | None = None
        self._render_counter = 0

        cb = ReentrantCallbackGroup()

        self._joint_state_pub = self.create_publisher(JointState, "/joint_states", 10)
        self._yellow_pose_pub = self.create_publisher(PoseStamped, "/yellow_block_pose", 10)
        self._red_pose_pub = self.create_publisher(PoseStamped, "/red_block_pose", 10)
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
        if orth_workspace_only:
            self.get_logger().info("Constraining block spawn to orth_workspace")

        y_base = y_pos - self._base_link_pos
        r_base = r_pos - self._base_link_pos
        self.get_logger().info(
            f"MuJoCo bridge ready — model={xml_path.name}  "
            f"base_link@world={self._base_link_pos.tolist()}  "
            f"yellow_base=[{y_base[0]:.3f},{y_base[1]:.3f},{y_base[2]:.3f}] yaw={y_yaw:.2f}  "
            f"red_base=[{r_base[0]:.3f},{r_base[1]:.3f},{r_base[2]:.3f}] yaw={r_yaw:.2f}"
        )

    def _set_block_qpos(self, qpos_adr: int, pos: np.ndarray, yaw: float) -> None:
        quat = _yaw_to_quat(yaw)
        self._data.qpos[qpos_adr:qpos_adr + 3] = pos
        self._data.qpos[qpos_adr + 3:qpos_adr + 7] = quat

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

        n_frames = len(self._video_frames)
        if n_frames == 0:
            self.get_logger().warn("No video frames captured, skipping video")
            return

        sim_duration = self._sim_step_count * self._snap_dt
        self.get_logger().info(
            f"Writing {n_frames} pre-rendered frames "
            f"({sim_duration:.1f}s sim time) -> {path}"
        )

        imageio.mimwrite(
            str(path),
            self._video_frames,
            fps=self._video_fps,
            quality=10,
            codec="libx264",
            pixelformat="yuv420p",
            output_params=["-preset", "fast", "-crf", "17"],
        )
        self._video_frames.clear()
        self.get_logger().info(f"Video saved: {path}")
        # Exit the executor cleanly once the recording is flushed to disk.
        rclpy.try_shutdown()

    _SIM_SUBSTEPS = 20

    def _step_sim(self) -> None:
        with self._lock:
            for _ in range(self._SIM_SUBSTEPS):
                mujoco.mj_step(self._model, self._data)
            if self._recording and not self._video_saved:
                self._sim_step_count += 1
                if self._sim_step_count % self._video_frame_interval == 0:
                    self._renderer.update_scene(
                        self._data, camera=self._camera_name,
                    )
                    self._video_frames.append(self._renderer.render().copy())
                return
            self._render_counter += 1
            if self._render_counter % 3 == 0:
                self._renderer.update_scene(self._data, camera=self._camera_name)
                self._latest_rgb = self._renderer.render().copy()

    def _make_pose_msg(self, stamp, body_id) -> PoseStamped:
        pos = self._data.body(body_id).xpos.copy()
        quat = self._data.body(body_id).xquat.copy()
        base = pos - self._base_link_pos
        msg = PoseStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(base[0])
        msg.pose.position.y = float(base[1])
        msg.pose.position.z = float(base[2])
        msg.pose.orientation.w = float(quat[0])
        msg.pose.orientation.x = float(quat[1])
        msg.pose.orientation.y = float(quat[2])
        msg.pose.orientation.z = float(quat[3])
        return msg

    def _publish_state(self) -> None:
        now = self.get_clock().now().to_msg()

        with self._lock:
            qpos = self._data.qpos.copy()
            qvel = self._data.qvel.copy()
            yellow_msg = self._make_pose_msg(now, self._yellow_body_id)
            red_msg = self._make_pose_msg(now, self._red_body_id)

        js_msg = JointState()
        js_msg.header.stamp = now
        js_msg.name = list(ALL_JOINT_NAMES)
        js_msg.position = [float(qpos[self._joint_qpos_adr[n]]) for n in ALL_JOINT_NAMES]
        js_msg.velocity = [float(qvel[self._joint_qvel_adr[n]]) for n in ALL_JOINT_NAMES]
        self._joint_state_pub.publish(js_msg)

        self._yellow_pose_pub.publish(yellow_msg)
        self._red_pose_pub.publish(red_msg)

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

    parser = argparse.ArgumentParser(description="MuJoCo ROS2 bridge for X3Plus")
    parser.add_argument("--model", default=sim_cfg["model_path"])
    parser.add_argument("--camera", default=sim_cfg.get("camera_name", "front_cam"))
    parser.add_argument(
        "--resolution", nargs=2, type=int,
        default=sim_cfg.get("camera_resolution", [256, 256]),
    )
    parser.add_argument("--record-video", type=str, default=None,
                        help="Path to save MP4 video on task_complete")
    parser.add_argument("--video-fps", type=int,
                        default=sim_cfg.get("video_fps", 30))
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for block placement (None = random)")
    parser.add_argument(
        "--orth-workspace-only",
        action="store_true",
        help="Spawn both blocks only at positions reachable with fixed top-down grasping",
    )
    args = parser.parse_args()

    rclpy.init()
    node = MuJoCoBridgeNode(
        model_path=args.model,
        camera_name=args.camera,
        resolution=tuple(args.resolution),
        record_video=args.record_video,
        video_fps=args.video_fps,
        seed=args.seed,
        orth_workspace_only=args.orth_workspace_only,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
