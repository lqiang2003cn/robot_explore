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
"""

from __future__ import annotations

import argparse
import os
import pathlib
import threading

os.environ.setdefault("MUJOCO_GL", "egl")

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

        self._lock = threading.Lock()

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

        self.get_logger().info(
            f"MuJoCo bridge ready — model={xml_path.name}  "
            f"place={self._place_position.tolist()}"
        )

    def _on_joint_command(self, msg: JointState) -> None:
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                idx = self._actuator_ctrl_idx.get(name)
                if idx is not None:
                    self._data.ctrl[idx] = pos

    def _on_task_complete(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.get_logger().info("Received task_complete — resetting environment")
        with self._lock:
            mujoco.mj_resetData(self._model, self._data)
            self._data.ctrl[:] = 0.0
            mujoco.mj_forward(self._model, self._data)

    def _step_sim(self) -> None:
        with self._lock:
            mujoco.mj_step(self._model, self._data)

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

        cube_msg = PoseStamped()
        cube_msg.header.stamp = now
        cube_msg.header.frame_id = "world"
        cube_msg.pose.position.x = float(cube_pos[0])
        cube_msg.pose.position.y = float(cube_pos[1])
        cube_msg.pose.position.z = float(cube_pos[2])
        cube_msg.pose.orientation.w = float(cube_quat[0])
        cube_msg.pose.orientation.x = float(cube_quat[1])
        cube_msg.pose.orientation.y = float(cube_quat[2])
        cube_msg.pose.orientation.z = float(cube_quat[3])
        self._cube_pose_pub.publish(cube_msg)

        target_msg = PoseStamped()
        target_msg.header.stamp = now
        target_msg.header.frame_id = "world"
        target_msg.pose.position.x = float(self._place_position[0])
        target_msg.pose.position.y = float(self._place_position[1])
        target_msg.pose.position.z = float(self._place_position[2])
        target_msg.pose.orientation.w = 1.0
        self._target_pose_pub.publish(target_msg)

        with self._lock:
            self._renderer.update_scene(self._data, camera=self._camera_name)
            rgb = self._renderer.render().copy()

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
    args = parser.parse_args()

    rclpy.init()
    node = MuJoCoBridgeNode(
        model_path=args.model,
        place_position=args.place_pos,
        camera_name=args.camera,
        resolution=tuple(args.resolution),
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
