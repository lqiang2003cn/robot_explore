"""ROS2 node that orchestrates X3Plus pick-and-place via decomposed motion.

Motion is decomposed into three kinematic groups driven by a YAML-configured
behavior tree:

  - arm_joint1          → base yaw (horizontal rotation)
  - arm_joint2/3/4      → sagittal-plane reach + height
  - arm_joint5          → wrist roll (orientation alignment)
  - grip_joint          → finger open / close

Position goals are solved with constrained analytical IK.  Joints are driven
by publishing to /joint_command; the MuJoCo bridge's position actuators
handle the dynamics.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from ament_index_python.packages import get_package_share_directory

import py_trees

from x3plus_pick_place.bt_nodes import build_tree_from_config

ARM_JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 6)]

MOVE_DURATION = 2.0
MOVE_HZ = 25.0
CONVERGENCE_TOL = 0.05
CONVERGENCE_TIMEOUT = 10.0


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class BlockState:
    """Tracked state for a single block (position + orientation)."""

    def __init__(self) -> None:
        self.position: np.ndarray | None = None
        self.yaw: float | None = None

    def update(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        self.position = np.array([p.x, p.y, p.z])
        self.yaw = quat_to_yaw(o.w, o.x, o.y, o.z)

    @property
    def received(self) -> bool:
        return self.position is not None


class PickPlaceContext:
    """Shared context between BT nodes and the ROS2 node."""

    def __init__(self, node: Node, config: dict[str, Any]):
        self._node = node
        self._logger = node.get_logger()
        self.config = config

        self._joint_cmd_pub = node.create_publisher(
            JointState, "/joint_command", 10,
        )
        self._gripper_pub = node.create_publisher(
            Float64MultiArray, "/x3plus_gripper_controller/commands", 10,
        )
        self._done_pub = node.create_publisher(Bool, "/task_complete", 10)

        self.yellow = BlockState()
        self.red = BlockState()
        self._current_joints: dict[str, float] = {}

    def get_block(self, name: str) -> BlockState:
        if name == "yellow":
            return self.yellow
        elif name == "red":
            return self.red
        raise ValueError(f"Unknown block: {name}")

    # ── Joint state helpers ────────────────────────────────────────────────

    def update_joints(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._current_joints[name] = pos

    def get_arm_joints(self) -> list[float]:
        return [self._current_joints.get(f"arm_joint{i}", 0.0) for i in range(1, 6)]

    def get_gripper_position(self) -> float:
        return self._current_joints.get("grip_joint", 0.0)

    def wait_for_joint_convergence(
        self,
        target: list[float],
        joint_names: list[str] | None = None,
        timeout: float = CONVERGENCE_TIMEOUT,
        tolerance: float = CONVERGENCE_TOL,
    ) -> bool:
        if joint_names is None:
            joint_names = list(ARM_JOINT_NAMES)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            errs = [
                abs(self._current_joints.get(n, 0.0) - t)
                for n, t in zip(joint_names, target)
            ]
            if max(errs) < tolerance:
                return True
            time.sleep(0.02)
        for n, t in zip(joint_names, target):
            c = self._current_joints.get(n, 0.0)
            if abs(c - t) > tolerance:
                self._logger.warn(
                    f"{n} convergence err: |{c:.4f} - {t:.4f}| = {abs(c-t):.4f} rad"
                )
        return False

    # ── Low-level joint command ────────────────────────────────────────────

    def _send_joint_command(self, names: list[str], positions: list[float]) -> None:
        cmd = JointState()
        cmd.name = list(names)
        cmd.position = [float(p) for p in positions]
        self._joint_cmd_pub.publish(cmd)

    def _interpolate_joints(
        self,
        names: list[str],
        start: np.ndarray,
        goal: np.ndarray,
        duration: float = MOVE_DURATION,
    ) -> None:
        n_steps = max(1, int(duration * MOVE_HZ))
        dt = 1.0 / MOVE_HZ
        t0 = time.monotonic()
        for step in range(1, n_steps + 1):
            t = step / n_steps
            t_smooth = 3 * t * t - 2 * t * t * t
            interp = start + (goal - start) * t_smooth
            self._send_joint_command(names, interp.tolist())
            target_time = t0 + step * dt
            remaining = target_time - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self._send_joint_command(names, goal.tolist())

    # ── Decomposed motion primitives ───────────────────────────────────────

    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """Move all 5 arm joints to target positions."""
        current = np.array(self.get_arm_joints())
        goal = np.array(joint_positions)
        self._logger.info(
            f"MoveAll: {[f'{c:.3f}' for c in current]} → "
            f"{[f'{g:.3f}' for g in joint_positions]}"
        )
        self._interpolate_joints(list(ARM_JOINT_NAMES), current, goal)
        if not self.wait_for_joint_convergence(joint_positions):
            self._logger.error("Joint convergence FAILED (move_to_joints)")
            return False
        self._logger.info("Joint convergence verified (move_to_joints)")
        return True

    def rotate_base(self, target_yaw: float) -> bool:
        """Rotate only arm_joint1 to face a direction."""
        current = self._current_joints.get("arm_joint1", 0.0)
        self._logger.info(f"RotateBase: {current:.3f} → {target_yaw:.3f}")
        self._interpolate_joints(
            ["arm_joint1"],
            np.array([current]),
            np.array([target_yaw]),
            duration=1.0,
        )
        if not self.wait_for_joint_convergence(
            [target_yaw], ["arm_joint1"],
        ):
            self._logger.error("RotateBase convergence FAILED")
            return False
        self._logger.info("RotateBase OK")
        return True

    def move_in_plane(self, q234: list[float]) -> bool:
        """Move joints 2, 3, 4 in the sagittal plane."""
        names = ["arm_joint2", "arm_joint3", "arm_joint4"]
        current = np.array([self._current_joints.get(n, 0.0) for n in names])
        goal = np.array(q234)
        self._logger.info(
            f"MoveInPlane: {[f'{c:.3f}' for c in current]} → "
            f"{[f'{g:.3f}' for g in q234]}"
        )
        self._interpolate_joints(names, current, goal)
        if not self.wait_for_joint_convergence(q234, names):
            self._logger.error("MoveInPlane convergence FAILED")
            return False
        self._logger.info("MoveInPlane OK")
        return True

    def align_wrist(self, target_angle: float) -> bool:
        """Rotate arm_joint5 to align gripper with block orientation."""
        current = self._current_joints.get("arm_joint5", 0.0)
        self._logger.info(f"AlignWrist: {current:.3f} → {target_angle:.3f}")
        self._interpolate_joints(
            ["arm_joint5"],
            np.array([current]),
            np.array([target_angle]),
            duration=1.0,
        )
        if not self.wait_for_joint_convergence(
            [target_angle], ["arm_joint5"],
        ):
            self._logger.error("AlignWrist convergence FAILED")
            return False
        self._logger.info("AlignWrist OK")
        return True

    def set_gripper(self, position: float) -> None:
        msg = Float64MultiArray()
        msg.data = [position]
        self._gripper_pub.publish(msg)

        cmd = JointState()
        cmd.name = ["grip_joint"]
        cmd.position = [position]
        self._joint_cmd_pub.publish(cmd)

    def signal_done(self) -> None:
        msg = Bool()
        msg.data = True
        for _ in range(10):
            self._done_pub.publish(msg)
            time.sleep(0.05)


def _load_config(node: Node) -> dict[str, Any]:
    share = get_package_share_directory("x3plus_pick_place")
    config_path = f"{share}/config/pick_place_tree.yaml"
    node.get_logger().info(f"Loading BT config from {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    rclpy.init()

    node = rclpy.create_node("x3plus_pick_place_orchestrator")
    logger = node.get_logger()

    config = _load_config(node)
    ctx = PickPlaceContext(node, config)

    cb_group = ReentrantCallbackGroup()

    yellow_topic = config["blocks"]["yellow"]["topic"]
    red_topic = config["blocks"]["red"]["topic"]

    node.create_subscription(
        PoseStamped, yellow_topic, ctx.yellow.update, 10,
        callback_group=cb_group,
    )
    node.create_subscription(
        PoseStamped, red_topic, ctx.red.update, 10,
        callback_group=cb_group,
    )
    node.create_subscription(
        JointState, "/joint_states", ctx.update_joints, 10,
        callback_group=cb_group,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    logger.info("Waiting for block poses from simulator ...")
    while rclpy.ok() and not (ctx.yellow.received and ctx.red.received):
        time.sleep(0.1)

    if not rclpy.ok():
        return

    logger.info(
        f"Yellow block: pos={ctx.yellow.position} yaw={ctx.yellow.yaw:.2f}"
    )
    logger.info(
        f"Red block: pos={ctx.red.position} yaw={ctx.red.yaw:.2f}"
    )

    bb = py_trees.blackboard.Client(name="Main")
    bb.register_key(key="/ctx", access=py_trees.common.Access.WRITE)
    bb.set("/ctx", ctx)

    tree = build_tree_from_config(config)
    tree.setup_with_descendants()

    logger.info("Running BehaviorTree pick-and-place sequence ...")

    rate = 10.0
    while rclpy.ok():
        tree.tick_once()
        status = tree.status

        if status == py_trees.common.Status.SUCCESS:
            logger.info("BehaviorTree completed successfully!")
            break
        elif status == py_trees.common.Status.FAILURE:
            logger.error("BehaviorTree FAILED — aborting")
            break

        time.sleep(1.0 / rate)

    logger.info("Pick-and-place node shutting down")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
