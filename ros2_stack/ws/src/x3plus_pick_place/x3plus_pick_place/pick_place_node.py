"""ROS2 node that orchestrates X3Plus pick-and-place via MoveIt execution.

Motion is decomposed into three kinematic groups driven by a YAML-configured
behavior tree:

  - arm_joint1          → base yaw (horizontal rotation)
  - arm_joint2/3/4      → sagittal-plane reach + height
  - arm_joint5          → wrist roll (orientation alignment)
  - grip_joint          → finger open / close

Position goals are still solved with constrained analytical IK, but arm
motions are executed through MoveIt2 + ros2_control so the simulator receives
time-parameterized joint trajectories instead of coarse direct setpoints.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
import yaml
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from ament_index_python.packages import get_package_share_directory

import py_trees

from x3plus_pick_place.bt_nodes import build_tree_from_config

ARM_JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 6)]

CONVERGENCE_TOL = 0.05
CONVERGENCE_TIMEOUT = 20.0
MOVEIT_ALLOWED_PLANNING_TIME = 5.0
MOVEIT_NUM_PLANNING_ATTEMPTS = 10
MOVEIT_VEL_SCALE = 0.2
MOVEIT_ACC_SCALE = 0.2
DIRECT_CMD_PERIOD = 0.05


def quat_to_yaw(w: float, x: float, y: float, z: float) -> float:
    """Extract yaw (rotation about Z) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class BlockState:
    """Tracked state for a single block (position + orientation)."""

    _MIN_VALID_Z = -0.15

    def __init__(self) -> None:
        self.position: np.ndarray | None = None
        self.yaw: float | None = None

    def update(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if p.z < self._MIN_VALID_Z:
            return
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

        self._move_group_client = ActionClient(node, MoveGroup, "move_action")
        self._joint_cmd_pub = node.create_publisher(
            JointState, "/joint_command", 10,
        )
        self._gripper_pub = node.create_publisher(
            Float64MultiArray, "/x3plus_gripper_controller/commands", 10,
        )
        self._done_pub = node.create_publisher(Bool, "/task_complete", 10)

        self.yellow = BlockState()
        self.red = BlockState()
        self.yellow_rest_pos: np.ndarray | None = None
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

    def have_arm_state(self) -> bool:
        return all(name in self._current_joints for name in [*ARM_JOINT_NAMES, "grip_joint"])

    def _send_move_group_goal(self, goal_msg: MoveGroup.Goal) -> bool:
        """Send a MoveGroup goal and wait for the execution result."""
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self._logger.error("MoveGroup action server not available")
            return False

        result_event = threading.Event()
        result_holder = [False]

        def result_cb(future):
            result = future.result()
            if result and result.result.error_code.val == 1:
                result_holder[0] = True
            else:
                error_val = result.result.error_code.val if result else "timeout"
                self._logger.error(f"MoveGroup failed with error code: {error_val}")
            result_event.set()

        def goal_response_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._logger.error("MoveGroup goal rejected")
                result_event.set()
                return
            goal_handle.get_result_async().add_done_callback(result_cb)

        send_future = self._move_group_client.send_goal_async(goal_msg)
        send_future.add_done_callback(goal_response_cb)

        if not result_event.wait(timeout=90.0):
            self._logger.error("Timed out waiting for MoveGroup result")
            return False
        return result_holder[0]

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

    def _execute_direct_arm_goal(self, goal: list[float], label: str) -> bool:
        self._logger.warn(f"{label}: falling back to direct joint command")
        deadline = time.monotonic() + CONVERGENCE_TIMEOUT
        while time.monotonic() < deadline:
            self._send_joint_command(ARM_JOINT_NAMES, goal)
            if self.wait_for_joint_convergence(
                goal,
                timeout=DIRECT_CMD_PERIOD + 0.1,
            ):
                self._logger.info(f"{label} OK (direct)")
                return True
            time.sleep(DIRECT_CMD_PERIOD)
        self._logger.error(f"{label} convergence FAILED (direct)")
        return False

    def _execute_arm_goal(self, goal: list[float], label: str) -> bool:
        if self.wait_for_joint_convergence(goal, timeout=0.2):
            self._logger.info(f"{label} already at goal")
            return True

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = "x3plus_arm"
        goal_msg.request.num_planning_attempts = MOVEIT_NUM_PLANNING_ATTEMPTS
        goal_msg.request.allowed_planning_time = MOVEIT_ALLOWED_PLANNING_TIME
        goal_msg.request.max_velocity_scaling_factor = MOVEIT_VEL_SCALE
        goal_msg.request.max_acceleration_scaling_factor = MOVEIT_ACC_SCALE

        constraints = Constraints()
        for name, pos in zip(ARM_JOINT_NAMES, goal):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraints)

        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 3

        self._logger.info(
            f"{label}: {[f'{c:.3f}' for c in self.get_arm_joints()]} → "
            f"{[f'{g:.3f}' for g in goal]}"
        )
        if not self._send_move_group_goal(goal_msg):
            return self._execute_direct_arm_goal(goal, label)
        if not self.wait_for_joint_convergence(goal):
            self._logger.warn(f"{label} convergence FAILED after MoveIt execution")
            return self._execute_direct_arm_goal(goal, label)
        self._logger.info(f"{label} OK")
        return True

    # ── Decomposed motion primitives ───────────────────────────────────────

    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """Move all 5 arm joints to target positions."""
        return self._execute_arm_goal(joint_positions, "MoveAll")

    def rotate_base(self, target_yaw: float) -> bool:
        """Rotate only arm_joint1 to face a direction."""
        goal = self.get_arm_joints()
        goal[0] = target_yaw
        return self._execute_arm_goal(goal, "RotateBase")

    def move_in_plane(self, q234: list[float]) -> bool:
        """Move joints 2, 3, 4 in the sagittal plane."""
        goal = self.get_arm_joints()
        goal[1:4] = q234
        return self._execute_arm_goal(goal, "MoveInPlane")

    def align_wrist(self, target_angle: float) -> bool:
        """Rotate arm_joint5 to align gripper with block orientation."""
        goal = self.get_arm_joints()
        goal[4] = target_angle
        return self._execute_arm_goal(goal, "AlignWrist")

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

    logger.info("Waiting for block poses and joint states from simulator ...")
    while rclpy.ok() and not (
        ctx.yellow.received and ctx.red.received and ctx.have_arm_state()
    ):
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
