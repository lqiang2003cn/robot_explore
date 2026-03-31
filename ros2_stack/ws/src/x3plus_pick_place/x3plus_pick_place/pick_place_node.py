"""ROS2 node that orchestrates X3Plus pick-and-place via MoveGroup action + py_trees.

Uses the MoveGroup action server for trajectory planning with position-only
Cartesian goals (5-DOF arm cannot satisfy arbitrary orientations). Gripper
is controlled via /x3plus_gripper_controller/commands.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningOptions,
    PositionConstraint,
)
from shape_msgs.msg import SolidPrimitive

import py_trees

from x3plus_pick_place.bt_nodes import build_pick_place_tree

ARM_JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 6)]


class PickPlaceContext:
    """Shared context between BT nodes and the ROS2 node."""

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        self._move_group_client = ActionClient(node, MoveGroup, "move_action")
        self._gripper_pub = node.create_publisher(
            Float64MultiArray, "/x3plus_gripper_controller/commands", 10,
        )
        self._joint_cmd_pub = node.create_publisher(
            JointState, "/joint_command", 10,
        )
        self._done_pub = node.create_publisher(Bool, "/task_complete", 10)

        self.cube_pose: np.ndarray | None = None
        self.target_pose: np.ndarray | None = None
        self._current_joints: dict[str, float] = {}

    def update_cube_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        o = msg.pose.orientation
        self.cube_pose = np.array([p.x, p.y, p.z, o.w, o.x, o.y, o.z])

    def update_target_pose(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        self.target_pose = np.array([p.x, p.y, p.z])

    def update_joints(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            self._current_joints[name] = pos

    def _send_move_group_goal(self, goal_msg: MoveGroup.Goal) -> bool:
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self._logger.error("MoveGroup action server not available")
            return False

        result_event = threading.Event()
        result_holder = [None]

        def goal_response_cb(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                self._logger.error("MoveGroup goal rejected")
                result_holder[0] = False
                result_event.set()
                return
            self._logger.info("Goal accepted, waiting for result...")
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(result_cb)

        def result_cb(future):
            result = future.result()
            if result and result.result.error_code.val == 1:
                self._logger.info("Motion completed successfully")
                result_holder[0] = True
            else:
                error_val = result.result.error_code.val if result else "timeout"
                self._logger.error(f"Motion failed with error code: {error_val}")
                result_holder[0] = False
            result_event.set()

        send_future = self._move_group_client.send_goal_async(goal_msg)
        send_future.add_done_callback(goal_response_cb)

        if not result_event.wait(timeout=60.0):
            self._logger.error("Timed out waiting for MoveGroup result")
            return False

        return result_holder[0]

    def move_to_pose(self, position: list[float]) -> bool:
        """Plan and execute a position-only Cartesian goal (no orientation constraint)."""
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "base_link"
        goal_pose.pose.position.x = position[0]
        goal_pose.pose.position.y = position[1]
        goal_pose.pose.position.z = position[2]
        goal_pose.pose.orientation.w = 1.0

        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = "x3plus_arm"
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = "base_link"
        pos_constraint.link_name = "arm_link5"
        bounding = SolidPrimitive()
        bounding.type = SolidPrimitive.SPHERE
        bounding.dimensions = [0.01]
        pos_constraint.constraint_region.primitives.append(bounding)
        pos_constraint.constraint_region.primitive_poses.append(goal_pose.pose)
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        goal_msg.request.goal_constraints.append(constraints)

        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 3

        self._logger.info(f"Sending position-only MoveGroup goal: pos={position}")
        return self._send_move_group_goal(goal_msg)

    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """Plan and execute a joint-space goal via MoveGroup action."""
        goal_msg = MoveGroup.Goal()
        goal_msg.request = MotionPlanRequest()
        goal_msg.request.group_name = "x3plus_arm"
        goal_msg.request.num_planning_attempts = 10
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.max_velocity_scaling_factor = 0.3
        goal_msg.request.max_acceleration_scaling_factor = 0.3

        constraints = Constraints()
        for name, pos in zip(ARM_JOINT_NAMES, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = pos
            jc.tolerance_above = 0.01
            jc.tolerance_below = 0.01
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)
        goal_msg.request.goal_constraints.append(constraints)

        goal_msg.planning_options = PlanningOptions()
        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.replan = True
        goal_msg.planning_options.replan_attempts = 3

        self._logger.info("Sending MoveGroup joint goal")
        return self._send_move_group_goal(goal_msg)

    def set_gripper(self, position: float) -> None:
        """Publish grip_joint target via ros2_control and direct topic."""
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


def main():
    rclpy.init()

    node = rclpy.create_node("x3plus_pick_place_orchestrator")
    logger = node.get_logger()

    ctx = PickPlaceContext(node)

    cb_group = ReentrantCallbackGroup()
    node.create_subscription(
        PoseStamped, "/cube_pose", ctx.update_cube_pose, 10,
        callback_group=cb_group,
    )
    node.create_subscription(
        PoseStamped, "/target_place_pose", ctx.update_target_pose, 10,
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

    logger.info("Waiting for cube and target poses from simulator ...")
    while rclpy.ok() and (ctx.cube_pose is None or ctx.target_pose is None):
        time.sleep(0.1)

    if not rclpy.ok():
        return

    logger.info(f"Cube pose: {ctx.cube_pose[:3]}")
    logger.info(f"Target pose: {ctx.target_pose}")

    bb = py_trees.blackboard.Client(name="Main")
    bb.register_key(key="/ctx", access=py_trees.common.Access.WRITE)
    bb.set("/ctx", ctx)

    tree = build_pick_place_tree(
        cube_pos=ctx.cube_pose[:3].tolist(),
        place_pos=ctx.target_pose.tolist(),
    )
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
            logger.error("BehaviorTree failed -- aborting")
            break

        time.sleep(1.0 / rate)

    logger.info("Pick-and-place node shutting down")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
