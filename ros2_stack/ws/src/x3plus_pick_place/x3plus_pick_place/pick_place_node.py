"""ROS2 node that orchestrates X3Plus pick-and-place via direct joint control + py_trees.

Position goals are converted to joint-space goals using a custom analytical
IK solver.  Joints are driven by publishing to /joint_command; the MuJoCo
bridge's position actuators handle the dynamics.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray

import py_trees

from x3plus_pick_place.bt_nodes import build_pick_place_tree
from x3plus_pick_place.ik_solver import forward_kinematics, solve_ik

ARM_JOINT_NAMES = [f"arm_joint{i}" for i in range(1, 6)]

MOVE_DURATION = 4.0
MOVE_HZ = 25.0
CONVERGENCE_TOL = 0.02
CONVERGENCE_TIMEOUT = 8.0


class PickPlaceContext:
    """Shared context between BT nodes and the ROS2 node."""

    def __init__(self, node: Node):
        self._node = node
        self._logger = node.get_logger()

        self._joint_cmd_pub = node.create_publisher(
            JointState, "/joint_command", 10,
        )
        self._gripper_pub = node.create_publisher(
            Float64MultiArray, "/x3plus_gripper_controller/commands", 10,
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

    # ── Verification helpers ──────────────────────────────────────────────

    def get_arm_joints(self) -> list[float]:
        return [self._current_joints.get(f"arm_joint{i}", 0.0) for i in range(1, 6)]

    def get_gripper_position(self) -> float:
        return self._current_joints.get("grip_joint", 0.0)

    def get_cube_position(self) -> np.ndarray | None:
        if self.cube_pose is not None:
            return self.cube_pose[:3].copy()
        return None

    def wait_for_joint_convergence(
        self, target: list[float], timeout: float = CONVERGENCE_TIMEOUT,
        tolerance: float = CONVERGENCE_TOL,
    ) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            current = self.get_arm_joints()
            max_err = max(abs(c - t) for c, t in zip(current, target))
            if max_err < tolerance:
                return True
            time.sleep(0.05)
        current = self.get_arm_joints()
        for i, (c, t) in enumerate(zip(current, target)):
            if abs(c - t) > tolerance:
                self._logger.warn(
                    f"arm_joint{i+1} convergence err: |{c:.4f} - {t:.4f}| = {abs(c-t):.4f} rad"
                )
        return False

    # ── Motion primitives ─────────────────────────────────────────────────

    def _send_joint_command(self, positions: list[float]) -> None:
        cmd = JointState()
        cmd.name = list(ARM_JOINT_NAMES)
        cmd.position = [float(p) for p in positions]
        self._joint_cmd_pub.publish(cmd)

    def move_to_joints(self, joint_positions: list[float]) -> bool:
        """Linearly interpolate from current joints to target and drive
        via /joint_command.  Then hold and verify convergence."""
        current = self.get_arm_joints()
        start = np.array(current)
        goal = np.array(joint_positions)

        n_steps = max(1, int(MOVE_DURATION * MOVE_HZ))
        dt = 1.0 / MOVE_HZ

        self._logger.info(
            f"Direct move: {[f'{c:.3f}' for c in current]} → "
            f"{[f'{g:.3f}' for g in joint_positions]}"
        )

        for step in range(1, n_steps + 1):
            t = step / n_steps
            t_smooth = 3 * t * t - 2 * t * t * t
            interp = start + (goal - start) * t_smooth
            self._send_joint_command(interp.tolist())
            time.sleep(dt)

        for _ in range(30):
            self._send_joint_command(joint_positions)
            time.sleep(0.05)

        if not self.wait_for_joint_convergence(joint_positions):
            self._logger.error("Joint convergence FAILED after direct move")
            return False

        self._logger.info("Joint convergence verified")
        return True

    def move_to_pose(self, position: list[float]) -> bool:
        """Solve analytical IK then execute as a direct joint move."""
        current = [
            self._current_joints.get(f"arm_joint{i}", 0.0)
            for i in range(1, 6)
        ]
        solution = solve_ik(position, q5=0.0, current=current)
        if solution is None:
            self._logger.error(
                f"IK: no solution for position {position}"
            )
            return False

        fk = forward_kinematics(solution)
        err = np.linalg.norm(fk - np.asarray(position))
        self._logger.info(
            f"IK → joints [{', '.join(f'{q:.4f}' for q in solution)}]  "
            f"FK verify [{fk[0]:.4f}, {fk[1]:.4f}, {fk[2]:.4f}]  "
            f"err {err*1000:.2f} mm"
        )
        if not self.move_to_joints(solution):
            return False

        actual_joints = self.get_arm_joints()
        fk_actual = forward_kinematics(actual_joints)
        ee_err = np.linalg.norm(fk_actual - np.asarray(position))
        self._logger.info(
            f"EE post-move: actual [{fk_actual[0]:.4f}, {fk_actual[1]:.4f}, {fk_actual[2]:.4f}]  "
            f"err {ee_err*1000:.1f} mm"
        )
        if ee_err > 0.02:
            self._logger.error(f"EE error {ee_err*1000:.1f} mm exceeds 20 mm limit")
            return False
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

    logger.info(f"Cube pose (base_link): {ctx.cube_pose[:3]}")
    logger.info(f"Target pose (base_link): {ctx.target_pose}")

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
            logger.error("BehaviorTree FAILED — aborting")
            break

        time.sleep(1.0 / rate)

    logger.info("Pick-and-place node shutting down")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
