"""py_trees BehaviorTree leaf nodes for X3Plus pick-and-place.

Adapted from panda_pick_place/bt_nodes.py for the 5-DOF X3Plus arm.
MoveToPose uses position-only goals (no orientation constraint).

Every phase has a definite success criterion checked at runtime:
  ready → pre-grasp → grasp → close gripper → lift → pre-place
  → place → open gripper → retreat → ready
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import py_trees

if TYPE_CHECKING:
    from x3plus_pick_place.pick_place_node import PickPlaceContext


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bb_ctx(node: py_trees.behaviour.Behaviour) -> "PickPlaceContext":
    client = node.attach_blackboard_client()
    client.register_key(key="/ctx", access=py_trees.common.Access.READ)
    return client.get("/ctx")


# ── Motion leaf nodes ────────────────────────────────────────────────────────

class MoveToPose(py_trees.behaviour.Behaviour):
    """Plan and execute a position-only Cartesian goal via MoveIt2.

    Success: MoveGroup reports success AND actual EE (FK of converged joints)
    is within 20 mm of the target position.
    """

    def __init__(self, name: str, target_position: list[float]):
        super().__init__(name)
        self._target_position = target_position
        self._started = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            self.logger.info(f"Planning to position {self._target_position}")
            success = self.ctx.move_to_pose(self._target_position)
            if success:
                self.logger.info(f"Motion + EE verified for {self.name}")
                return py_trees.common.Status.SUCCESS
            else:
                self.logger.error(f"Motion/verify FAILED for {self.name}")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class MoveToJoints(py_trees.behaviour.Behaviour):
    """Move to a named joint configuration.

    Success: MoveGroup reports success AND actual joints converge within
    0.05 rad of each target.
    """

    def __init__(self, name: str, joint_positions: list[float]):
        super().__init__(name)
        self._joint_positions = joint_positions
        self._started = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            self.logger.info(f"Moving to joint config: {self.name}")
            success = self.ctx.move_to_joints(self._joint_positions)
            if success:
                return py_trees.common.Status.SUCCESS
            else:
                self.logger.error(f"Joint motion FAILED for {self.name}")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class SetGripper(py_trees.behaviour.Behaviour):
    """Command the gripper and verify it converges.

    For closing (min_closed is set): success when gripper position goes below
    min_closed, meaning the fingers have closed enough to grip an object even
    if the object prevents reaching the full target.

    For opening (min_closed is None): success when within *tolerance* of target.
    """

    def __init__(
        self,
        name: str,
        position: float,
        timeout: float = 3.0,
        tolerance: float = 0.5,
        min_closed: float | None = None,
    ):
        super().__init__(name)
        self._position = position
        self._timeout = timeout
        self._tolerance = tolerance
        self._min_closed = min_closed
        self._start_time: float | None = None
        self._command_sent = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._command_sent = False
        self._start_time = None

    def update(self) -> py_trees.common.Status:
        if not self._command_sent:
            self.logger.info(f"Gripper → {self._position:.2f}")
            self.ctx.set_gripper(self._position)
            self._command_sent = True
            self._start_time = time.time()
            return py_trees.common.Status.RUNNING

        self.ctx.set_gripper(self._position)
        actual = self.ctx.get_gripper_position()

        if self._min_closed is not None:
            if actual >= self._min_closed:
                self.logger.info(
                    f"Gripper CLOSED OK: actual={actual:.3f} threshold={self._min_closed:.3f}"
                )
                return py_trees.common.Status.SUCCESS
        else:
            err = abs(actual - self._position)
            if err < self._tolerance:
                self.logger.info(
                    f"Gripper OK: actual={actual:.3f} target={self._position:.3f}"
                )
                return py_trees.common.Status.SUCCESS

        if time.time() - self._start_time > self._timeout:
            self.logger.error(
                f"Gripper TIMEOUT: actual={actual:.3f} target={self._position:.3f}"
            )
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING


# ── Wait / sensor nodes ─────────────────────────────────────────────────────

class WaitForSettle(py_trees.behaviour.Behaviour):
    """Wait a fixed duration to let physics settle."""

    def __init__(self, name: str, duration: float = 0.5):
        super().__init__(name)
        self._duration = duration
        self._start_time: float | None = None

    def initialise(self):
        self._start_time = time.time()

    def update(self) -> py_trees.common.Status:
        if time.time() - self._start_time >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class WaitForCubePose(py_trees.behaviour.Behaviour):
    """Wait until a cube pose has been received from the simulator."""

    def __init__(self, name: str = "WaitForCubePose"):
        super().__init__(name)

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def update(self) -> py_trees.common.Status:
        if self.ctx.cube_pose is not None:
            self.logger.info(f"Cube pose received: {self.ctx.cube_pose[:3]}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# ── Verification nodes ───────────────────────────────────────────────────────

class VerifyCubeGrasped(py_trees.behaviour.Behaviour):
    """Verify the cube is held by checking its Z is above a threshold.

    Called after lifting; if the cube stayed on the table, the grasp failed.
    """

    def __init__(self, name: str, min_cube_z: float, timeout: float = 2.0):
        super().__init__(name)
        self._min_z = min_cube_z
        self._timeout = timeout
        self._start_time: float | None = None

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._start_time = time.time()

    def update(self) -> py_trees.common.Status:
        pos = self.ctx.get_cube_position()
        if pos is not None and pos[2] >= self._min_z:
            self.logger.info(
                f"Cube grasped ✓  z={pos[2]:.4f} ≥ {self._min_z:.4f}"
            )
            return py_trees.common.Status.SUCCESS

        if time.time() - self._start_time > self._timeout:
            z = f"{pos[2]:.4f}" if pos is not None else "N/A"
            self.logger.error(
                f"Cube NOT grasped: z={z}, need ≥ {self._min_z:.4f}"
            )
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class VerifyCubePlaced(py_trees.behaviour.Behaviour):
    """Verify the cube ended up near the target XY after release.

    Success: horizontal distance from cube to target < tolerance.
    """

    def __init__(
        self,
        name: str,
        target_xy: list[float],
        tolerance: float = 0.05,
        timeout: float = 3.0,
    ):
        super().__init__(name)
        self._target_xy = np.array(target_xy[:2])
        self._tolerance = tolerance
        self._timeout = timeout
        self._start_time: float | None = None

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._start_time = time.time()

    def update(self) -> py_trees.common.Status:
        pos = self.ctx.get_cube_position()
        if pos is not None:
            err_xy = float(np.linalg.norm(pos[:2] - self._target_xy))
            if err_xy < self._tolerance:
                self.logger.info(
                    f"Cube placed ✓  XY err={err_xy*1000:.1f} mm"
                )
                return py_trees.common.Status.SUCCESS

        if time.time() - self._start_time > self._timeout:
            if pos is not None:
                err_xy = float(np.linalg.norm(pos[:2] - self._target_xy))
                self.logger.error(
                    f"Cube NOT at target: XY err={err_xy*1000:.1f} mm "
                    f"> {self._tolerance*1000:.0f} mm"
                )
            else:
                self.logger.error("Cube pose unavailable")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class SignalTaskComplete(py_trees.behaviour.Behaviour):
    """Publish task_complete=True to signal the simulator."""

    def __init__(self, name: str = "SignalTaskComplete"):
        super().__init__(name)

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def update(self) -> py_trees.common.Status:
        self.ctx.signal_done()
        self.logger.info("Task complete signal sent")
        return py_trees.common.Status.SUCCESS


# ── Tree builder ─────────────────────────────────────────────────────────────

FINGER_EE_OFFSET_Z = 0.039

def build_pick_place_tree(
    cube_pos: list[float],
    place_pos: list[float],
    gripper_open: float = -1.54,
    gripper_closed: float = 0.0,
    settle_time: float = 0.8,
    lift_height: float = 0.05,
) -> py_trees.behaviour.Behaviour:
    """Build the pick-and-place BehaviorTree for the X3Plus arm.

    The gripper fingers sit ~FINGER_EE_OFFSET_Z above the arm_link5 origin
    (IK end-effector) due to the wrist angle. EE targets are offset downward
    so that the physical fingers align with the cube/place heights.
    """
    grasp_ee_z = cube_pos[2] - FINGER_EE_OFFSET_Z
    place_ee_z = place_pos[2] - FINGER_EE_OFFSET_Z

    pre_grasp = [cube_pos[0], cube_pos[1], grasp_ee_z + lift_height]
    grasp     = [cube_pos[0], cube_pos[1], grasp_ee_z]
    lift      = [cube_pos[0], cube_pos[1], grasp_ee_z + lift_height]
    pre_place = [place_pos[0], place_pos[1], place_ee_z + lift_height]
    place     = [place_pos[0], place_pos[1], place_ee_z]

    ready_joints = [0.0, -0.5, 0.5, -0.5, 0.0]

    cube_lifted_min_z = cube_pos[2] + lift_height * 0.3

    root = py_trees.composites.Sequence("PickAndPlace", memory=True)
    root.add_children([
        WaitForCubePose("WaitForCubePose"),
        SetGripper("OpenGripperInit", gripper_open, timeout=6.0, tolerance=0.6),
        WaitForSettle("SettleInit", settle_time),

        MoveToJoints("MoveToReady", ready_joints),
        WaitForSettle("SettleReady", settle_time),

        MoveToPose("MoveToPreGrasp", pre_grasp),
        WaitForSettle("SettlePreGrasp", 0.5),

        MoveToPose("MoveToGrasp", grasp),
        WaitForSettle("SettleGrasp", 0.5),

        SetGripper("CloseGripper", gripper_closed, timeout=8.0, min_closed=-0.35),
        WaitForSettle("SettleGripperClose", settle_time),

        MoveToPose("LiftCube", lift),
        WaitForSettle("SettleLift", 0.8),
        VerifyCubeGrasped("VerifyLift", cube_lifted_min_z),

        MoveToPose("MoveToPrePlace", pre_place),
        WaitForSettle("SettlePrePlace", 0.5),
        VerifyCubeGrasped("VerifyTransit", cube_lifted_min_z),

        MoveToPose("MoveToPlace", place),
        WaitForSettle("SettlePlace", 0.5),

        SetGripper("OpenGripperRelease", gripper_open, timeout=6.0, tolerance=0.6),
        WaitForSettle("SettleRelease", settle_time),

        MoveToPose("Retreat", pre_place),
        WaitForSettle("SettleRetreat", 0.5),

        MoveToJoints("ReturnToReady", ready_joints),
        WaitForSettle("SettleFinal", settle_time),
        VerifyCubePlaced("VerifyPlacement", place_pos, tolerance=0.05),

        SignalTaskComplete(),
    ])
    return root
