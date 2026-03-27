"""py_trees BehaviorTree leaf nodes for pick-and-place orchestration.

Each node wraps a single phase of the task (move arm, actuate gripper, wait)
and communicates with the ROS2 infrastructure via a shared blackboard.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import py_trees

if TYPE_CHECKING:
    from panda_pick_place.pick_place_node import PickPlaceContext


class MoveToPose(py_trees.behaviour.Behaviour):
    """Plan and execute a Cartesian goal for the panda_arm via MoveIt2."""

    def __init__(self, name: str, target_pose: list[float], orientation: list[float] | None = None):
        super().__init__(name)
        self._target_pose = target_pose
        self._orientation = orientation or [1.0, 0.0, 0.0, 0.0]
        self._started = False

    def setup(self, **kwargs):
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            self.logger.info(f"Planning to {self._target_pose}")
            success = self.ctx.move_to_pose(self._target_pose, self._orientation)
            if success:
                self.logger.info(f"Trajectory execution started for {self.name}")
                return py_trees.common.Status.SUCCESS
            else:
                self.logger.error(f"Planning failed for {self.name}")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class MoveToJoints(py_trees.behaviour.Behaviour):
    """Move to a named joint configuration (e.g. 'ready')."""

    def __init__(self, name: str, joint_positions: list[float]):
        super().__init__(name)
        self._joint_positions = joint_positions
        self._started = False

    def setup(self, **kwargs):
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

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
                self.logger.error(f"Joint motion failed for {self.name}")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class SetGripper(py_trees.behaviour.Behaviour):
    """Open or close the gripper by publishing finger joint positions."""

    def __init__(self, name: str, width: float):
        super().__init__(name)
        self._width = width

    def setup(self, **kwargs):
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

    def update(self) -> py_trees.common.Status:
        self.logger.info(f"Gripper -> {self._width}")
        self.ctx.set_gripper(self._width)
        return py_trees.common.Status.SUCCESS


class WaitForSettle(py_trees.behaviour.Behaviour):
    """Wait a fixed duration to let physics/gripper settle."""

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
    """Wait until we have received at least one cube pose from Isaac Sim."""

    def __init__(self, name: str = "WaitForCubePose"):
        super().__init__(name)

    def setup(self, **kwargs):
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

    def update(self) -> py_trees.common.Status:
        if self.ctx.cube_pose is not None:
            self.logger.info(f"Cube pose received: {self.ctx.cube_pose[:3]}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class SignalTaskComplete(py_trees.behaviour.Behaviour):
    """Publish task_complete=True to signal Isaac Sim to stop."""

    def __init__(self, name: str = "SignalTaskComplete"):
        super().__init__(name)

    def setup(self, **kwargs):
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

    def update(self) -> py_trees.common.Status:
        self.ctx.signal_done()
        self.logger.info("Task complete signal sent")
        return py_trees.common.Status.SUCCESS


def build_pick_place_tree(
    cube_pos: list[float],
    place_pos: list[float],
    grasp_offset_z: float = 0.04,
    gripper_open: float = 0.04,
    gripper_closed: float = 0.001,
    settle_time: float = 0.5,
    lift_height: float = 0.15,
) -> py_trees.behaviour.Behaviour:
    """Build the full pick-and-place BehaviorTree.

    The orientation keeps the gripper pointing straight down (top-grasp).
    """
    # Gripper-down orientation (w, x, y, z) — 180-deg rotation about X
    grasp_orient = [0.0, 1.0, 0.0, 0.0]

    pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_offset_z + lift_height]
    grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_offset_z]
    lift = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_offset_z + lift_height]
    pre_place = [place_pos[0], place_pos[1], place_pos[2] + grasp_offset_z + lift_height]
    place = [place_pos[0], place_pos[1], place_pos[2] + grasp_offset_z]

    # Panda "ready" joint config (arm up, clear of table)
    ready_joints = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    root = py_trees.composites.Sequence("PickAndPlace", memory=True)
    root.add_children([
        WaitForCubePose("WaitForCubePose"),
        SetGripper("OpenGripperInit", gripper_open),
        WaitForSettle("SettleInit", settle_time),
        MoveToJoints("MoveToReady", ready_joints),
        WaitForSettle("SettleReady", settle_time),
        MoveToPose("MoveToPreGrasp", pre_grasp, grasp_orient),
        WaitForSettle("SettlePreGrasp", 0.3),
        MoveToPose("MoveToGrasp", grasp, grasp_orient),
        WaitForSettle("SettleGrasp", 0.3),
        SetGripper("CloseGripper", gripper_closed),
        WaitForSettle("SettleGripperClose", settle_time * 2),
        MoveToPose("LiftCube", lift, grasp_orient),
        WaitForSettle("SettleLift", 0.3),
        MoveToPose("MoveToPrePlace", pre_place, grasp_orient),
        WaitForSettle("SettlePrePlace", 0.3),
        MoveToPose("MoveToPlace", place, grasp_orient),
        WaitForSettle("SettlePlace", 0.3),
        SetGripper("OpenGripperRelease", gripper_open),
        WaitForSettle("SettleRelease", settle_time),
        MoveToPose("Retreat", pre_place, grasp_orient),
        WaitForSettle("SettleRetreat", 0.3),
        MoveToJoints("ReturnToReady", ready_joints),
        WaitForSettle("SettleFinal", settle_time),
        SignalTaskComplete(),
    ])
    return root
