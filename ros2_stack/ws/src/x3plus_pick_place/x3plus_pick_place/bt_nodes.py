"""py_trees BehaviorTree leaf nodes for X3Plus pick-and-place.

Adapted from panda_pick_place/bt_nodes.py for the 5-DOF X3Plus arm.
MoveToPose uses position-only goals (no orientation constraint).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import py_trees

if TYPE_CHECKING:
    from x3plus_pick_place.pick_place_node import PickPlaceContext


class MoveToPose(py_trees.behaviour.Behaviour):
    """Plan and execute a position-only Cartesian goal via MoveIt2."""

    def __init__(self, name: str, target_position: list[float]):
        super().__init__(name)
        self._target_position = target_position
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
            self.logger.info(f"Planning to position {self._target_position}")
            success = self.ctx.move_to_pose(self._target_position)
            if success:
                self.logger.info(f"Motion completed for {self.name}")
                return py_trees.common.Status.SUCCESS
            else:
                self.logger.error(f"Planning failed for {self.name}")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class MoveToJoints(py_trees.behaviour.Behaviour):
    """Move to a named joint configuration."""

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
    """Set the grip_joint position (0 = open, -1.54 = closed)."""

    def __init__(self, name: str, position: float):
        super().__init__(name)
        self._position = position

    def setup(self, **kwargs):
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

    def update(self) -> py_trees.common.Status:
        self.logger.info(f"Gripper -> {self._position}")
        self.ctx.set_gripper(self._position)
        return py_trees.common.Status.SUCCESS


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
        client = self.attach_blackboard_client()
        client.register_key(key="/ctx", access=py_trees.common.Access.READ)
        self.ctx: PickPlaceContext = client.get("/ctx")

    def update(self) -> py_trees.common.Status:
        if self.ctx.cube_pose is not None:
            self.logger.info(f"Cube pose received: {self.ctx.cube_pose[:3]}")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class SignalTaskComplete(py_trees.behaviour.Behaviour):
    """Publish task_complete=True to signal the simulator."""

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
    grasp_offset_z: float = 0.02,
    gripper_open: float = 0.0,
    gripper_closed: float = -1.54,
    settle_time: float = 0.5,
    lift_height: float = 0.06,
) -> py_trees.behaviour.Behaviour:
    """Build the pick-and-place BehaviorTree for the X3Plus arm.

    Waypoints are computed relative to the cube/place positions.
    All goals are position-only (orientation is left free for the 5-DOF arm).
    """
    pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_offset_z + lift_height]
    grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_offset_z]
    lift = [cube_pos[0], cube_pos[1], cube_pos[2] + grasp_offset_z + lift_height]
    pre_place = [place_pos[0], place_pos[1], place_pos[2] + grasp_offset_z + lift_height]
    place = [place_pos[0], place_pos[1], place_pos[2] + grasp_offset_z]

    ready_joints = [0.0, -0.5, 0.5, -0.5, 0.0]

    root = py_trees.composites.Sequence("PickAndPlace", memory=True)
    root.add_children([
        WaitForCubePose("WaitForCubePose"),
        SetGripper("OpenGripperInit", gripper_open),
        WaitForSettle("SettleInit", settle_time),
        MoveToJoints("MoveToReady", ready_joints),
        WaitForSettle("SettleReady", settle_time),
        MoveToPose("MoveToPreGrasp", pre_grasp),
        WaitForSettle("SettlePreGrasp", 0.3),
        MoveToPose("MoveToGrasp", grasp),
        WaitForSettle("SettleGrasp", 0.3),
        SetGripper("CloseGripper", gripper_closed),
        WaitForSettle("SettleGripperClose", settle_time * 2),
        MoveToPose("LiftCube", lift),
        WaitForSettle("SettleLift", 0.3),
        MoveToPose("MoveToPrePlace", pre_place),
        WaitForSettle("SettlePrePlace", 0.3),
        MoveToPose("MoveToPlace", place),
        WaitForSettle("SettlePlace", 0.3),
        SetGripper("OpenGripperRelease", gripper_open),
        WaitForSettle("SettleRelease", settle_time),
        MoveToPose("Retreat", pre_place),
        WaitForSettle("SettleRetreat", 0.3),
        MoveToJoints("ReturnToReady", ready_joints),
        WaitForSettle("SettleFinal", settle_time),
        SignalTaskComplete(),
    ])
    return root
