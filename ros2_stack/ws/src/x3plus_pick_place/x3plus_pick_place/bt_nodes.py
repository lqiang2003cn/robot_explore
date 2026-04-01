"""py_trees BehaviorTree leaf nodes for X3Plus two-block pick-and-place.

Motion is decomposed into three kinematic groups that match the YAML-driven
tree configuration:

  rotate_base_to_block   → arm_joint1 only
  move_above_block       → arm_joint2, 3, 4 (sagittal plane IK)
  align_wrist_to_block   → arm_joint5 only
  open/close_gripper     → grip_joint

The tree sequence is loaded from config/pick_place_tree.yaml at startup by
``build_tree_from_config``.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import py_trees

from x3plus_pick_place.ik_solver import (
    compute_base_yaw,
    cartesian_to_sagittal,
    solve_orthogonal_planar_ik,
    solve_planar_ik,
    compute_wrist_roll,
    compute_place_wrist_roll,
)

if TYPE_CHECKING:
    from x3plus_pick_place.pick_place_node import PickPlaceContext


# ── Helpers ──────────────────────────────────────────────────────────────────

def _bb_ctx(node: py_trees.behaviour.Behaviour) -> "PickPlaceContext":
    client = node.attach_blackboard_client()
    client.register_key(key="/ctx", access=py_trees.common.Access.READ)
    return client.get("/ctx")


def _resolve_height(step: dict, config: dict) -> float:
    """Resolve a height_offset value that may be a string key or a float."""
    raw = step.get("height_offset", 0)
    if isinstance(raw, str):
        return float(config["movement"][raw])
    return float(raw)


def _compute_ee_z(
    block_z: float,
    height_offset: float,
    config: dict,
    stack_on_top: bool = False,
    pick_block_half_h: float = 0.0,
    place_block_half_h: float = 0.0,
) -> float:
    """Compute the EE (arm_link5) target Z for a block approach.

    When the arm points down the fingertips sit *above* arm_link5, so
    arm_link5 must go *below* the block centre by ``finger_offset``.
    """
    finger_offset = config["robot"]["finger_ee_offset_z"]
    if stack_on_top:
        target_surface_z = block_z + place_block_half_h + pick_block_half_h
        return target_surface_z - finger_offset + height_offset
    return block_z - finger_offset + height_offset


# ── Wait / sensor nodes ─────────────────────────────────────────────────────

class WaitForBlockPoses(py_trees.behaviour.Behaviour):
    """Wait until both yellow and red block poses are received."""

    def __init__(self, name: str = "WaitForBlockPoses"):
        super().__init__(name)

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def update(self) -> py_trees.common.Status:
        if self.ctx.yellow.received and self.ctx.red.received:
            if self.ctx.yellow_rest_pos is None:
                self.ctx.yellow_rest_pos = self.ctx.yellow.position.copy()
            self.logger.info("Both block poses received")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# ── Gripper nodes ────────────────────────────────────────────────────────────

class OpenGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "OpenGripper"):
        super().__init__(name)
        self._command_sent = False
        self._start_time: float | None = None

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._command_sent = False
        self._start_time = None

    def update(self) -> py_trees.common.Status:
        pos = self.ctx.config["robot"]["gripper_open"]
        if not self._command_sent:
            self.logger.info(f"Gripper OPEN → {pos:.2f}")
            self.ctx.set_gripper(pos)
            self._command_sent = True
            self._start_time = time.time()
            return py_trees.common.Status.RUNNING

        self.ctx.set_gripper(pos)
        actual = self.ctx.get_gripper_position()
        if abs(actual - pos) < 0.6:
            self.logger.info(f"Gripper OPEN OK: {actual:.3f}")
            return py_trees.common.Status.SUCCESS
        if time.time() - self._start_time > 6.0:
            self.logger.error(f"Gripper OPEN TIMEOUT: {actual:.3f}")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class CloseGripper(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "CloseGripper"):
        super().__init__(name)
        self._command_sent = False
        self._start_time: float | None = None

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._command_sent = False
        self._start_time = None

    def update(self) -> py_trees.common.Status:
        pos = self.ctx.config["robot"]["gripper_closed"]
        threshold = self.ctx.config["robot"]["gripper_close_threshold"]
        if not self._command_sent:
            self.logger.info(f"Gripper CLOSE → {pos:.2f}")
            self.ctx.set_gripper(pos)
            self._command_sent = True
            self._start_time = time.time()
            return py_trees.common.Status.RUNNING

        self.ctx.set_gripper(pos)
        actual = self.ctx.get_gripper_position()
        if actual >= threshold:
            self.logger.info(
                f"Gripper CLOSED OK: {actual:.3f} >= {threshold:.3f}"
            )
            return py_trees.common.Status.SUCCESS
        if time.time() - self._start_time > 15.0:
            self.logger.error(f"Gripper CLOSE TIMEOUT: {actual:.3f}")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


# ── Settle ───────────────────────────────────────────────────────────────────

class Settle(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "Settle", duration: float = 0.5):
        super().__init__(name)
        self._duration = duration
        self._start_time: float | None = None

    def initialise(self):
        self._start_time = time.time()

    def update(self) -> py_trees.common.Status:
        if time.time() - self._start_time >= self._duration:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


# ── Init / return ────────────────────────────────────────────────────────────

class MoveToInit(py_trees.behaviour.Behaviour):
    """Move all arm joints to the configured init pose."""

    def __init__(self, name: str = "MoveToInit"):
        super().__init__(name)
        self._started = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            joints = self.ctx.config["robot"]["init_joints"]
            self.logger.info(f"Moving to init pose {joints}")
            if self.ctx.move_to_joints(joints):
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


# ── Decomposed motion nodes ─────────────────────────────────────────────────

class RotateBaseToBlock(py_trees.behaviour.Behaviour):
    """Rotate arm_joint1 to face a named block."""

    def __init__(self, name: str, block_name: str):
        super().__init__(name)
        self._block_name = block_name
        self._started = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            block = self.ctx.get_block(self._block_name)
            q1 = compute_base_yaw(block.position[:2])
            if q1 is None:
                self.logger.error(
                    f"Cannot compute base yaw for {self._block_name}"
                )
                return py_trees.common.Status.FAILURE
            self.logger.info(
                f"RotateBase to {self._block_name}: q1={q1:.3f}"
            )
            if self.ctx.rotate_base(q1):
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class MoveAboveBlock(py_trees.behaviour.Behaviour):
    """Move joints 2, 3, 4 to position the EE above a named block."""

    def __init__(
        self,
        name: str,
        block_name: str,
        height_offset: float,
        config: dict,
        stack_on_top: bool = False,
        use_rest_position: bool = False,
    ):
        super().__init__(name)
        self._block_name = block_name
        self._height_offset = height_offset
        self._config = config
        self._stack_on_top = stack_on_top
        self._use_rest_position = use_rest_position
        self._started = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            if self._use_rest_position and self._block_name == "yellow":
                block_pos = self.ctx.yellow_rest_pos
            else:
                block_pos = self.ctx.get_block(self._block_name).position
            q1 = self.ctx.get_arm_joints()[0]

            yellow_hh = self._config["blocks"]["yellow"]["half_height"]
            red_hh = self._config["blocks"]["red"]["half_height"]
            if self._block_name == "red":
                pick_hh, place_hh = yellow_hh, red_hh
            else:
                pick_hh, place_hh = yellow_hh, yellow_hh

            ee_z = _compute_ee_z(
                block_pos[2],
                self._height_offset,
                self._config,
                stack_on_top=self._stack_on_top,
                pick_block_half_h=pick_hh,
                place_block_half_h=place_hh,
            )
            target_xyz = [block_pos[0], block_pos[1], ee_z]
            S, Z = cartesian_to_sagittal(target_xyz, q1)

            current_q234 = self.ctx.get_arm_joints()[1:4]
            sol = solve_orthogonal_planar_ik(S, Z, current_q234=current_q234)
            if sol is None:
                sol = solve_planar_ik(S, Z, current_q234=current_q234)
                if sol is not None:
                    self.logger.warning(
                        f"Orth IK failed for {self._block_name}; "
                        f"using general planar IK"
                    )
            if sol is None:
                self.logger.error(
                    f"Planar IK failed for {self._block_name} "
                    f"S={S:.4f} Z={Z:.4f}"
                )
                return py_trees.common.Status.FAILURE

            self.logger.info(
                f"MoveAbove {self._block_name}: q234={[f'{q:.3f}' for q in sol]} "
                f"ee_z={ee_z:.3f}"
            )
            if self.ctx.move_in_plane(sol):
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class AlignWristToBlock(py_trees.behaviour.Behaviour):
    """Rotate arm_joint5 to align the gripper with a block's orientation.

    *align_to*: when set to another block name, computes q5 so that the
    currently-held block (picked with its own yaw) will land with its local
    frame aligned to *align_to*'s local frame (modulo 90° square symmetry).
    When ``None`` (default / pick phase), q5 is chosen so the gripper's inner
    faces are parallel to the target block's vertical faces.
    """

    def __init__(self, name: str, block_name: str, align_to: str | None = None):
        super().__init__(name)
        self._block_name = block_name
        self._align_to = align_to
        self._started = False

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._started = False

    def update(self) -> py_trees.common.Status:
        if not self._started:
            self._started = True
            block = self.ctx.get_block(self._block_name)
            q1 = self.ctx.get_arm_joints()[0]
            current_q5 = self.ctx.get_arm_joints()[4]

            if self._align_to is not None:
                target_block = self.ctx.get_block(self._align_to)
                q5 = compute_place_wrist_roll(
                    block.yaw, target_block.yaw, q1, current_q5,
                )
                self.logger.info(
                    f"AlignWrist(place) {self._block_name}→{self._align_to}: "
                    f"yellow_yaw={block.yaw:.3f} red_yaw={target_block.yaw:.3f} "
                    f"q1={q1:.3f} → q5={q5:.3f}"
                )
            else:
                q5 = compute_wrist_roll(block.yaw, q1)
                self.logger.info(
                    f"AlignWrist(grasp) to {self._block_name}: "
                    f"block_yaw={block.yaw:.3f} q1={q1:.3f} → q5={q5:.3f}"
                )

            if self.ctx.align_wrist(q5):
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


# ── Verification nodes ───────────────────────────────────────────────────────

class VerifyGrasp(py_trees.behaviour.Behaviour):
    """Verify the yellow block Z is elevated above its resting height."""

    def __init__(self, name: str = "VerifyGrasp"):
        super().__init__(name)
        self._start_time: float | None = None

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._start_time = time.time()

    def update(self) -> py_trees.common.Status:
        block = self.ctx.yellow
        rest_pos = self.ctx.yellow_rest_pos
        lift_h = self.ctx.config["movement"]["lift_height"]
        if block.position is not None and rest_pos is not None:
            lifted = block.position[2] - rest_pos[2]
            min_lift = lift_h * 0.3
            if lifted >= min_lift:
                self.logger.info(
                    f"Grasp verified: lifted {lifted*1000:.1f}mm >= {min_lift*1000:.1f}mm"
                )
                return py_trees.common.Status.SUCCESS
        if time.time() - self._start_time > 2.0:
            if block.position is not None and rest_pos is not None:
                lifted = block.position[2] - rest_pos[2]
                self.logger.error(
                    f"Grasp FAILED: lifted {lifted*1000:.1f}mm"
                )
            else:
                self.logger.error("Grasp FAILED: no block pose")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class VerifyPlacement(py_trees.behaviour.Behaviour):
    """Verify the yellow block XY is near the red block XY."""

    def __init__(self, name: str = "VerifyPlacement", tolerance: float = 0.05):
        super().__init__(name)
        self._tolerance = tolerance
        self._start_time: float | None = None

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def initialise(self):
        self._start_time = time.time()

    def update(self) -> py_trees.common.Status:
        yellow = self.ctx.yellow
        red = self.ctx.red
        if yellow.position is not None and red.position is not None:
            err_xy = float(np.linalg.norm(
                yellow.position[:2] - red.position[:2]
            ))
            if err_xy < self._tolerance:
                self.logger.info(f"Placement verified: XY err={err_xy*1000:.1f} mm")
                return py_trees.common.Status.SUCCESS
        if time.time() - self._start_time > 3.0:
            if yellow.position is not None and red.position is not None:
                err_xy = float(np.linalg.norm(
                    yellow.position[:2] - red.position[:2]
                ))
                self.logger.error(
                    f"Placement FAILED: XY err={err_xy*1000:.1f} mm "
                    f"> {self._tolerance*1000:.0f} mm"
                )
            else:
                self.logger.error("Block pose unavailable for verification")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.RUNNING


class SignalComplete(py_trees.behaviour.Behaviour):
    """Publish task_complete=True."""

    def __init__(self, name: str = "SignalComplete"):
        super().__init__(name)

    def setup(self, **kwargs):
        self.ctx: PickPlaceContext = _bb_ctx(self)

    def update(self) -> py_trees.common.Status:
        self.ctx.signal_done()
        self.logger.info("Task complete signal sent")
        return py_trees.common.Status.SUCCESS


# ── YAML-driven tree builder ────────────────────────────────────────────────

_ACTION_MAP = {
    "wait_for_block_poses": lambda _s, _c: WaitForBlockPoses(),
    "open_gripper": lambda _s, _c: OpenGripper(),
    "close_gripper": lambda _s, _c: CloseGripper(),
    "settle": lambda s, c: Settle(
        duration=c["movement"].get("settle_time", 0.5),
    ),
    "move_to_init": lambda _s, _c: MoveToInit(),
    "verify_grasp": lambda _s, _c: VerifyGrasp(),
    "verify_placement": lambda _s, _c: VerifyPlacement(),
    "signal_complete": lambda _s, _c: SignalComplete(),
}


def _build_node(step: dict, config: dict, idx: int) -> py_trees.behaviour.Behaviour:
    action = step["action"]
    factory = _ACTION_MAP.get(action)
    if factory is not None:
        node = factory(step, config)
        node.name = f"{action}_{idx}"
        return node

    block_name = step.get("block", "")
    label = f"{action}_{block_name}_{idx}"

    if action == "rotate_base_to_block":
        return RotateBaseToBlock(label, block_name)

    if action in ("move_above_block", "lift_block", "retreat_above_block"):
        height_offset = _resolve_height(step, config)
        stack_on_top = step.get("stack_on_top", False)
        use_rest = (action == "lift_block")
        return MoveAboveBlock(
            label, block_name, height_offset, config,
            stack_on_top=stack_on_top,
            use_rest_position=use_rest,
        )

    if action == "align_wrist_to_block":
        align_to = step.get("align_to", None)
        return AlignWristToBlock(label, block_name, align_to=align_to)

    raise ValueError(f"Unknown BT action: {action}")


def build_tree_from_config(config: dict) -> py_trees.behaviour.Behaviour:
    """Build a py_trees Sequence from a YAML config dict."""
    steps = config["tree"]
    children = [_build_node(step, config, i) for i, step in enumerate(steps)]
    root = py_trees.composites.Sequence("PickAndPlace", memory=True)
    root.add_children(children)
    return root
