"""Analytical position-only IK solver for the Yahboom X3Plus 5-DOF arm.

Kinematic chain (base_link → arm_link5), from x3plus.urdf:

  Joint 1 – base yaw    – axis (0,0,-1) – origin xyz(0.09825, 0, 0.102)
  Joint 2 – shoulder    – axis (0,0,-1) – origin xyz(0, 0, 0.0405) rpy(-π/2,0,0)
  Joint 3 – elbow       – axis (0,0,-1) – origin xyz(0, -0.0829, 0)
  Joint 4 – wrist pitch – axis (0,0,-1) – origin xyz(0, -0.0829, 0)
  Joint 5 – wrist roll  – axis (0,0,+1) – origin xyz(-0.00215, -0.17455, 0) rpy(π/2,0,0)

Joint 5 (roll) does not affect the origin of arm_link5, so position-only
IK requires solving only joints 1–4.  Joint 5 is a free parameter.

FK derivation (position of arm_link5 origin in base_link frame):

  Let α = q2+q3+q4  (total pitch from vertical).

  S = L1·sin(q2) + L2·sin(q2+q3) + D5·cos(α) + L3·sin(α)
  Z = L1·cos(q2) + L2·cos(q2+q3) + L3·cos(α) − D5·sin(α)

  px = BASE_X − S·cos(q1)
  py = S·sin(q1)
  pz = SHOULDER_Z + Z

IK strategy:
  1. Solve q1 from the target XY position (base rotation).
  2. Sweep the free parameter α (total pitch) over a dense grid.
  3. For each α, solve the resulting 2R planar subproblem for (q2, q3)
     in closed form, then recover q4 = α − q2 − q3.
  4. Verify FK, enforce joint limits, pick solution closest to current joints.
"""

from __future__ import annotations

import math

import numpy as np

# ── URDF constants (metres) ──────────────────────────────────────────────────
BASE_X = 0.09825
BASE_Z = 0.102
DZ_SHOULDER = 0.0405
SHOULDER_Z = BASE_Z + DZ_SHOULDER  # 0.1425

L1 = 0.0829    # upper arm   (link2 → link3)
L2 = 0.0829    # forearm     (link3 → link4)
L3 = 0.17455   # wrist link  (link4 → link5, dominant Y-component)
D5 = 0.00215   # wrist link  (link4 → link5, small X-component)

# ── Joint limits (rad) from URDF <limit> tags ────────────────────────────────
J_LO = np.array([-1.5708, -1.5708, -1.5708, -1.5708, -1.5708])
J_HI = np.array([1.5708, 1.5708, 1.5708, 1.5708, 3.14159])
_MARGIN = 0.005

_L12_SQ = L1 * L1 + L2 * L2
_2L1L2 = 2.0 * L1 * L2
_PI = math.pi
_2PI = 2.0 * _PI
ORTH_WORKSPACE_ALPHA = _PI


def forward_kinematics(q: np.ndarray | list[float]) -> np.ndarray:
    """Return arm_link5 origin ``[x, y, z]`` in the ``base_link`` frame."""
    q1, q2, q3, q4 = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    c1, s1 = math.cos(q1), math.sin(q1)
    q23 = q2 + q3
    q234 = q23 + q4
    s2, c2 = math.sin(q2), math.cos(q2)
    s23, c23 = math.sin(q23), math.cos(q23)
    s234, c234 = math.sin(q234), math.cos(q234)

    S = L1 * s2 + L2 * s23 + D5 * c234 + L3 * s234
    Z = L1 * c2 + L2 * c23 + L3 * c234 - D5 * s234

    return np.array([BASE_X - S * c1, S * s1, SHOULDER_Z + Z])


def _wrap(a: float) -> float:
    """Wrap angle to (−π, π]."""
    return (a + _PI) % _2PI - _PI


def solve_ik(
    target: np.ndarray | list[float],
    q5: float = 0.0,
    current: np.ndarray | list[float] | None = None,
    pos_tol: float = 0.003,
) -> list[float] | None:
    """Solve position-only IK for the X3Plus 5-DOF arm.

    Parameters
    ----------
    target  : desired ``[x, y, z]`` of arm_link5 in ``base_link`` frame.
    q5      : desired wrist-roll angle (free DOF, default 0).
    current : current joint positions ``[q1..q5]`` – used to pick the
              closest valid solution for smooth motion.
    pos_tol : maximum acceptable FK position error (metres).

    Returns
    -------
    ``[q1, q2, q3, q4, q5]`` or ``None`` if the target is unreachable.
    """
    px = float(target[0])
    py = float(target[1])
    pz = float(target[2])
    dx = px - BASE_X
    dy = py
    r = math.sqrt(dx * dx + dy * dy)
    h = pz - SHOULDER_Z

    # ── Step 1: base-rotation candidates ─────────────────────────────────
    #
    # From FK:  dx = −S·cos(q1),  dy = S·sin(q1)
    #
    #   "forward" reach (S < 0): q1 = atan2(−dy, dx),  S = −r
    #   "backward" reach (S > 0): q1 = atan2(dy, −dx),  S = +r
    q1_S: list[tuple[float, float]] = []
    if r < 1e-6:
        q1_def = float(current[0]) if current is not None else 0.0
        q1_S.append((q1_def, 0.0))
    else:
        for q1c, Sval in [
            (math.atan2(-dy, dx), -r),
            (math.atan2(dy, -dx), r),
        ]:
            if J_LO[0] + _MARGIN <= q1c <= J_HI[0] - _MARGIN:
                q1_S.append((q1c, Sval))
    if not q1_S:
        return None

    best: list[float] | None = None
    best_cost = float("inf")

    # ── Step 2: sweep total-pitch α = q2+q3+q4 ──────────────────────────
    n_alpha = 600
    alphas = np.linspace(-3.0, 3.0, n_alpha)

    for q1, S in q1_S:
        for alpha in alphas:
            sa = math.sin(alpha)
            ca = math.cos(alpha)

            # Subtract the last-link contribution to get the 2R target.
            Rv = S - D5 * ca - L3 * sa
            Hv = h - L3 * ca + D5 * sa

            # ── Step 3: solve 2R sub-problem ─────────────────────────────
            D_sq = Rv * Rv + Hv * Hv
            cq3 = (D_sq - _L12_SQ) / _2L1L2
            if cq3 < -1.0 or cq3 > 1.0:
                continue
            sq3_abs = math.sqrt(max(0.0, 1.0 - cq3 * cq3))

            for sign3 in (1.0, -1.0):
                sq3 = sign3 * sq3_abs
                q3 = math.atan2(sq3, cq3)
                if q3 < J_LO[2] + _MARGIN or q3 > J_HI[2] - _MARGIN:
                    continue

                A = L1 + L2 * cq3
                B = L2 * sq3
                q2 = _wrap(math.atan2(Rv, Hv) - math.atan2(B, A))
                if q2 < J_LO[1] + _MARGIN or q2 > J_HI[1] - _MARGIN:
                    continue

                q4 = _wrap(alpha - q2 - q3)
                if q4 < J_LO[3] + _MARGIN or q4 > J_HI[3] - _MARGIN:
                    continue

                # ── Step 4: FK verification ──────────────────────────────
                sol = [q1, q2, q3, q4, q5]
                fk = forward_kinematics(sol)
                err = math.sqrt(
                    (fk[0] - px) ** 2 + (fk[1] - py) ** 2 + (fk[2] - pz) ** 2
                )
                if err > pos_tol:
                    continue

                # Prefer the solution closest to the current configuration.
                if current is not None:
                    cost = sum(
                        (sol[i] - float(current[i])) ** 2 for i in range(4)
                    )
                else:
                    cost = sum(sol[i] ** 2 for i in range(4))

                if cost < best_cost:
                    best_cost = cost
                    best = list(sol)

    return best


# ── Decomposed motion helpers ────────────────────────────────────────────────

def compute_base_yaw(target_xy: list[float] | np.ndarray) -> float | None:
    """Compute ``arm_joint1`` angle to face a target XY in the base_link frame.

    Uses the "forward reach" branch (S < 0) which keeps the arm extending
    away from the base toward the target.  Returns ``None`` if out of range.
    """
    dx = float(target_xy[0]) - BASE_X
    dy = float(target_xy[1])
    r = math.sqrt(dx * dx + dy * dy)
    if r < 1e-6:
        return 0.0
    q1 = math.atan2(-dy, dx)
    if J_LO[0] + _MARGIN <= q1 <= J_HI[0] - _MARGIN:
        return q1
    q1_back = math.atan2(dy, -dx)
    if J_LO[0] + _MARGIN <= q1_back <= J_HI[0] - _MARGIN:
        return q1_back
    return None


def cartesian_to_sagittal(
    target_xyz: list[float] | np.ndarray,
    q1: float,
) -> tuple[float, float]:
    """Convert a 3D target (base_link frame) to sagittal-plane coordinates.

    Returns ``(S, Z)`` where S is the signed horizontal reach in the arm's
    sagittal plane and Z is the height above the shoulder.
    """
    px, py, pz = float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])
    dx = px - BASE_X
    dy = py
    c1, s1 = math.cos(q1), math.sin(q1)
    S = -dx * c1 + dy * s1
    Z = pz - SHOULDER_Z
    return S, Z


_MAX_JOINT_DELTA = 2.0


def solve_planar_ik(
    S_target: float,
    Z_target: float,
    alpha: float | None = None,
    current_q234: list[float] | None = None,
) -> list[float] | None:
    """Solve for joints 2, 3, 4 in the sagittal plane.

    If ``alpha`` is given, enforces q2+q3+q4 = alpha (fixes the wrist pitch
    direction).  Otherwise sweeps alpha to find the best reachable solution.

    Returns ``[q2, q3, q4]`` or ``None``.
    """
    if alpha is not None:
        alphas = np.array([alpha])
    else:
        base = np.linspace(-3.5, 4.5, 600)
        if current_q234 is not None:
            cur_alpha = float(sum(current_q234))
            extras = []
            for shift in (0.0, -_2PI, _2PI):
                center = cur_alpha + shift
                if -5.0 <= center <= 6.0:
                    extras.append(np.linspace(center - 0.5, center + 0.5, 100))
            alphas = np.concatenate([base] + extras) if extras else base
        else:
            alphas = base

    best: list[float] | None = None
    best_cost = float("inf")
    fallback: list[float] | None = None
    fallback_cost = float("inf")

    for a in alphas:
        sa = math.sin(a)
        ca = math.cos(a)
        Rv = S_target - D5 * ca - L3 * sa
        Hv = Z_target - L3 * ca + D5 * sa

        D_sq = Rv * Rv + Hv * Hv
        cq3 = (D_sq - _L12_SQ) / _2L1L2
        if cq3 < -1.0 or cq3 > 1.0:
            continue
        sq3_abs = math.sqrt(max(0.0, 1.0 - cq3 * cq3))

        for sign3 in (1.0, -1.0):
            sq3 = sign3 * sq3_abs
            q3 = math.atan2(sq3, cq3)
            if q3 < J_LO[2] + _MARGIN or q3 > J_HI[2] - _MARGIN:
                continue

            A = L1 + L2 * cq3
            B = L2 * sq3
            q2 = _wrap(math.atan2(Rv, Hv) - math.atan2(B, A))
            if q2 < J_LO[1] + _MARGIN or q2 > J_HI[1] - _MARGIN:
                continue

            q4 = _wrap(a - q2 - q3)
            if q4 < J_LO[3] + _MARGIN or q4 > J_HI[3] - _MARGIN:
                continue

            S_check = L1 * math.sin(q2) + L2 * math.sin(q2 + q3) + D5 * ca + L3 * sa
            Z_check = L1 * math.cos(q2) + L2 * math.cos(q2 + q3) + L3 * ca - D5 * sa
            err = math.sqrt((S_check - S_target) ** 2 + (Z_check - Z_target) ** 2)
            if err > 0.005:
                continue

            if current_q234 is not None:
                cost = sum(
                    (v - float(current_q234[i])) ** 2
                    for i, v in enumerate([q2, q3, q4])
                )
                max_delta = max(
                    abs(q2 - float(current_q234[0])),
                    abs(q3 - float(current_q234[1])),
                    abs(q4 - float(current_q234[2])),
                )
                if max_delta > _MAX_JOINT_DELTA:
                    if cost < fallback_cost:
                        fallback_cost = cost
                        fallback = [q2, q3, q4]
                    continue
            else:
                cost = q2 ** 2 + q3 ** 2 + q4 ** 2

            if cost < best_cost:
                best_cost = cost
                best = [q2, q3, q4]

    return best if best is not None else fallback


def solve_orthogonal_planar_ik(
    S_target: float,
    Z_target: float,
    current_q234: list[float] | None = None,
) -> list[float] | None:
    """Solve planar IK while keeping the gripper orthogonal to the floor."""
    return solve_planar_ik(
        S_target,
        Z_target,
        alpha=ORTH_WORKSPACE_ALPHA,
        current_q234=current_q234,
    )


def solve_orthogonal_ik(
    target: np.ndarray | list[float],
    q5: float = 0.0,
    current: np.ndarray | list[float] | None = None,
    pos_tol: float = 0.003,
) -> list[float] | None:
    """Solve IK inside ``orth_workspace`` (fixed downward gripper pitch)."""
    q1 = compute_base_yaw(target[:2])
    if q1 is None:
        return None

    current_q234 = None if current is None else list(current[1:4])
    S_target, Z_target = cartesian_to_sagittal(target, q1)
    q234 = solve_orthogonal_planar_ik(
        S_target,
        Z_target,
        current_q234=current_q234,
    )
    if q234 is None:
        return None

    sol = [q1, *q234, q5]
    fk = forward_kinematics(sol)
    err = math.sqrt(sum((float(fk[i]) - float(target[i])) ** 2 for i in range(3)))
    if err > pos_tol:
        return None
    return sol


def is_in_orth_workspace(
    target: np.ndarray | list[float],
    current_q234: list[float] | None = None,
) -> bool:
    """Return ``True`` if a Cartesian target is reachable in ``orth_workspace``."""
    return solve_orthogonal_ik(
        target,
        current=None if current_q234 is None else [0.0, *current_q234, 0.0],
    ) is not None


def _best_90deg_candidate(
    base_angle: float,
    candidates: list[float],
) -> float:
    """Return the candidate closest to *base_angle* (angular distance)."""
    best = candidates[0]
    best_d = abs(_wrap(candidates[0] - base_angle))
    for c in candidates[1:]:
        d = abs(_wrap(c - base_angle))
        if d < best_d:
            best_d = d
            best = c
    return best


def compute_wrist_roll(block_yaw: float, q1: float) -> float:
    """Compute ``arm_joint5`` so the gripper's inner faces are parallel to the
    block's vertical faces (for grasping).

    The gripper's pinch axis (line connecting the two finger pads) lies along
    arm_link5's local Y.  When the arm points straight down the world-frame
    yaw of the pinch axis is ``-q1 + q5`` (joint1 axis is ``(0,0,-1)``,
    joint5 axis is ``(0,0,+1)``).

    For the gripper's inner faces to be parallel to the block's vertical
    faces, the pinch axis must be **aligned** with one of the block's
    horizontal axes.  Because the block has a square cross-section (12 × 12
    mm) there are four equivalent orientations separated by 90°.  We pick the
    one that requires the least joint5 motion (closest to 0).

    The result is clamped to ``arm_joint5`` limits.
    """
    candidates = [_wrap(block_yaw + q1 + k * _PI / 2) for k in range(4)]
    valid = [c for c in candidates
             if J_LO[4] + _MARGIN <= c <= J_HI[4] - _MARGIN]
    if not valid:
        valid = candidates
    q5 = _best_90deg_candidate(0.0, valid)
    q5 = max(J_LO[4] + _MARGIN, min(J_HI[4] - _MARGIN, q5))
    return q5


def compute_place_wrist_roll(
    yellow_yaw: float,
    red_yaw: float,
    q1: float,
    current_q5: float,
) -> float:
    """Compute ``arm_joint5`` for placing so the yellow block's local frame
    aligns with the red block's local frame.

    While the yellow block is gripped, its world-frame yaw is locked to the
    gripper.  The gripper's world yaw is ``-q1 + q5``.  When the block was
    picked, its yaw was ``yellow_yaw``, so the relative offset between the
    gripper's world yaw and the block's yaw is fixed:

        block_world_yaw = gripper_world_yaw = -q1_pick + q5_pick

    At place time (possibly different q1), we want the block's world yaw to
    equal the red block's yaw modulo 90° (square symmetry).  The gripper
    world yaw at place time is ``-q1 + q5``, and the block yaw equals
    ``-q1 + q5 + (yellow_yaw - (-q1_pick + q5_pick))``.

    Since friction locks the block rigidly to the gripper fingers at grasp
    time, the block's world yaw simply equals ``-q1 + q5 + offset`` where
    ``offset = yellow_yaw - (-q1_pick + q5_pick)``.  But we don't track
    q1_pick/q5_pick here.  Instead, we note that after grasping, the block
    yaw in the world tracks the gripper yaw.  The simplest correct approach:
    we want the *gripper* world yaw at place time to match the red block's
    yaw modulo 90°, adjusted by the same offset that was used during pick.

    To keep the interface simple, we compute q5 so that the gripper world
    yaw ``(-q1 + q5)`` equals ``red_yaw + k*pi/2`` for the k that is closest
    to ``current_q5``, which preserves the relative block-gripper offset from
    the pick phase (since the pick alignment already matched the yellow
    block's face axis).

    Returns the clamped q5 value.
    """
    candidates = [_wrap(red_yaw + q1 + k * _PI / 2) for k in range(4)]
    valid = [c for c in candidates
             if J_LO[4] + _MARGIN <= c <= J_HI[4] - _MARGIN]
    if not valid:
        valid = candidates
    q5 = _best_90deg_candidate(current_q5, valid)
    q5 = max(J_LO[4] + _MARGIN, min(J_HI[4] - _MARGIN, q5))
    return q5
