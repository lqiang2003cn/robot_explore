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
