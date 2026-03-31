"""Scripted pick-and-place controller for the X3Plus MuJoCo environment.

Uses Jacobian-pseudoinverse IK for Cartesian end-effector control and a
waypoint state machine to sequence the pick-and-place phases.

Standalone usage (saves a demo video):
    source activate_env.sh vla_x3plus
    python -m src.scripted_controller
"""

from __future__ import annotations

import enum
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.env import X3PlusPickCubeEnv  # noqa: E402

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent

ARM_JOINT_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
]


class Phase(enum.IntEnum):
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    TRANSPORT = 4
    PLACE = 5
    RELEASE = 6
    RETREAT = 7
    DONE = 8


class ScriptedPickPlace:
    """Jacobian-based IK controller that executes a pick-and-place sequence."""

    def __init__(
        self,
        env: X3PlusPickCubeEnv,
        gain: float = 2.0,
        pos_threshold: float = 0.012,
        grasp_steps: int = 25,
        release_steps: int = 20,
        approach_height: float = 0.06,
        grasp_height: float = 0.005,
        lift_height: float = 0.10,
    ):
        self.env = env
        self.gain = gain
        self.pos_threshold = pos_threshold
        self.grasp_steps = grasp_steps
        self.release_steps = release_steps
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lift_height = lift_height

        model = env.mj_model
        self._ee_body_id = model.body("arm_link5").id
        self._arm_dof_indices = np.array([
            model.joint(name).dofadr[0] for name in ARM_JOINT_NAMES
        ])

        self._phase = Phase.DONE
        self._phase_counter = 0
        self._cube_pos = np.zeros(3)
        self._place_pos = np.zeros(3)

    def reset(self, cube_pos: np.ndarray, place_pos: np.ndarray) -> None:
        self._cube_pos = cube_pos.copy()
        self._place_pos = place_pos.copy()
        self._phase = Phase.APPROACH
        self._phase_counter = 0

    @property
    def done(self) -> bool:
        return self._phase == Phase.DONE

    def _current_ee_pos(self) -> np.ndarray:
        return self.env.mj_data.body("arm_link5").xpos.copy()

    def _current_cube_pos(self) -> np.ndarray:
        return self.env.mj_data.body("target_cube").xpos.copy()

    def _target_for_phase(self) -> np.ndarray:
        cube = self._current_cube_pos()
        if self._phase == Phase.APPROACH:
            return cube + np.array([0.0, 0.0, self.approach_height])
        elif self._phase == Phase.DESCEND:
            return cube + np.array([0.0, 0.0, self.grasp_height])
        elif self._phase == Phase.LIFT:
            return self._cube_pos + np.array([0.0, 0.0, self.lift_height])
        elif self._phase == Phase.TRANSPORT:
            return self._place_pos + np.array([0.0, 0.0, self.lift_height])
        elif self._phase == Phase.PLACE:
            return self._place_pos + np.array([0.0, 0.0, self.grasp_height])
        elif self._phase == Phase.RETREAT:
            return self._place_pos + np.array([0.0, 0.0, self.lift_height])
        return self._current_ee_pos()

    def _compute_arm_action(self, target: np.ndarray) -> np.ndarray:
        """Jacobian pseudoinverse IK: compute normalised joint deltas toward *target*."""
        model = self.env.mj_model
        data = self.env.mj_data

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, self._ee_body_id)
        J_arm = jacp[:, self._arm_dof_indices]  # (3, 5)

        ee_pos = self._current_ee_pos()
        x_err = target - ee_pos
        x_err *= self.gain

        q_delta, _, _, _ = np.linalg.lstsq(J_arm, x_err, rcond=1e-4)
        action = q_delta / self.env.delta_scale
        max_abs = np.max(np.abs(action))
        if max_abs > 1.0:
            action /= max_abs
        return action

    def __call__(self) -> np.ndarray:
        """Return (6,) action for the current phase."""
        action = np.zeros(6, dtype=np.float32)

        if self._phase == Phase.DONE:
            return action

        if self._phase in (Phase.GRASP, Phase.RELEASE):
            grip_action = -1.0 if self._phase == Phase.GRASP else 1.0
            action[5] = grip_action
            # Keep arm still during grasp/release by targeting current EE position
            target = self._current_ee_pos()
            action[:5] = self._compute_arm_action(target) * 0.1
            self._phase_counter += 1
            limit = self.grasp_steps if self._phase == Phase.GRASP else self.release_steps
            if self._phase_counter >= limit:
                self._advance_phase()
            return action

        target = self._target_for_phase()
        action[:5] = self._compute_arm_action(target)

        # Keep gripper closed during transport / place / retreat
        if self._phase in (Phase.LIFT, Phase.TRANSPORT, Phase.PLACE):
            action[5] = -1.0
        elif self._phase == Phase.RETREAT:
            action[5] = 1.0

        ee_pos = self._current_ee_pos()
        if np.linalg.norm(ee_pos - target) < self.pos_threshold:
            self._advance_phase()

        return action

    def _advance_phase(self) -> None:
        next_val = self._phase.value + 1
        if next_val > Phase.DONE:
            next_val = Phase.DONE
        self._phase = Phase(next_val)
        self._phase_counter = 0


# ── Standalone demo ───────────────────────────────────────────


def _load_config() -> dict:
    with open(COMPONENT_DIR / "config.yml") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = _load_config()
    sim_cfg = config["simulation"]
    ds_cfg = config.get("dataset", {})

    place_position = ds_cfg.get("place_position", [0.15, -0.15, 0.39])
    env = X3PlusPickCubeEnv(
        model_path=sim_cfg["model_path"],
        camera_name=sim_cfg["camera_name"],
        resolution=tuple(sim_cfg["camera_resolution"]),
        timestep=sim_cfg["timestep"],
        control_dt=sim_cfg["control_dt"],
        max_episode_steps=sim_cfg["max_episode_steps"],
        place_position=place_position,
    )

    controller = ScriptedPickPlace(env)

    obs, info = env.reset(seed=42)
    cube_pos = env.mj_data.body("target_cube").xpos.copy()
    controller.reset(cube_pos, env.place_position)

    frames = [obs["observation.images.front_cam"]]
    frame_skip = sim_cfg.get("frame_skip", 2)
    max_steps = sim_cfg["max_episode_steps"]

    print(f"Running scripted pick-and-place demo (max {max_steps} steps) ...")
    print(f"  Cube: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    print(f"  Place: {env.place_position}")

    for step in range(max_steps):
        action = controller()
        obs, reward, terminated, truncated, info = env.step(action)

        if step % frame_skip == 0:
            frames.append(obs["observation.images.front_cam"])

        if step % 50 == 0:
            ee = info["ee_pos"]
            cube = info["cube_pos"]
            print(
                f"  step {step:4d}  phase={Phase(controller._phase).name:<10s}  "
                f"ee=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                f"cube=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]"
            )

        if controller.done or terminated or truncated:
            print(f"  Finished at step {step}  success={info.get('success', False)}")
            break

    env.close()

    output_dir = COMPONENT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "scripted_demo.mp4"

    import imageio

    video_fps = sim_cfg.get("video_fps", 30)
    print(f"Saving {len(frames)} frames -> {video_path} ({video_fps} fps)")
    imageio.mimwrite(
        str(video_path),
        frames,
        fps=video_fps,
        quality=10,
        codec="libx264",
        pixelformat="yuv420p",
        output_params=["-preset", "slow", "-crf", "17"],
    )
    print("Done.")


if __name__ == "__main__":
    main()
