"""Gymnasium environment wrapping the x3plus MuJoCo pick-and-place scene.

Observation space (Dict):
    observation.images.front_cam  – (H, W, 3) uint8 RGB
    observation.state             – (6,) float32 actuated joint positions

Action space:
    Box(-1, 1, shape=(6,)) – scaled joint position deltas
    (5 arm joints + 1 gripper)
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent

ACTUATOR_NAMES = [
    "arm_joint1",
    "arm_joint2",
    "arm_joint3",
    "arm_joint4",
    "arm_joint5",
    "grip_joint",
]

ACTUATOR_RANGES = np.array([
    [-1.5708, 1.5708],   # arm_joint1
    [-1.5708, 1.5708],   # arm_joint2
    [-1.5708, 1.5708],   # arm_joint3
    [-1.5708, 1.5708],   # arm_joint4
    [-1.5708, 3.14159],  # arm_joint5
    [-1.54,   0.0],      # grip_joint
], dtype=np.float32)

NUM_ACTUATORS = len(ACTUATOR_NAMES)


class X3PlusPickCubeEnv(gym.Env):
    """MuJoCo x3plus arm picking a cube, with LeRobot-compatible observations."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        model_path: str = "models/x3plus_mujoco.xml",
        camera_name: str = "front_cam",
        resolution: tuple[int, int] = (256, 256),
        timestep: float = 0.002,
        control_dt: float = 0.05,
        delta_scale: float = 0.05,
        max_episode_steps: int = 400,
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        self.render_mode = render_mode
        self.camera_name = camera_name
        self.delta_scale = delta_scale
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        xml_path = COMPONENT_DIR / model_path
        self.mj_model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.mj_model.opt.timestep = timestep
        self.mj_data = mujoco.MjData(self.mj_model)

        self.n_substeps = int(control_dt / timestep)
        h, w = resolution
        self._renderer = mujoco.Renderer(self.mj_model, height=h, width=w)

        self._joint_qpos_idx = np.array([
            self.mj_model.joint(name).qposadr[0] for name in ACTUATOR_NAMES
        ])

        self._cube_jnt_adr = self.mj_model.joint("cube_joint").qposadr[0]

        self.observation_space = spaces.Dict({
            "observation.images.front_cam": spaces.Box(
                0, 255, shape=(h, w, 3), dtype=np.uint8,
            ),
            "observation.state": spaces.Box(
                -np.inf, np.inf, shape=(NUM_ACTUATORS,), dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            -1.0, 1.0, shape=(NUM_ACTUATORS,), dtype=np.float32,
        )

        self._cube_init_pos = np.array([0.15, 0.0, 0.39])
        self._rng = np.random.default_rng()

    def _get_obs(self) -> dict[str, np.ndarray]:
        image = self._render_camera()
        state = self.mj_data.qpos[self._joint_qpos_idx].astype(np.float32).copy()
        return {
            "observation.images.front_cam": image,
            "observation.state": state,
        }

    def _render_camera(self) -> np.ndarray:
        self._renderer.update_scene(self.mj_data, camera=self.camera_name)
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        self._renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = True
        return self._renderer.render().copy()

    def _compute_reward(self) -> float:
        ee = self.mj_data.body("arm_link5").xpos
        cube = self.mj_data.body("target_cube").xpos
        return -float(np.linalg.norm(ee - cube))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.mj_model, self.mj_data)

        noise = self._rng.uniform(-0.03, 0.03, size=2)
        self.mj_data.qpos[self._cube_jnt_adr: self._cube_jnt_adr + 3] = (
            self._cube_init_pos + np.array([noise[0], noise[1], 0.0])
        )

        self.mj_data.ctrl[:NUM_ACTUATORS] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.mj_data.ctrl[:NUM_ACTUATORS].copy()
        ctrl += action * self.delta_scale
        ctrl = np.clip(ctrl, ACTUATOR_RANGES[:, 0], ACTUATOR_RANGES[:, 1])
        self.mj_data.ctrl[:NUM_ACTUATORS] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        ee = self.mj_data.body("arm_link5").xpos
        cube = self.mj_data.body("target_cube").xpos
        info = {
            "ee_pos": ee.copy(),
            "cube_pos": cube.copy(),
            "dist": float(np.linalg.norm(ee - cube)),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_camera()
        return None

    def close(self):
        self._renderer.close()
