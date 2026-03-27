"""Simulation runner for the robot exploration environment (Isaac Sim 6.0.0)."""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import yaml
import numpy as np

import isaacsim
from isaacsim.simulation_app import SimulationApp


class SimConfig:
    """Load and validate simulation parameters from YAML."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    @property
    def world_size(self) -> tuple[float, float]:
        ws = self.cfg.get("world_size", [10.0, 10.0])
        return (ws[0], ws[1])

    @property
    def time_step(self) -> float:
        return self.cfg.get("time_step", 1 / 60)

    @property
    def max_steps(self) -> int:
        return self.cfg.get("max_steps", 10000)

    @property
    def headless(self) -> bool:
        return self.cfg.get("headless", True)

    @property
    def renderer(self) -> str:
        return self.cfg.get("renderer", "RayTracedLighting")

    @property
    def robot_usd(self) -> str:
        return self.cfg["robot"]["usd_path"]

    @property
    def start_position(self) -> list[float]:
        return self.cfg["robot"]["start_position"]

    @property
    def goal_position(self) -> list[float]:
        return self.cfg["robot"]["goal_position"]


class SimRunner:
    """Isaac Sim based simulation loop."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.step_count = 0
        self.app = SimulationApp({"headless": config.headless})
        self._setup_scene()

    def _setup_scene(self):
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.core.robots import Robot

        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=self.config.time_step,
            rendering_dt=self.config.time_step,
        )

        self.world.scene.add_default_ground_plane()

        add_reference_to_stage(
            usd_path=self.config.robot_usd,
            prim_path="/World/Robot",
        )
        self.robot = self.world.scene.add(
            Robot(prim_path="/World/Robot", name="exploration_robot")
        )
        self.robot.set_world_pose(
            position=np.array(self.config.start_position),
        )

        self.world.reset()

    def reset(self):
        self.step_count = 0
        self.world.reset()
        self.robot.set_world_pose(
            position=np.array(self.config.start_position),
        )

    def step(self) -> dict:
        """Advance the simulation by one physics step.

        Returns a dict with at least 'position' and 'done' keys.
        """
        self.world.step(render=not self.config.headless)
        self.step_count += 1

        position, _ = self.robot.get_world_pose()
        goal = np.array(self.config.goal_position)
        reached = bool(np.linalg.norm(position - goal) < 0.5)
        done = reached or self.step_count >= self.config.max_steps

        return {
            "position": position,
            "done": done,
            "reached_goal": reached,
            "step": self.step_count,
        }

    def close(self):
        self.app.close()
