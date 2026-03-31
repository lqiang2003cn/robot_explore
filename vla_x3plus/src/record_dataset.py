"""Record a LeRobot-format dataset from the scripted pick-and-place controller.

Usage:
    source activate_env.sh vla_x3plus
    python -m src.record_dataset --num-episodes 100
    python -m src.record_dataset --num-episodes 5 --repo-id local/x3plus_test
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

from src.env import ACTUATOR_NAMES, X3PlusPickCubeEnv  # noqa: E402
from src.scripted_controller import ScriptedPickPlace  # noqa: E402

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent

FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": list(ACTUATOR_NAMES),
    },
    "observation.images.front_cam": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": list(ACTUATOR_NAMES),
    },
}


def load_config() -> dict:
    with open(COMPONENT_DIR / "config.yml") as f:
        return yaml.safe_load(f)


def record(
    num_episodes: int,
    repo_id: str,
    root: str | None = None,
) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = load_config()
    sim_cfg = config["simulation"]
    ds_cfg = config.get("dataset", {})
    task_text = config["model"].get("task", "pick up the red cube")
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

    fps = int(1.0 / sim_cfg["control_dt"])

    create_kwargs: dict = dict(
        repo_id=repo_id,
        fps=fps,
        robot_type="x3plus",
        features=FEATURES,
        use_videos=True,
    )
    if root is not None:
        create_kwargs["root"] = root

    dataset = LeRobotDataset.create(**create_kwargs)
    controller = ScriptedPickPlace(env)

    success_count = 0
    episode_lengths: list[int] = []
    t0 = time.time()

    for ep in range(num_episodes):
        obs, info = env.reset(seed=ep)
        cube_pos = env.mj_data.body("target_cube").xpos.copy()
        controller.reset(cube_pos, env.place_position)

        ep_success = False
        step = 0

        for step in range(sim_cfg["max_episode_steps"]):
            action = controller()

            frame = {
                "observation.state": obs["observation.state"],
                "observation.images.front_cam": obs["observation.images.front_cam"],
                "action": action,
                "task": task_text,
            }
            dataset.add_frame(frame)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                ep_success = True
                break
            if truncated or controller.done:
                break

        dataset.save_episode()

        if ep_success:
            success_count += 1
        episode_lengths.append(step + 1)

        status = "OK" if ep_success else "FAIL"
        elapsed = time.time() - t0
        print(
            f"  Episode {ep + 1:4d}/{num_episodes}  "
            f"steps={step + 1:4d}  {status}  "
            f"[{elapsed:.1f}s elapsed]"
        )

    env.close()
    dataset.finalize()

    avg_len = np.mean(episode_lengths) if episode_lengths else 0
    print("\n--- Dataset recording complete ---")
    print(f"  Episodes : {num_episodes}")
    print(f"  Success  : {success_count}/{num_episodes} ({100 * success_count / max(num_episodes, 1):.1f}%)")
    print(f"  Avg steps: {avg_len:.1f}")
    print(f"  Repo ID  : {repo_id}")
    print(f"  Elapsed  : {time.time() - t0:.1f}s")


def main() -> None:
    config = load_config()
    ds_cfg = config.get("dataset", {})

    parser = argparse.ArgumentParser(description="Record LeRobot dataset from scripted controller")
    parser.add_argument(
        "--num-episodes", type=int,
        default=ds_cfg.get("num_episodes", 100),
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--repo-id", type=str,
        default=ds_cfg.get("repo_id", "local/x3plus_pick_cube"),
        help="LeRobot dataset repo ID (local/... for local-only)",
    )
    parser.add_argument(
        "--root", type=str, default=None,
        help="Override dataset root directory",
    )
    args = parser.parse_args()

    print(f"Recording {args.num_episodes} episodes -> {args.repo_id}")
    record(
        num_episodes=args.num_episodes,
        repo_id=args.repo_id,
        root=args.root,
    )


if __name__ == "__main__":
    main()
