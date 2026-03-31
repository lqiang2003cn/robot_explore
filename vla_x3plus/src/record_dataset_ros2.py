"""Record a LeRobot-format dataset by subscribing to ROS2 topics.

Independent of both ros2_stack (controller) and the MuJoCo bridge internals.
Subscribes to /joint_states, /front_cam/image_raw, /joint_command, and
/task_complete. Records observation/action pairs at 20 Hz into a LeRobot
dataset with absolute joint positions as actions.

Usage:
    source /opt/ros/jazzy/setup.bash
    source activate_env.sh vla_x3plus
    python -m src.record_dataset_ros2 --num-episodes 100
    python -m src.record_dataset_ros2 --num-episodes 5 --repo-id local/x3plus_test
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent

ACTUATOR_NAMES = [
    "arm_joint1", "arm_joint2", "arm_joint3",
    "arm_joint4", "arm_joint5", "grip_joint",
]

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


class DatasetRecorderNode(Node):
    """ROS2 node that records topic data into a LeRobot dataset."""

    def __init__(
        self,
        num_episodes: int,
        repo_id: str,
        task_text: str,
        fps: int = 20,
        root: str | None = None,
    ):
        super().__init__("dataset_recorder")

        self._num_episodes = num_episodes
        self._repo_id = repo_id
        self._task_text = task_text
        self._fps = fps
        self._root = root

        self._latest_joint_state: np.ndarray | None = None
        self._latest_image: np.ndarray | None = None
        self._latest_command: np.ndarray | None = None
        self._task_complete = False

        self._lock = threading.Lock()

        cb = ReentrantCallbackGroup()

        self.create_subscription(
            JointState, "/joint_states", self._on_joint_states, 10,
            callback_group=cb,
        )
        self.create_subscription(
            Image, "/front_cam/image_raw", self._on_image, 10,
            callback_group=cb,
        )
        self.create_subscription(
            JointState, "/joint_command", self._on_joint_command, 10,
            callback_group=cb,
        )
        self.create_subscription(
            Bool, "/task_complete", self._on_task_complete, 10,
            callback_group=cb,
        )

    def _on_joint_states(self, msg: JointState) -> None:
        state = np.zeros(6, dtype=np.float32)
        name_to_idx = {n: i for i, n in enumerate(ACTUATOR_NAMES)}
        for name, pos in zip(msg.name, msg.position):
            idx = name_to_idx.get(name)
            if idx is not None:
                state[idx] = float(pos)
        with self._lock:
            self._latest_joint_state = state

    def _on_image(self, msg: Image) -> None:
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
        with self._lock:
            self._latest_image = arr

    def _on_joint_command(self, msg: JointState) -> None:
        cmd = np.zeros(6, dtype=np.float32)
        name_to_idx = {n: i for i, n in enumerate(ACTUATOR_NAMES)}
        for name, pos in zip(msg.name, msg.position):
            idx = name_to_idx.get(name)
            if idx is not None:
                cmd[idx] = float(pos)
        with self._lock:
            self._latest_command = cmd

    def _on_task_complete(self, msg: Bool) -> None:
        if msg.data:
            with self._lock:
                self._task_complete = True

    def get_snapshot(self) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        with self._lock:
            return (
                self._latest_joint_state.copy() if self._latest_joint_state is not None else None,
                self._latest_image.copy() if self._latest_image is not None else None,
                self._latest_command.copy() if self._latest_command is not None else None,
            )

    def is_task_complete(self) -> bool:
        with self._lock:
            return self._task_complete

    def clear_task_complete(self) -> None:
        with self._lock:
            self._task_complete = False


def load_config() -> dict:
    with open(COMPONENT_DIR / "config.yml") as f:
        return yaml.safe_load(f)


def record(node: DatasetRecorderNode, executor: MultiThreadedExecutor) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    create_kwargs: dict = dict(
        repo_id=node._repo_id,
        fps=node._fps,
        robot_type="x3plus",
        features=FEATURES,
        use_videos=True,
    )
    if node._root is not None:
        create_kwargs["root"] = node._root

    dataset = LeRobotDataset.create(**create_kwargs)

    logger = node.get_logger()
    logger.info(f"Waiting for topics (/joint_states, /front_cam/image_raw, /joint_command) ...")

    while rclpy.ok():
        state, image, cmd = node.get_snapshot()
        if state is not None and image is not None:
            break
        time.sleep(0.1)

    logger.info("Topics active — starting recording")
    interval = 1.0 / node._fps
    success_count = 0
    t0 = time.time()

    for ep in range(node._num_episodes):
        node.clear_task_complete()
        step = 0

        while rclpy.ok():
            tick_start = time.time()

            state, image, cmd = node.get_snapshot()
            if state is None or image is None:
                time.sleep(interval)
                continue

            action = cmd if cmd is not None else np.zeros(6, dtype=np.float32)

            frame = {
                "observation.state": state,
                "observation.images.front_cam": image,
                "action": action,
                "task": node._task_text,
            }
            dataset.add_frame(frame)
            step += 1

            if node.is_task_complete():
                success_count += 1
                break

            elapsed_tick = time.time() - tick_start
            sleep_time = interval - elapsed_tick
            if sleep_time > 0:
                time.sleep(sleep_time)

        dataset.save_episode()

        elapsed = time.time() - t0
        status = "OK" if node.is_task_complete() else "TIMEOUT"
        logger.info(
            f"  Episode {ep + 1:4d}/{node._num_episodes}  "
            f"steps={step:4d}  {status}  [{elapsed:.1f}s elapsed]"
        )

    dataset.finalize()

    logger.info("\n--- Dataset recording complete ---")
    logger.info(f"  Episodes : {node._num_episodes}")
    logger.info(f"  Success  : {success_count}/{node._num_episodes}")
    logger.info(f"  Repo ID  : {node._repo_id}")
    logger.info(f"  Elapsed  : {time.time() - t0:.1f}s")


def main() -> None:
    config = load_config()
    ds_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})

    parser = argparse.ArgumentParser(description="Record LeRobot dataset from ROS2 topics")
    parser.add_argument(
        "--num-episodes", type=int,
        default=ds_cfg.get("num_episodes", 100),
    )
    parser.add_argument(
        "--repo-id", type=str,
        default=ds_cfg.get("repo_id_ros2", "local/x3plus_ros2"),
    )
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument(
        "--task", type=str,
        default=model_cfg.get("task", "pick up the red cube"),
    )
    args = parser.parse_args()

    rclpy.init()
    node = DatasetRecorderNode(
        num_episodes=args.num_episodes,
        repo_id=args.repo_id,
        task_text=args.task,
        fps=int(1.0 / config["simulation"].get("control_dt", 0.05)),
        root=args.root,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        record(node, executor)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
