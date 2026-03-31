"""Pick-and-place demo: LeRobot policy controlling x3plus arm in MuJoCo.

Policies are loaded via HuggingFace LeRobot (PyTorch). The x3plus MuJoCo
scene is wrapped as a Gymnasium env with a native 6-D joint-space action
space (5 arm joints + 1 gripper).

Run:
    source activate_env.sh vla_x3plus
    python -m src.pick_and_place [--random] [--policy PATH]

Pass --random to skip policy loading and use random actions (useful for
testing the MuJoCo scene without GPU / large model download).

Pass --policy to override the pretrained path from config.yml.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import torch
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

from src.env import X3PlusPickCubeEnv  # noqa: E402

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(COMPONENT_DIR / "config.yml") as f:
        return yaml.safe_load(f)


# ── Policy loading ────────────────────────────────────────────

POLICY_REGISTRY = {
    "smolvla": "lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy",
    "act": "lerobot.policies.act.modeling_act.ACTPolicy",
    "diffusion": "lerobot.policies.diffusion.modeling_diffusion.DiffusionPolicy",
}


def _import_policy_class(policy_type: str):
    """Dynamically import a LeRobot policy class by type name."""
    if policy_type not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy_type={policy_type!r}. "
            f"Available: {list(POLICY_REGISTRY)}"
        )
    module_path, class_name = POLICY_REGISTRY[policy_type].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_policy(config: dict, policy_path: str | None = None):
    """Load a pretrained LeRobot policy.

    Returns (policy, device) or (None, None) on failure.
    """
    model_cfg = config["model"]
    policy_type = model_cfg["policy_type"]
    pretrained = policy_path or model_cfg["pretrained_path"]

    try:
        PolicyClass = _import_policy_class(policy_type)
    except (ImportError, TypeError) as exc:
        print(
            f"WARNING: failed to import lerobot policy module for {policy_type!r}: {exc}. "
            "Falling back to random actions. "
            "Install/update via:  ./setup_envs.sh vla_x3plus"
        )
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {policy_type} policy from {pretrained} ...")
    policy = PolicyClass.from_pretrained(pretrained)
    policy.eval()
    policy.to(device)
    print(f"Policy loaded on {device}.")
    return policy, device


def obs_to_policy_input(
    obs: dict[str, np.ndarray],
    device: torch.device,
    image_keys: list[str] | None = None,
    lang_tokens: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert Gymnasium observation dict to batched torch tensors for the policy.

    If *image_keys* is provided (e.g. from the pretrained policy config), the
    single env camera image is broadcast to every expected image key so that
    pretrained models trained with different camera names still receive valid input.

    *lang_tokens* should contain pre-tokenized language keys
    (``observation.language.tokens`` and ``observation.language.attention_mask``).
    """
    image = obs["observation.images.front_cam"]
    image_t = (
        torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255.0).to(device)
    )

    state = obs["observation.state"]
    state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)

    batch: dict[str, torch.Tensor] = {"observation.state": state_t}
    for key in (image_keys or ["observation.images.front_cam"]):
        batch[key] = image_t
    if lang_tokens is not None:
        batch.update(lang_tokens)
    return batch


def tokenize_task(policy, task_text: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Tokenize a task instruction for a VLA policy.

    Returns a dict with ``observation.language.tokens`` and
    ``observation.language.attention_mask`` ready to merge into the policy batch.
    """
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    cfg = policy.config
    max_len = getattr(cfg, "tokenizer_max_length", 48)
    padding = getattr(cfg, "pad_language_to", "longest")

    encoded = tokenizer(
        [task_text],
        max_length=max_len,
        truncation=True,
        padding=padding,
        return_tensors="pt",
    )
    return {
        "observation.language.tokens": encoded["input_ids"].to(device),
        "observation.language.attention_mask": encoded["attention_mask"].to(device, dtype=torch.bool),
    }


# ── Main loop ─────────────────────────────────────────────────


def run(use_random: bool = False, policy_path: str | None = None):
    config = load_config()
    sim_cfg = config["simulation"]
    model_cfg = config["model"]

    env = X3PlusPickCubeEnv(
        model_path=sim_cfg["model_path"],
        camera_name=sim_cfg["camera_name"],
        resolution=tuple(sim_cfg["camera_resolution"]),
        timestep=sim_cfg["timestep"],
        control_dt=sim_cfg["control_dt"],
        max_episode_steps=sim_cfg["max_episode_steps"],
    )

    max_steps = sim_cfg["max_episode_steps"]
    frame_skip = sim_cfg.get("frame_skip", 2)
    video_fps = sim_cfg.get("video_fps", 30)
    resolution = sim_cfg["camera_resolution"]
    task_text = model_cfg.get("task", "pick up the red cube")

    policy = None
    device = None
    image_keys: list[str] | None = None
    lang_tokens: dict[str, torch.Tensor] | None = None
    if not use_random:
        policy, device = load_policy(config, policy_path)
    if policy is not None:
        cfg = getattr(policy, "config", None)
        if cfg is not None and hasattr(cfg, "image_features"):
            image_keys = list(cfg.image_features.keys())
            print(f"  Policy expects image keys: {image_keys}")
        if hasattr(policy, "model") and hasattr(policy.model, "vlm_with_expert"):
            lang_tokens = tokenize_task(policy, task_text, device)
            print(f"  Language tokens shape: {lang_tokens['observation.language.tokens'].shape}")
    if policy is None:
        use_random = True
        np_rng = np.random.default_rng(42)

    output_dir = COMPONENT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_name = "random" if use_random else model_cfg["policy_type"]
    print(
        f"Running pick-and-place demo  ({max_steps} steps, {policy_name} actions)"
    )
    print(f"  Task : \"{task_text}\"")
    print(f"  Camera: {env.camera_name} @ {resolution[0]}x{resolution[1]}")
    print(f"  Video : {video_fps} fps, capture every {frame_skip} steps")

    obs, info = env.reset(seed=0)
    frames: list[np.ndarray] = []

    for step in range(max_steps):
        if step % frame_skip == 0:
            frames.append(obs["observation.images.front_cam"])

        if use_random:
            action = np_rng.standard_normal(6).astype(np.float32) * 0.3
        else:
            with torch.no_grad():
                policy_input = obs_to_policy_input(obs, device, image_keys, lang_tokens)
                action_t = policy.select_action(policy_input)
            action = action_t.squeeze(0).cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(action)

        if step % 50 == 0:
            ee = info["ee_pos"]
            cube = info["cube_pos"]
            print(
                f"  step {step:4d}  "
                f"ee=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                f"cube=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]"
            )

        if terminated or truncated:
            break

    frames.append(obs["observation.images.front_cam"])
    env.close()

    video_path = output_dir / "pick_and_place_demo.mp4"
    print(f"Saving video ({len(frames)} frames, {video_fps} fps) -> {video_path}")

    import imageio

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


def main():
    parser = argparse.ArgumentParser(
        description="X3Plus pick-and-place demo with LeRobot policy in MuJoCo"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions instead of a policy (no GPU / model needed)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Override pretrained policy path (HF Hub ID or local path)",
    )
    args = parser.parse_args()
    run(use_random=args.random, policy_path=args.policy)


if __name__ == "__main__":
    main()
