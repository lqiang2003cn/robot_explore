"""Pick-and-place demo: Octo VLA controlling x3plus arm in MuJoCo.

Octo was trained on real-robot data (Open X-Embodiment) and has not been
finetuned on the x3plus.  Actions will therefore be semi-random; the demo
exists to prove that the full integration pipeline works end-to-end.

Run:
    source activate_env.sh vla_x3plus
    python -m src.pick_and_place [--random]

Pass --random to skip Octo loading and use random actions (useful for
testing the MuJoCo scene without GPU / large model download).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import yaml

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402 (after env var)

COMPONENT_DIR = pathlib.Path(__file__).resolve().parent.parent

ACTUATOR_RANGES = [
    (-1.5708, 1.5708),   # arm_joint1
    (-1.5708, 1.5708),   # arm_joint2
    (-1.5708, 1.5708),   # arm_joint3
    (-1.5708, 1.5708),   # arm_joint4
    (-1.5708, 3.14159),  # arm_joint5
    (-1.54,   0.0),      # grip_joint
]

NUM_ACTUATORS = len(ACTUATOR_RANGES)


def load_config() -> dict:
    with open(COMPONENT_DIR / "config.yml") as f:
        return yaml.safe_load(f)


# ── MuJoCo helpers ────────────────────────────────────────────


def create_sim(config: dict):
    model_path = COMPONENT_DIR / config["simulation"]["model_path"]
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = config["simulation"]["timestep"]
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    return model, data


def make_renderer(model: mujoco.MjModel, resolution: list[int]):
    return mujoco.Renderer(model, height=resolution[0], width=resolution[1])


def render_frame(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = True
    return renderer.render().copy()


# ── Octo helpers ──────────────────────────────────────────────


def load_octo(checkpoint: str):
    try:
        from octo.model.octo_model import OctoModel  # noqa: F811
    except ImportError:
        print(
            "WARNING: octo is not installed.  "
            "Falling back to random actions.  "
            "Install via:  ./setup_envs.sh vla_x3plus"
        )
        return None

    print(f"Loading Octo from {checkpoint} ...")
    model = OctoModel.load_pretrained(checkpoint)
    print("Octo model loaded.")
    return model


def octo_predict(octo_model, image: np.ndarray, task_text: str, rng_key):
    """Run a single Octo inference step, returning a 7-D action vector."""
    import jax  # noqa: F811

    observation = {
        "image_primary": image[np.newaxis, np.newaxis, ...],
        "timestep_pad_mask": np.array([[True]]),
    }
    task = octo_model.create_tasks(texts=[task_text])
    actions = octo_model.sample_actions(
        observation,
        task,
        unnormalization_statistics=octo_model.dataset_statistics[
            "bridge_dataset"
        ]["action"],
        rng=rng_key,
    )
    return np.asarray(actions[0, 0])  # shape (7,)


# ── Action mapping ────────────────────────────────────────────


def map_action_to_ctrl(
    action_7d: np.ndarray,
    current_ctrl: np.ndarray,
    delta_scale: float = 0.05,
) -> np.ndarray:
    """Map Octo 7-D delta to 6 x3plus actuator commands (no IK).

    Octo output convention: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    We apply the first 5 deltas directly to arm joints (scaled) and
    threshold the gripper dimension for open/close.
    """
    ctrl = current_ctrl.copy()

    for i in range(min(5, len(action_7d) - 1)):
        ctrl[i] += action_7d[i] * delta_scale

    gripper_cmd = action_7d[6] if len(action_7d) > 6 else 0.0
    ctrl[5] = -1.54 if gripper_cmd > 0.5 else 0.0

    for i in range(NUM_ACTUATORS):
        lo, hi = ACTUATOR_RANGES[i]
        ctrl[i] = np.clip(ctrl[i], lo, hi)

    return ctrl


# ── Main loop ─────────────────────────────────────────────────


def run(use_random: bool = False):
    config = load_config()
    sim_cfg = config["simulation"]

    model, data = create_sim(config)
    resolution = sim_cfg["camera_resolution"]
    camera_name = sim_cfg["camera_name"]
    max_steps = sim_cfg["max_episode_steps"]
    control_dt = sim_cfg["control_dt"]
    n_substeps = int(control_dt / model.opt.timestep)
    frame_skip = sim_cfg.get("frame_skip", 2)
    video_fps = sim_cfg.get("video_fps", 30)

    renderer = make_renderer(model, resolution)

    octo_model = None
    rng = None
    if not use_random:
        octo_model = load_octo(config["model"]["checkpoint"])
    if octo_model is not None:
        import jax

        rng = jax.random.PRNGKey(0)
    else:
        use_random = True
        np_rng = np.random.default_rng(42)

    output_dir = COMPONENT_DIR / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    task_text = "pick up the red cube"
    ctrl = np.zeros(NUM_ACTUATORS)
    frames: list[np.ndarray] = []

    print(f"Running pick-and-place demo  ({max_steps} steps, "
          f"{'random' if use_random else 'octo'} actions)")
    print(f"  Task : \"{task_text}\"")
    print(f"  Camera: {camera_name} @ {resolution[1]}x{resolution[0]}")
    print(f"  Video : {video_fps} fps, capture every {frame_skip} steps")

    for step in range(max_steps):
        need_image = not use_random or (step % frame_skip == 0)
        image = render_frame(renderer, data, camera_name) if need_image else None

        if step % frame_skip == 0:
            frames.append(image if image is not None
                          else render_frame(renderer, data, camera_name))

        if use_random:
            action = np_rng.standard_normal(7).astype(np.float32) * 0.3
        else:
            import jax

            rng, sub = jax.random.split(rng)
            action = octo_predict(octo_model, image, task_text, sub)

        ctrl = map_action_to_ctrl(action, ctrl)
        data.ctrl[:NUM_ACTUATORS] = ctrl

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)

        if step % 50 == 0:
            ee = data.body("arm_link5").xpos
            cube = data.body("target_cube").xpos
            print(
                f"  step {step:4d}  "
                f"ee=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}]  "
                f"cube=[{cube[0]:.3f},{cube[1]:.3f},{cube[2]:.3f}]"
            )

    frames.append(render_frame(renderer, data, camera_name))
    renderer.close()

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
        description="X3Plus pick-and-place demo with Octo VLA in MuJoCo"
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random actions instead of Octo (no GPU / model needed)",
    )
    args = parser.parse_args()
    run(use_random=args.random)


if __name__ == "__main__":
    main()
