"""Photogrammetry reconstruction wrapper around Meshroom's meshroom_batch CLI.

Provides both a callable function and a CLI entry point.  Accepts either a
directory of images or a video file as input — videos are automatically
extracted to frames via ffmpeg before reconstruction.

The default pipeline (photogrammetryObject) uses Grounding DINO + SAM to
detect and segment a target object before reconstruction.  Pass --prompt to
specify what to reconstruct (e.g. "red cup").

CLI usage:
    python -m src.reconstruct --name my_object
    python -m src.reconstruct --name my_object --input-dir /path/to/images
    python -m src.reconstruct --name red_cup --prompt "red cup"
    python -m src.reconstruct --name red_cup --fps 2   # video → frames at 2 fps

Programmatic usage:
    from src.reconstruct import run_reconstruction
    mesh_path = run_reconstruction("red_cup", prompt="red cup")
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yml"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_meshroom_batch() -> str:
    path = shutil.which("meshroom_batch")
    if path is None:
        sys.exit(
            "meshroom_batch not found on PATH. "
            "Run 'source activate_env.sh meshroom' first, "
            "or run './setup_envs.sh meshroom' to install."
        )
    return path


def _extract_frames(video_path: Path, dest_dir: Path, fps: float = 2.0) -> list[Path]:
    """Extract frames from a video file into *dest_dir* using ffmpeg."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required for video input but was not found on PATH.")

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dest_dir.glob("frame_*.jpg"))
    if existing:
        print(f"  Reusing {len(existing)} previously extracted frames in {dest_dir}")
        return existing

    print(f"  Extracting frames from {video_path.name} at {fps} fps ...")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(dest_dir / "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    frames = sorted(dest_dir.glob("frame_*.jpg"))
    print(f"  Extracted {len(frames)} frames")
    return frames


def _resolve_input(input_base: Path, name: str, fps: float) -> tuple[Path, list[Path]]:
    """Return (image_dir, image_list) — extracting from video if needed.

    Lookup order:
      1. ``<input_base>/<name>/`` directory with images already present
      2. ``<input_base>/<name>.{mp4,avi,...}`` video file → extract frames
    """
    obj_dir = input_base / name

    if obj_dir.is_dir():
        images = sorted(
            p for p in obj_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            return obj_dir, images

    for ext in VIDEO_EXTENSIONS:
        video = input_base / f"{name}{ext}"
        if video.is_file():
            obj_dir.mkdir(parents=True, exist_ok=True)
            frames = _extract_frames(video, obj_dir, fps=fps)
            if frames:
                return obj_dir, frames

    if obj_dir.is_dir():
        sys.exit(f"No images ({', '.join(IMAGE_EXTENSIONS)}) found in {obj_dir}")
    sys.exit(
        f"No input found for '{name}': expected a directory {obj_dir} with images, "
        f"or a video file {input_base / name}.mp4"
    )


def run_reconstruction(
    name: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    pipeline: str | None = None,
    prompt: str | None = None,
    force: bool | None = None,
    fps: float = 2.0,
) -> Path:
    """Run meshroom_batch and return the path to the output directory.

    Args:
        name: Object identifier. Images are read from ``<input_dir>/<name>/``
              (or extracted from ``<input_dir>/<name>.mp4``) and results
              written to ``<output_dir>/<name>/``.
        input_dir: Override for the base input directory (default from config).
        output_dir: Override for the base output directory (default from config).
        config_path: Path to the YAML config file.
        pipeline: Meshroom pipeline name (default from config).
        prompt: Text description of the object to detect/segment (e.g. "red cup").
                Used by ImageDetectionPrompt node (Grounding DINO).
                Falls back to config default, then to *name* if still empty.
        force: Force recomputation even if cache exists.
        fps: Frame extraction rate when input is a video (default 2.0).

    Returns:
        Path to the output directory for this object.
    """
    cfg = _load_config(config_path)
    defaults = cfg.get("defaults", {})

    input_base = Path(input_dir) if input_dir else SCRIPT_DIR / cfg["input_dir"]
    output_base = Path(output_dir) if output_dir else SCRIPT_DIR / cfg["output_dir"]
    pipeline = pipeline or cfg.get("pipeline", "photogrammetry")
    force = force if force is not None else defaults.get("force_compute", False)

    if prompt is None:
        prompt = defaults.get("prompt", "") or ""
    if not prompt:
        prompt = name.replace("_", " ")

    force_detection = defaults.get("force_detection", True)

    obj_input, images = _resolve_input(input_base, name, fps)
    obj_output = output_base / name
    obj_output.mkdir(parents=True, exist_ok=True)

    meshroom_bin = _find_meshroom_batch()

    cmd: list[str] = [
        meshroom_bin,
        "--input", str(obj_input),
        "--pipeline", pipeline,
        "--output", str(obj_output),
    ]

    param_overrides: list[str] = []
    if prompt:
        param_overrides.append(f"ImageDetectionPrompt:prompt={prompt}")
        param_overrides.append("ImageDetectionPrompt:synonyms=")
    if force_detection:
        param_overrides.append("ImageDetectionPrompt:forceDetection=True")
    if param_overrides:
        cmd.extend(["--paramOverrides"] + param_overrides)

    if force:
        cmd.append("--forceCompute")

    print(f"Running Meshroom {pipeline} pipeline")
    print(f"  Input:  {obj_input}  ({len(images)} images)")
    print(f"  Output: {obj_output}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True)

    print(f"Reconstruction complete: {obj_output}")
    return obj_output


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct a 3D mesh from images or video using Meshroom"
    )
    p.add_argument("--name", required=True, help="Object name (subdirectory or video under input/)")
    p.add_argument("--input-dir", type=Path, default=None, help="Override base input directory")
    p.add_argument("--output-dir", type=Path, default=None, help="Override base output directory")
    p.add_argument("--pipeline", type=str, default=None, help="Meshroom pipeline (default: from config)")
    p.add_argument("--prompt", type=str, default=None, help="Object description for detection/segmentation (e.g. 'red cup')")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")
    p.add_argument("--force", action="store_true", help="Force recomputation")
    p.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for video input (default: 2)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_reconstruction(
        args.name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        pipeline=args.pipeline,
        prompt=args.prompt,
        force=args.force,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
