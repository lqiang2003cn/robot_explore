"""3D Gaussian Splatting reconstruction using nerfstudio.

Runs COLMAP for camera pose estimation, trains a splatfacto model, and
extracts a mesh via TSDF fusion.  Accepts either a directory of images or a
video file as input.

When ``--prompt`` is provided, Grounding DINO + SAM2 video tracking segment
the target object first, producing per-frame masks that splatfacto uses to
ignore the background during training (transparency carving).

CLI usage:
    python -m src.reconstruct --name my_object
    python -m src.reconstruct --name red_cup --prompt "red cup"
    python -m src.reconstruct --name red_cup --iterations 7000
    python -m src.reconstruct --name red_cup --fps 2   # video → frames

Programmatic usage:
    from src.reconstruct import run_reconstruction
    mesh_path = run_reconstruction("red_cup", prompt="red cup")
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yml"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_input(input_base: Path, name: str) -> tuple[Path, str]:
    """Return (input_path, input_type) where type is 'images' or 'video'."""
    obj_dir = input_base / name
    if obj_dir.is_dir():
        images = [p for p in obj_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        if images:
            return obj_dir, "images"

    for ext in VIDEO_EXTENSIONS:
        video = input_base / f"{name}{ext}"
        if video.is_file():
            return video, "video"

    if obj_dir.is_dir():
        sys.exit(f"No images ({', '.join(IMAGE_EXTENSIONS)}) found in {obj_dir}")
    sys.exit(
        f"No input found for '{name}': expected a directory {obj_dir} with images, "
        f"or a video file {input_base / name}.mp4"
    )


def _run_cmd(cmd: list[str], description: str) -> None:
    border = "=" * 60
    print(f"\n{border}")
    print(f"  {description}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{border}\n")
    subprocess.run(cmd, check=True)


def _find_training_config(models_dir: Path) -> Path:
    """Find the latest training config.yml in nerfstudio's output tree."""
    configs = sorted(
        models_dir.rglob("config.yml"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not configs:
        sys.exit(f"No training config.yml found in {models_dir}. Training may have failed.")
    return configs[0]


def _find_mesh(mesh_dir: Path) -> Path | None:
    """Find the exported mesh file (OBJ preferred, then PLY)."""
    for pattern in ("*.obj", "*.ply"):
        matches = sorted(mesh_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _inject_mask_paths(transforms_path: Path, masks_dir: Path) -> None:
    """Add ``mask_path`` entries to each frame in *transforms_path*.

    Nerfstudio expects ``mask_path`` relative to the data directory
    (the parent of ``transforms.json``).  Masks are stored under
    ``<output>/<name>/masks/`` and symlinked/copied into the processed
    directory so the relative paths resolve correctly.
    """
    with open(transforms_path) as f:
        data = json.load(f)

    processed_dir = transforms_path.parent
    local_masks = processed_dir / "masks"
    if not local_masks.exists():
        local_masks.symlink_to(masks_dir.resolve())

    modified = False
    for frame in data.get("frames", []):
        file_path = frame.get("file_path", "")
        stem = Path(file_path).stem
        mask_rel = f"masks/{stem}.png"
        mask_abs = processed_dir / mask_rel
        if mask_abs.exists() and frame.get("mask_path") != mask_rel:
            frame["mask_path"] = mask_rel
            modified = True

    if modified:
        with open(transforms_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Injected mask_path into {len(data.get('frames', []))} frames")


def run_reconstruction(
    name: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    iterations: int | None = None,
    target_faces: int | None = None,
    fps: float = 2.0,
    force: bool = False,
    prompt: str | None = None,
    reference_frame: int | None = None,
    box_threshold: float | None = None,
) -> Path:
    """Run the full 3DGS pipeline and return the path to the exported mesh.

    Pipeline (when *prompt* is provided):
      0. GDINO + SAM2 — object detection and mask propagation
      1. ``ns-process-data`` — COLMAP SfM for camera poses
      1.5. Inject ``mask_path`` into ``transforms.json``
      2. ``ns-train splatfacto`` — 3DGS training (with transparency carving)
      3. ``ns-export tsdf`` — TSDF fusion mesh extraction

    Without *prompt*, steps 0 and 1.5 are skipped (full-scene reconstruction).

    Args:
        name: Object identifier.  Images are read from ``<input_dir>/<name>/``
              (or video from ``<input_dir>/<name>.mp4``) and results written
              to ``<output_dir>/<name>/``.
        input_dir: Override for the base input directory (default from config).
        output_dir: Override for the base output directory (default from config).
        config_path: Path to the YAML config file.
        iterations: Training iterations (default from config, typically 7000).
        target_faces: Target face count for mesh decimation (default 50000).
        fps: Frame extraction rate when input is a video.
        force: If True, rerun all steps even if output exists.
        prompt: Text description of the target object for segmentation.
                When set, enables GDINO + SAM2 masking.
        reference_frame: Frame index for GDINO detection (default: middle).
        box_threshold: GDINO box confidence threshold (default from config).

    Returns:
        Path to the exported mesh file.
    """
    cfg = _load_config(config_path)
    defaults = cfg.get("defaults", {})

    input_base = Path(input_dir) if input_dir else SCRIPT_DIR / cfg["input_dir"]
    output_base = Path(output_dir) if output_dir else SCRIPT_DIR / cfg["output_dir"]
    iterations = iterations or defaults.get("max_iterations", 7000)
    target_faces = target_faces or defaults.get("target_faces", 50000)
    force = force if force is not None else defaults.get("force_compute", False)

    input_path, input_type = _resolve_input(input_base, name)

    obj_output = output_base / name
    processed_dir = obj_output / "processed"
    models_dir = obj_output / "models"
    mesh_dir = obj_output / "mesh"

    use_masks = prompt is not None
    total_steps = 4 if use_masks else 3
    step = 0

    existing_mesh = _find_mesh(mesh_dir) if mesh_dir.exists() else None
    if existing_mesh and not force:
        print(f"Mesh already exists: {existing_mesh}")
        print("Use --force to recompute.")
        return existing_mesh

    print(f"3DGS Reconstruction: {name}")
    print(f"  Input:        {input_path} ({input_type})")
    print(f"  Output:       {obj_output}")
    print(f"  Iterations:   {iterations}")
    print(f"  Target faces: {target_faces}")
    if use_masks:
        print(f"  Prompt:       {prompt!r}")

    # ── Step 0 (optional): Segment object ─────────────────────
    masks_dir = None
    if use_masks:
        from src.segment import segment_object

        step += 1
        print(f"\n{'=' * 60}")
        print(f"  Step {step}/{total_steps}: Object segmentation (GDINO + SAM2)")
        print(f"{'=' * 60}\n")

        masks_dir = segment_object(
            name,
            prompt,
            input_dir=input_dir,
            output_dir=output_dir,
            config_path=config_path,
            reference_frame=reference_frame,
            box_threshold=box_threshold,
            force=force,
        )

    # ── Step 1: COLMAP via ns-process-data ─────────────────────
    step += 1
    transforms = processed_dir / "transforms.json"
    if transforms.exists() and not force:
        print(f"\nStep {step}/{total_steps}: COLMAP data already processed, skipping.")
    else:
        process_cmd = [
            "ns-process-data", input_type,
            "--data", str(input_path),
            "--output-dir", str(processed_dir),
            "--no-gpu",
        ]
        if input_type == "video":
            process_cmd.extend(["--num-frames-target", str(int(30 / fps * 10))])
        _run_cmd(process_cmd, f"Step {step}/{total_steps}: Running COLMAP (camera pose estimation)")

    # ── Step 1.5 (optional): Inject mask paths ────────────────
    if use_masks and masks_dir is not None:
        print(f"\n  Injecting mask paths into transforms.json...")
        _inject_mask_paths(transforms, masks_dir)

    # ── Step 2: Train splatfacto ──────────────────────────────
    step += 1
    train_cmd = [
        "ns-train", "splatfacto",
        "--data", str(processed_dir),
        "--output-dir", str(models_dir),
        "--max-num-iterations", str(iterations),
        "--vis", "tensorboard",
    ]
    if use_masks:
        train_cmd.extend(["--pipeline.model.background-color", "random"])
    _run_cmd(train_cmd, f"Step {step}/{total_steps}: Training 3D Gaussian Splatting")

    # ── Step 3: Export mesh via TSDF fusion ────────────────────
    step += 1
    train_config = _find_training_config(models_dir)
    export_cmd = [
        "ns-export", "tsdf",
        "--load-config", str(train_config),
        "--output-dir", str(mesh_dir),
        "--target-num-faces", str(target_faces),
    ]
    _run_cmd(export_cmd, f"Step {step}/{total_steps}: Extracting mesh via TSDF fusion")

    mesh_file = _find_mesh(mesh_dir)
    if mesh_file:
        print(f"\nReconstruction complete: {mesh_file}")
        return mesh_file

    sys.exit(
        f"No mesh file found in {mesh_dir} after export. "
        f"Contents: {list(mesh_dir.iterdir()) if mesh_dir.exists() else '(directory missing)'}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Reconstruct a 3D mesh from images/video using 3D Gaussian Splatting"
    )
    p.add_argument("--name", required=True, help="Object name (subdirectory or video under input/)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Object description for GDINO+SAM2 segmentation (e.g. 'red cup'). "
                        "Enables masked reconstruction of the target object only.")
    p.add_argument("--input-dir", type=Path, default=None, help="Override base input directory")
    p.add_argument("--output-dir", type=Path, default=None, help="Override base output directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")
    p.add_argument("--iterations", type=int, default=None, help="Training iterations (default from config)")
    p.add_argument("--target-faces", type=int, default=None, help="Target mesh face count (default from config)")
    p.add_argument("--fps", type=float, default=2.0, help="Frame extraction rate for video input")
    p.add_argument("--reference-frame", type=int, default=None,
                   help="Frame index for GDINO detection (default: middle)")
    p.add_argument("--box-threshold", type=float, default=None,
                   help="GDINO box confidence threshold (default from config)")
    p.add_argument("--force", action="store_true", help="Force recomputation of all steps")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run_reconstruction(
        args.name,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        iterations=args.iterations,
        target_faces=args.target_faces,
        fps=args.fps,
        force=args.force,
        prompt=args.prompt,
        reference_frame=args.reference_frame,
        box_threshold=args.box_threshold,
    )


if __name__ == "__main__":
    main()
