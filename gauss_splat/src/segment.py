"""Object segmentation via Grounding DINO + SAM2 video tracking.

Detects a target object with Grounding DINO on a single reference frame,
then propagates the mask across all frames using SAM2's video predictor
for temporally consistent per-frame masks.

CLI usage:
    python -m src.segment --name red_cup --prompt "red cup"
    python -m src.segment --name red_cup --prompt "red cup" --reference-frame 5

Programmatic usage:
    from src.segment import segment_object
    masks_dir = segment_object("red_cup", prompt="red cup")
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _load_config(path: Path = DEFAULT_CONFIG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_frames(frame_dir: Path) -> list[Path]:
    """Return sorted list of image files in *frame_dir*."""
    frames = sorted(
        p for p in frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        sys.exit(f"No images found in {frame_dir}")
    return frames


def _detect_object(
    image: Image.Image,
    prompt: str,
    *,
    model_id: str = "IDEA-Research/grounding-dino-base",
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run Grounding DINO and return the best bounding box as [x1, y1, x2, y2].

    Returns pixel-coordinate box (not normalized).
    """
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    # GDINO expects lowercase prompt ending with a period
    text = prompt.lower().strip()
    if not text.endswith("."):
        text += "."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],  # (h, w)
    )[0]

    if len(results["boxes"]) == 0:
        sys.exit(
            f"Grounding DINO found no detections for prompt '{prompt}' "
            f"(box_threshold={box_threshold}, text_threshold={text_threshold}). "
            f"Try lowering thresholds or changing the prompt."
        )

    best_idx = results["scores"].argmax().item()
    box = results["boxes"][best_idx].cpu().numpy()  # [x1, y1, x2, y2]
    score = results["scores"][best_idx].item()
    label = results["labels"][best_idx]
    print(f"  Detected '{label}' (score={score:.3f}), box={box.astype(int).tolist()}")
    return box


def _propagate_masks(
    frame_dir: Path,
    frames: list[Path],
    reference_idx: int,
    box: np.ndarray,
    *,
    model_id: str = "facebook/sam2.1-hiera-large",
) -> dict[int, np.ndarray]:
    """Run SAM2 video predictor: prompt with *box* on *reference_idx*, propagate.

    Returns {frame_index: binary_mask} for every frame.
    """
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    # SAM2 expects purely numeric filenames (e.g. 00000.jpg).
    # Create a temp dir with numeric symlinks so init_state can sort them.
    tmp_dir = Path(tempfile.mkdtemp(prefix="sam2_frames_"))
    try:
        for i, f in enumerate(frames):
            (tmp_dir / f"{i:05d}.jpg").symlink_to(f.resolve())

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = SAM2VideoPredictor.from_pretrained(model_id, device_map=device)

        with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(tmp_dir))

            box_tensor = torch.tensor(box, dtype=torch.float32).unsqueeze(0)
            _, _, _ = predictor.add_new_points_or_box(
                state,
                frame_idx=reference_idx,
                obj_id=0,
                box=box_tensor,
            )

            masks: dict[int, np.ndarray] = {}
            for frame_idx, _, preds in predictor.propagate_in_video(state):
                mask = (preds[0].cpu().squeeze().numpy() > 0.0).astype(np.uint8)
                masks[frame_idx] = mask
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return masks


def segment_object(
    name: str,
    prompt: str,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    reference_frame: int | None = None,
    box_threshold: float | None = None,
    text_threshold: float | None = None,
    force: bool = False,
) -> Path:
    """Detect and segment an object across all frames, return masks directory.

    Pipeline:
      1. Grounding DINO detects the object on a reference frame (bounding box).
      2. SAM2 video predictor propagates the mask across all frames.
      3. Binary masks (255=object, 0=background) saved as grayscale PNGs.

    Args:
        name: Object identifier (subdirectory under input/).
        prompt: Text description for Grounding DINO (e.g. "red cup").
        input_dir: Override base input directory.
        output_dir: Override base output directory.
        config_path: Path to config.yml.
        reference_frame: Frame index to run detection on (default: middle).
        box_threshold: GDINO box confidence threshold.
        text_threshold: GDINO text similarity threshold.
        force: Recompute even if masks exist.

    Returns:
        Path to the masks directory.
    """
    cfg = _load_config(config_path)
    seg_cfg = cfg.get("segmentation", {})

    input_base = Path(input_dir) if input_dir else SCRIPT_DIR / cfg["input_dir"]
    output_base = Path(output_dir) if output_dir else SCRIPT_DIR / cfg["output_dir"]

    gdino_model = seg_cfg.get("gdino_model", "IDEA-Research/grounding-dino-base")
    sam2_model = seg_cfg.get("sam2_model", "facebook/sam2.1-hiera-large")
    box_threshold = box_threshold if box_threshold is not None else seg_cfg.get("box_threshold", 0.35)
    text_threshold = text_threshold if text_threshold is not None else seg_cfg.get("text_threshold", 0.25)

    frame_dir = input_base / name
    if not frame_dir.is_dir():
        sys.exit(f"Input directory not found: {frame_dir}")

    frames = _collect_frames(frame_dir)
    masks_dir = output_base / name / "masks"

    if masks_dir.exists() and not force:
        existing = list(masks_dir.glob("*.png"))
        if len(existing) >= len(frames):
            print(f"Masks already exist ({len(existing)} files): {masks_dir}")
            print("Use --force to recompute.")
            return masks_dir

    masks_dir.mkdir(parents=True, exist_ok=True)

    ref_idx = reference_frame if reference_frame is not None else len(frames) // 2
    ref_idx = max(0, min(ref_idx, len(frames) - 1))

    print(f"Segmentation: {name}")
    print(f"  Prompt:          {prompt!r}")
    print(f"  Frames:          {len(frames)} in {frame_dir}")
    print(f"  Reference frame: {ref_idx} ({frames[ref_idx].name})")
    print(f"  Output:          {masks_dir}")

    # Step 1: Detect object on reference frame
    print(f"\n  Running Grounding DINO on frame {ref_idx}...")
    ref_image = Image.open(frames[ref_idx]).convert("RGB")
    box = _detect_object(
        ref_image,
        prompt,
        model_id=gdino_model,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Step 2: Propagate masks via SAM2 video tracker
    print(f"\n  Running SAM2 video propagation across {len(frames)} frames...")
    masks = _propagate_masks(
        frame_dir, frames, ref_idx, box, model_id=sam2_model,
    )

    # Step 3: Save masks
    saved = 0
    for idx, frame_path in enumerate(frames):
        mask_name = frame_path.stem + ".png"
        mask_path = masks_dir / mask_name
        if idx in masks:
            mask_uint8 = masks[idx] * 255
            Image.fromarray(mask_uint8, mode="L").save(mask_path)
            saved += 1
        else:
            h, w = ref_image.size[::-1]
            Image.fromarray(np.zeros((h, w), dtype=np.uint8), mode="L").save(mask_path)

    print(f"\n  Saved {saved} masks to {masks_dir}")
    return masks_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Segment an object from multi-view images using GDINO + SAM2"
    )
    p.add_argument("--name", required=True, help="Object name (subdirectory under input/)")
    p.add_argument("--prompt", required=True, help="Text description for detection (e.g. 'red cup')")
    p.add_argument("--input-dir", type=Path, default=None, help="Override base input directory")
    p.add_argument("--output-dir", type=Path, default=None, help="Override base output directory")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file path")
    p.add_argument("--reference-frame", type=int, default=None, help="Frame index for detection (default: middle)")
    p.add_argument("--box-threshold", type=float, default=None, help="GDINO box confidence threshold")
    p.add_argument("--text-threshold", type=float, default=None, help="GDINO text similarity threshold")
    p.add_argument("--force", action="store_true", help="Force recomputation")
    return p


def main() -> None:
    args = build_parser().parse_args()
    segment_object(
        args.name,
        args.prompt,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        reference_frame=args.reference_frame,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        force=args.force,
    )


if __name__ == "__main__":
    main()
