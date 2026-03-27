"""Import a reconstructed OBJ mesh into Isaac Sim as a USD asset.

Converts OBJ files from Meshroom or gauss_splat output into USD format using
Isaac Sim's asset converter, and optionally places the asset in the current stage.

Usage:
    python -m src.import_mesh --name my_object
    python -m src.import_mesh --name my_object --source gauss_splat
    python -m src.import_mesh --name my_object --position 0.4 0.0 0.45
    python -m src.import_mesh --name my_object --reconstruction-output /custom/path
"""

import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

import argparse
import asyncio
from pathlib import Path

import isaacsim
from isaacsim.simulation_app import SimulationApp

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_ASSETS_DIR = SCRIPT_DIR / "assets"

RECONSTRUCTION_SOURCES = {
    "meshroom": PROJECT_ROOT / "meshroom" / "output",
    "gauss_splat": PROJECT_ROOT / "gauss_splat" / "output",
}

MESH_EXTENSIONS = {".obj", ".ply", ".glb", ".gltf"}


def find_mesh_file(output_base: Path, name: str, source: str) -> Path:
    """Locate the reconstructed mesh in the output tree.

    Search order depends on the source:
    - meshroom: Texturing/texturedMesh.obj, then generic OBJ search
    - gauss_splat: mesh/mesh.obj, mesh/*.ply, then generic search
    """
    obj_dir = output_base / name

    if source == "meshroom":
        candidates = [
            obj_dir / "Texturing" / "texturedMesh.obj",
            obj_dir / "texturedMesh.obj",
            obj_dir / "mesh.obj",
        ]
    else:
        candidates = [
            obj_dir / "mesh" / "mesh.obj",
            obj_dir / "mesh" / "tsdf_mesh.ply",
        ]

    for c in candidates:
        if c.is_file():
            return c

    mesh_files = [
        p for p in obj_dir.rglob("*") if p.suffix.lower() in MESH_EXTENSIONS
    ]
    if mesh_files:
        mesh_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return mesh_files[0]

    raise FileNotFoundError(
        f"No mesh file found in {obj_dir}. "
        f"Verify that {source} reconstruction completed successfully."
    )


async def convert_obj_to_usd(
    obj_path: Path,
    usd_path: Path,
) -> Path:
    """Convert an OBJ file to USD using Isaac Sim's asset converter."""
    import omni.kit.asset_converter as converter

    context = converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.single_mesh = True
    context.smooth_normals = True
    context.export_preview_surface = True

    usd_path.parent.mkdir(parents=True, exist_ok=True)

    instance = converter.get_instance()
    task = instance.create_converter_task(
        str(obj_path),
        str(usd_path),
        progress_callback=None,
        asset_converter_context=context,
    )
    success = await task.wait_until_finished()
    if not success:
        raise RuntimeError(
            f"Asset conversion failed for {obj_path}. "
            f"Detailed status: {task.get_status()}, {task.get_detailed_error()}"
        )

    print(f"Converted: {obj_path} -> {usd_path}")
    return usd_path


def add_asset_to_stage(
    usd_path: Path,
    prim_path: str,
    position: tuple[float, float, float] | None = None,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Add a USD asset as a reference in the current stage."""
    from pxr import Gf, Sdf, UsdGeom

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    prim = stage.OverridePrim(Sdf.Path(prim_path))
    prim.GetReferences().AddReference(str(usd_path))

    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()

    if position is not None:
        xformable.AddTranslateOp().Set(Gf.Vec3d(*position))

    xformable.AddScaleOp().Set(Gf.Vec3f(*scale))
    print(f"Added {usd_path.name} to stage at {prim_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Import reconstructed mesh into Isaac Sim as USD")
    p.add_argument("--name", required=True, help="Object name (subdirectory under reconstruction output)")
    p.add_argument(
        "--source", choices=list(RECONSTRUCTION_SOURCES), default="meshroom",
        help="Reconstruction backend (default: meshroom)",
    )
    p.add_argument(
        "--reconstruction-output", type=Path, default=None,
        help="Override path to reconstruction output base directory",
    )
    p.add_argument(
        "--assets-dir", type=Path, default=DEFAULT_ASSETS_DIR,
        help="Directory to save converted USD assets",
    )
    p.add_argument(
        "--position", type=float, nargs=3, default=None,
        metavar=("X", "Y", "Z"),
        help="If provided, add the asset to the current stage at this position",
    )
    p.add_argument("--scale", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"))
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return p


def main() -> None:
    args = build_parser().parse_args()

    output_base = args.reconstruction_output or RECONSTRUCTION_SOURCES[args.source]
    obj_path = find_mesh_file(output_base, args.name, args.source)
    print(f"Found mesh: {obj_path}")

    usd_path = args.assets_dir / f"{args.name}.usd"

    app = SimulationApp({"headless": args.headless})

    asyncio.get_event_loop().run_until_complete(
        convert_obj_to_usd(obj_path, usd_path)
    )

    if args.position is not None:
        add_asset_to_stage(
            usd_path,
            prim_path=f"/World/{args.name}",
            position=tuple(args.position),
            scale=tuple(args.scale),
        )

    print(f"USD asset saved: {usd_path}")
    app.close()


if __name__ == "__main__":
    main()
