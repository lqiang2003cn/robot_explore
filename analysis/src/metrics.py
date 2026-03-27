"""Metrics and visualization for robot exploration runs."""

import pandas as pd
import numpy as np


def load_run_log(csv_path: str) -> pd.DataFrame:
    """Load a simulation run log CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    required = {"step", "x", "y", "theta"}
    if not required.issubset(df.columns):
        raise ValueError(f"Run log missing columns: {required - set(df.columns)}")
    return df


def path_efficiency(df: pd.DataFrame) -> float:
    """Ratio of straight-line distance to actual path length.

    Returns 1.0 for a perfectly straight path, <1.0 otherwise.
    """
    dx = np.diff(df["x"].values)
    dy = np.diff(df["y"].values)
    actual = np.sum(np.sqrt(dx**2 + dy**2))
    if actual == 0:
        return 0.0
    straight = np.sqrt(
        (df["x"].iloc[-1] - df["x"].iloc[0]) ** 2
        + (df["y"].iloc[-1] - df["y"].iloc[0]) ** 2
    )
    return straight / actual


def coverage_percentage(df: pd.DataFrame, grid_res: float = 0.5, world_size: tuple = (20, 20)) -> float:
    """Fraction of the world grid cells visited by the robot."""
    gx = (df["x"].values / grid_res).astype(int)
    gy = (df["y"].values / grid_res).astype(int)
    visited = set(zip(gx, gy))
    total_cells = int(world_size[0] / grid_res) * int(world_size[1] / grid_res)
    return len(visited) / total_cells
