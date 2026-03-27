"""Path planning algorithms for robot navigation."""

import numpy as np
import networkx as nx


class GridPlanner:
    """A* path planner on a 2D occupancy grid."""

    def __init__(self, grid: np.ndarray):
        """
        Args:
            grid: 2D binary array where 0 = free, 1 = obstacle.
        """
        self.grid = grid
        self.graph = self._build_graph()

    def _build_graph(self) -> nx.Graph:
        rows, cols = self.grid.shape
        g = nx.grid_2d_graph(rows, cols)
        obstacles = list(zip(*np.where(self.grid == 1)))
        g.remove_nodes_from(obstacles)
        return g

    def plan(self, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
        """Find shortest path from start to goal.

        Raises nx.NetworkXNoPath if no path exists.
        """
        return nx.astar_path(self.graph, start, goal, heuristic=self._heuristic)

    @staticmethod
    def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
