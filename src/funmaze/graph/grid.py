from collections.abc import Collection
from typing import TypeVar, Iterable

import numpy as np
import numpy.typing as npt

from funmaze.graph import Graph, neighbourhood_positions

GridNode = TypeVar("GridNode", bound=np.generic)


def grid_sequential(shape: tuple[int, ...]) -> npt.NDArray[np.uint]:
    seq = np.arange(0, np.product(shape), dtype=np.uint)
    return seq.reshape(shape)


def grid_replace_nodes(
        positions: Iterable[tuple[int, ...]],
        node: GridNode,
        grid: npt.NDArray[GridNode],
) -> npt.NDArray[GridNode]:
    grid2 = grid.copy()
    for pos in positions:
        grid2[pos] = node
    return grid2


def neighbourhood_graph(
        grid: npt.NDArray[GridNode], steps: Collection[int] = (-1, 1)
) -> Graph[GridNode]:
    """Build neighbourhood graph of *grid*."""
    def _edges():
        for pos1, node1 in np.ndenumerate(grid):
            for pos2 in neighbourhood_positions(grid.shape, pos1, steps):
                node2 = grid[pos2]
                if node1 != node2:
                    yield node1, node2
    return frozenset(_edges())
