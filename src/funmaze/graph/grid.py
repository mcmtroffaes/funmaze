from collections.abc import Collection
from typing import TypeVar, Iterable

import numpy as np
import numpy.typing as npt

from funmaze.graph import Edge, Graph


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


def neighbourhood_positions(
        shape: tuple[int, ...], pos: tuple[int, ...],
        steps: Collection[int] = (-1, 1),
) -> Iterable[tuple[int, ...]]:
    """List all positions neighbouring *pos* on a grid of the given *shape*,
    assuming we can move in *steps* along a single axis.
    """
    for i in range(len(shape)):
        for step in steps:
            pos2 = tuple(pos[j] + (step if i == j else 0)
                         for j in range(len(shape)))
            if all(0 <= x < m for x, m in zip(pos2, shape)):
                yield pos2


def neighbourhood_graph(grid: npt.NDArray[GridNode]) -> Graph[GridNode]:
    """Build neighbourhood graph of *grid*."""
    def _edges():
        for pos, node in np.ndenumerate(grid):
            for pos2 in neighbourhood_positions(grid.shape, pos):
                node2 = grid[pos2]
                if node != node2:
                    yield Edge({node, node2})
    return frozenset(_edges())
