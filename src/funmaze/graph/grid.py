from collections.abc import Collection
from typing import TypeVar, Iterable

import numpy as np
import numpy.typing as npt

from funmaze.graph import IGraph

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
    for i, (x, x_max) in enumerate(zip(pos, shape)):
        for step in steps:
            if 0 <= x + step < x_max:
                yield tuple((x2 + step) if i == j else x2
                            for j, x2 in enumerate(pos))


def graph_grid(shape: tuple[int, ...], steps: Collection[int] = (-1, 1)
               ) -> IGraph[tuple[int, ...]]:
    """Construct a grid shaped graph."""
    for pos1 in np.ndindex(*shape):
        for pos2 in neighbourhood_positions(shape, pos1, steps):
            yield pos1, pos2


def neighbourhood_graph(
        grid: npt.NDArray[GridNode], steps: Collection[int] = (-1, 1)
) -> IGraph[GridNode]:
    """Build neighbourhood graph of *grid*."""
    for pos1, node1 in np.ndenumerate(grid):
        for pos2 in neighbourhood_positions(grid.shape, pos1, steps):
            node2 = grid[pos2]
            if node1 != node2:
                yield node1, node2
