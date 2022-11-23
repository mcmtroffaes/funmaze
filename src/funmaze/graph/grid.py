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


def neighbourhood_graph(
        grid: npt.NDArray[GridNode]) -> Graph[GridNode]:
    """Build neighbourhood graph based on a grid."""
    def _edges():
        indices = list(range(len(grid.shape)))
        for pos, node in np.ndenumerate(grid):
            for i in indices:
                for delta in (-1, 1):
                    pos2 = tuple(pos[j] + (delta if i == j else 0)
                                 for j in indices)
                    if all(0 <= x < m for x, m in zip(pos2, grid.shape)):
                        node2 = grid[pos2]
                        if node != node2:
                            yield Edge({node, node2})
    return frozenset(_edges())
