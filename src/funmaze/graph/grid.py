from collections.abc import Collection, Iterable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from funmaze.graph import IGraph

GridNode = TypeVar("GridNode", bound=np.generic)


def grid_squares(shape: tuple[int, ...]) -> npt.NDArray[np.int_]:
    """Create grid containing equally sized square nodes."""
    seq = np.arange(0, np.product(shape), dtype=np.int_)
    return seq.reshape(shape)


def grid_triangles(num_ver: int, num_hor: int
                   ) -> tuple[npt.NDArray[np.int_], np.int_]:
    """Return a two-dimensional grid matching a triangular topology.
    The node for the mask (which must be ignored when building the
    neighbourhood graph) is returned as well.

    A triangular topology is one where every node is connected to
    three other nodes (except on the edge of the graph)::

      :   AAA BBB CCC
      : DDD EEE FFF GGG
      : HHH III JJJ KKK
      :   LLL MMM NNN
    """
    grid = np.full((num_ver, num_hor * 4 - 1), 0, dtype=np.int_)
    value = 1
    for i in range(num_ver):
        if i % 3 == 0:
            for j in range(num_hor - 1):
                grid[i, slice(2 + j * 4, 5 + j * 4)] = value
                value += 1
        else:
            for j in range(num_hor):
                grid[i, slice(j * 4, 3 + j * 4)] = value
                value += 1
    return grid, np.int_(0)


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
        grid: npt.NDArray[GridNode], steps: Collection[int] = (-1, 1),
        mask: GridNode | None = None
) -> IGraph[GridNode]:
    """Build neighbourhood graph of *grid*."""
    for pos1, node1 in np.ndenumerate(grid):
        for pos2 in neighbourhood_positions(grid.shape, pos1, steps):
            node2 = grid[pos2]
            is_unmasked = (mask is None) or (node1 != mask and node2 != mask)
            if is_unmasked and node1 != node2:
                yield node1, node2
