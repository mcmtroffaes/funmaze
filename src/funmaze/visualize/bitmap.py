import itertools
from collections.abc import Mapping, Set
from typing import Iterable

import numpy as np
import numpy.typing as npt

from funmaze.graph import Graph, graph_nodes, Edge
from funmaze.graph.grid import GridNode, neighbourhood_positions


def bitmap_render(grid: npt.NDArray[GridNode], graph: Graph[GridNode]
                  ) -> npt.NDArray[np.bool_]:
    """Render the *graph* of a *grid* as a bitmap.

    Nodes and edges are False and walls are True.
    This results in a bitmap representation of the graph's topology.
    The *graph* must be a subgraph of the neighbourhood graph
    of *grid*, that is,
    every edge in *graph* must correspond to neighbouring nodes in *grid*.
    """
    def _bitmap_pos(grid_pos: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(1 + 2 * i for i in grid_pos)
    # start with walls everywhere
    bitmap = np.full(_bitmap_pos(grid.shape), True)
    nodes = graph_nodes(graph)
    # remove walls at nodes in the graph
    for pos, node in np.ndenumerate(grid):
        if node in nodes:
            bitmap[_bitmap_pos(pos)] = False
    # insert edges
    missing_edges: set[Edge[GridNode]] = set(graph)
    for pos1, node1 in np.ndenumerate(grid):
        for pos2 in neighbourhood_positions(grid.shape, pos1, steps=(1,)):
            node1 = grid[pos1]
            node2 = grid[pos2]
            b_pos1 = _bitmap_pos(pos1)
            b_pos2 = _bitmap_pos(pos2)
            e_pos = tuple((i + j) // 2 for i, j in zip(b_pos1, b_pos2))
            if node1 == node2:
                if node1 in nodes:
                    bitmap[e_pos] = False
            else:
                edge = Edge({node1, node2})
                if edge in graph:
                    bitmap[e_pos] = False
                    missing_edges.discard(edge)
    if missing_edges:
        raise ValueError(
            f"cannot add edges {missing_edges} as nodes are not neighbours")
    return bitmap


_square_deltas = frozenset(
    itertools.product([-1, 0, 1], [-1, 0, 1])) - {(0, 0)}


def _square_around_position(axis1: int, axis2: int, pos: tuple[int, ...]
                            ) -> Iterable[tuple[int, ...]]:
    assert 0 <= axis1 < len(pos)
    assert 0 <= axis2 < len(pos)
    for delta in _square_deltas:
        yield tuple(
            x
            + (delta[0] if k == axis1 else 0)
            + (delta[1] if k == axis2 else 0)
            for k, x in enumerate(pos))


def bitmap_remove_dots(bitmap: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Remove individual walls that are completely surrounded by a square of
    non-walls (aka "dots").
    """
    bitmap2 = bitmap.copy()
    for pos, wall in np.ndenumerate(bitmap):
        for axis1, axis2 in itertools.combinations(
                range(len(bitmap.shape)), 2):
            if all(1 <= pos[k] < bitmap.shape[k] - 1 for k in {axis1, axis2}):
                surrounding_walls = (
                    bitmap[pos2]
                    for pos2 in _square_around_position(axis1, axis2, pos))
                if not any(surrounding_walls):
                    bitmap2[pos] = False
    return bitmap2
