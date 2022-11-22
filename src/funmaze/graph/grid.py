from collections.abc import Set
from typing import Iterable

from funmaze.graph import Edge, Graph

GridNode = tuple[int, int]


def grid_graph(size_x: int, size_y: int) -> Graph[GridNode]:
    """Build graph on a grid."""
    def _edges() -> Iterable[Edge[GridNode]]:
        for x in range(size_x):
            for y in range(size_y):
                if x < size_x - 1:
                    yield Edge([(x, y), (x + 1, y)])
                if y < size_y - 1:
                    yield Edge([(x, y), (x, y + 1)])
    return frozenset(_edges())


def grid_rectangle(xs: Iterable[int], ys: Iterable[int]
                   ) -> Set[GridNode]:
    """Build rectangular set of nodes."""
    return frozenset((x, y) for x in xs for y in ys)
