from collections.abc import Set
from typing import Iterable

from funmaze.graph import Edge, Graph

GridNode = tuple[int, int]


def grid_graph(size_x: int, size_y: int) -> Graph[GridNode]:
    def _edges() -> Iterable[Edge[GridNode]]:
        for x in range(size_x):
            for y in range(size_y):
                if x < size_x - 1:
                    yield Edge([(x, y), (x + 1, y)])
                if y < size_y - 1:
                    yield Edge([(x, y), (x, y + 1)])
    return set(_edges())


def grid_rectangle(top_left: GridNode, bottom_right: GridNode
                   ) -> Set[GridNode]:
    return set((x, y)
               for x in range(top_left[0], bottom_right[0] + 1)
               for y in range(top_left[1], bottom_right[1] + 1))
