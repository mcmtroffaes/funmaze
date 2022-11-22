from collections.abc import Mapping, Set

import numpy as np
import numpy.typing as npt

from funmaze.graph import Graph, Node, graph_nodes


def _connect(pos_set1: Set[tuple[int, int]],
             pos_set2: Set[tuple[int, int]]) -> tuple[int, int]:
    for pos1 in pos_set1:
        for pos2 in pos_set2:
            delta_x = abs(pos1[0] - pos2[0])
            delta_y = abs(pos1[1] - pos2[1])
            if {delta_x, delta_y} == {0, 2}:
                return (pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2
    print(f"warning: cannot connect {pos_set1} and {pos_set2}")


def graph_bitmap(
        graph: Graph[Node],
        positions: Mapping[Node, Set[tuple[int, int]]],
) -> npt.NDArray[int]:
    size_x = 2 + max(pos[0] for pos_set in positions.values() for pos in pos_set)
    size_y = 2 + max(pos[1] for pos_set in positions.values() for pos in pos_set)
    arr = np.full((size_x, size_y), -1)
    for node in graph_nodes(graph):
        for pos in positions[node]:
            arr[pos[0], pos[1]] = 0
    for edge in graph:
        pos_set1, pos_set2 = tuple(positions[node] for node in edge)
        pos = _connect(pos_set1, pos_set2)
        arr[pos] = 0
    return arr
