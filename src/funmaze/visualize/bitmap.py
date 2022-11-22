from collections.abc import Mapping, Set

import numpy as np
import numpy.typing as npt

from funmaze.graph import Graph, Node, graph_nodes


def _connect(pos_set1: Set[tuple[int, int]],
             pos_set2: Set[tuple[int, int]]) -> tuple[int, int] | None:
    for pos1 in sorted(pos_set1):
        for pos2 in sorted(pos_set2):
            delta_x = abs(pos1[0] - pos2[0])
            delta_y = abs(pos1[1] - pos2[1])
            if {delta_x, delta_y} == {0, 2}:
                return (pos1[0] + pos2[0]) // 2, (pos1[1] + pos2[1]) // 2
    return None


def graph_bitmap(
        graph: Graph[Node],
        positions: Mapping[Node, Set[tuple[int, int]]],
        node_colors: Mapping[Node, int],
        wall_color: int,
        edge_color: int,
) -> npt.NDArray[np.int_]:
    if len({wall_color, edge_color} | set(node_colors.values())
           ) != 2 + len(node_colors):
        raise ValueError("colors of nodes, walls, and edges not distinct")
    size_x = 2 + max(
        pos[0] for pos_set in positions.values() for pos in pos_set)
    size_y = 2 + max(
        pos[1] for pos_set in positions.values() for pos in pos_set)
    # set walls everywhere
    bitmap = np.full((size_x, size_y), wall_color)
    # insert nodes
    for node in graph_nodes(graph):
        for pos in positions[node]:
            bitmap[pos] = node_colors[node]
    # insert edges
    for edge in graph:
        pos_set1, pos_set2 = tuple(positions[node] for node in edge)
        val1, val2 = tuple(node_colors[node] for node in edge)
        edge_pos: tuple[int, int] | None = _connect(pos_set1, pos_set2)
        if edge_pos is None:
            raise ValueError(
                f"cannot create edge between {val1} and {val2}\n{bitmap}")
        if bitmap[edge_pos] != wall_color:
            raise ValueError(
                f"cannot create edge at {edge_pos} between {val1} and {val2}\n"
                f"{bitmap}")
        bitmap[edge_pos] = edge_color
    return bitmap
