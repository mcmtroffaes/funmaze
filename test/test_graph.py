from collections.abc import Set
from typing import Iterable

import numpy as np
import pytest

from funmaze.graph import graph_remove_nodes, Edge, graph_merge_nodes, Graph
from funmaze.graph.grid import grid_sequential, neighbourhood_graph


@pytest.mark.parametrize("shape,graph", [
    # 0 1
    ((1, 2), {(0, 1)}),
    # 0 1
    # 2 3
    # 4 5
    ((3, 2), {(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)}),
])
def test_grid_neighbourhood_graph(
        shape: tuple[int, ...], graph: Set[tuple[int, int]]) -> None:
    grid = grid_sequential(shape)
    assert neighbourhood_graph(grid) == graph


def test_graph_remove_nodes() -> None:
    grid = grid_sequential((4, 4))
    graph = neighbourhood_graph(grid)
    rec = {np.uint(i) for i in [5, 6, 7, 9, 10, 11, 13, 14, 15]}
    graph2 = graph_remove_nodes(graph, rec)
    # 0  1  2  3
    # 4  ** ** **
    # 8  ** ** **
    # 12 ** ** **
    assert graph2 == {
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 8), (8, 12),
    }


def test_graph_merge_nodes() -> None:
    grid = grid_sequential((4, 4))
    graph = neighbourhood_graph(grid)
    rec = {np.uint(i) for i in [5, 6, 7, 9, 10, 11, 13, 14, 15]}
    graph2 = graph_merge_nodes(graph, rec, np.uint(10))
    # 0  1  2  3
    # 4  ** ** **
    # 8  ** 10 **
    # 12 ** ** **
    assert graph2 == {
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 8), (8, 12),
        (1, 10), (2, 10), (3, 10),
        (4, 10),  (8, 10), (12, 10),
        (10, 10),
    }


def test_graph_merge_nodes_2() -> None:
    edge1 = (0, 1)
    edge2 = (1, 2)
    edge3 = (0, 2)
    graph: Graph[int] = {edge1, edge2, edge3}
    assert graph_merge_nodes(graph, {0, 1}, 0) == {(0, 0), (0, 2)}
    assert graph_merge_nodes(graph, {0, 1}, 1) == {(1, 1), (1, 2)}
    assert graph_merge_nodes(graph, {1, 2}, 1) == {(0, 1), (1, 1)}
    assert graph_merge_nodes(graph, {1, 2}, 2) == {(0, 2), (2, 2)}
    assert graph_merge_nodes(graph, {0, 2}, 0) == {(0, 0), (0, 1), (1, 0)}
    assert graph_merge_nodes(graph, {0, 2}, 2) == {(1, 2), (2, 1), (2, 2)}
