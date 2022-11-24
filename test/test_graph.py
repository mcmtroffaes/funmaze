import itertools
from typing import Iterable

import numpy as np
import pytest

from funmaze.graph import graph_remove_nodes, Edge, \
    graph_merge_nodes, graph_nodes, Graph
from funmaze.graph.grid import grid_sequential, neighbourhood_graph, \
    grid_replace_nodes
from funmaze.render.bitmap import render_bitmap, bitmap_remove_dots
from funmaze.render.graphviz import render_graphviz


def test_edge() -> None:
    with pytest.raises(ValueError):
        Edge([1, 1])
    e1 = Edge([1, 2])
    e2 = Edge([2, 1])
    assert e1 == e2
    e3 = Edge([2, 3])
    assert e1 != e3
    top = {e1, e2, e3}
    assert top == {e2, e3}
    assert top == {e1, e3}
    assert set(e1) == {1, 2}
    assert set(e2) == {1, 2}
    assert set(e3) == {2, 3}
    assert 1 in e1
    assert 2 in e1
    assert 3 not in e1
    assert "1" not in e1
    assert all(len(e) == 2 for e in [e1, e2, e3])


@pytest.mark.parametrize("shape,edges", [
    # 0 1
    ((1, 2), [(0, 1)]),
    # 0 1
    # 2 3
    # 4 5
    ((3, 2), [(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)]),
])
def test_grid_neighbourhood_graph(
        shape: tuple[int, ...], edges: Iterable[tuple[int, int]]) -> None:
    grid = grid_sequential(shape)
    graph = {Edge(edge) for edge in edges}
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
        Edge([0, 1]), Edge([1, 2]), Edge([2, 3]),
        Edge([0, 4]), Edge([4, 8]), Edge([8, 12]),
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
        Edge([0, 1]), Edge([1, 2]), Edge([2, 3]),
        Edge([0, 4]), Edge([4, 8]), Edge([8, 12]),
        Edge([1, 10]), Edge([2, 10]), Edge([3, 10]),
        Edge([4, 10]),  Edge([8, 10]), Edge([12, 10]),
    }


def test_graph_merge_nodes_2() -> None:
    grid = grid_sequential((2, 2))
    graph = neighbourhood_graph(grid)
    rec = {np.uint(i) for i in [1, 2]}
    with pytest.raises(ValueError, match="nodes must contain target"):
        graph_merge_nodes(graph, rec, np.uint(3))


def test_graph_merge_nodes_3() -> None:
    edge1 = Edge([0, 1])
    edge2 = Edge([1, 2])
    edge3 = Edge([0, 2])
    graph: Graph[int] = {edge1, edge2, edge3}
    assert graph_merge_nodes(graph, {0, 1}, 0) == {edge3}
    assert graph_merge_nodes(graph, {0, 1}, 1) == {edge2}
    assert graph_merge_nodes(graph, {1, 2}, 1) == {edge1}
    assert graph_merge_nodes(graph, {1, 2}, 2) == {edge3}
    assert graph_merge_nodes(graph, {0, 2}, 0) == {edge1}
    assert graph_merge_nodes(graph, {0, 2}, 2) == {edge2}
