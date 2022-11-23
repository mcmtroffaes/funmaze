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


def test_grid_graph_mask() -> None:
    grid = grid_sequential((4, 4))
    graph = neighbourhood_graph(grid)
    rec = {5, 6, 7, 9, 10, 11, 13, 14, 15}
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
    rec = {5, 6, 7, 9, 10, 11, 13, 14, 15}
    graph2 = graph_merge_nodes(graph, rec, 10)
    # 0  1  2  3
    # 4  ** ** **
    # 8  ** 10 **
    # 12 ** ** **
    assert graph2 == {
        Edge([0, 1]),
        Edge([1, 2]),
        Edge([2, 3]),
        Edge([0, 4]),
        Edge([4, 8]),
        Edge([8, 12]),
        Edge([1, 10]),
        Edge([2, 10]),
        Edge([3, 10]),
        Edge([4, 10]),
        Edge([8, 10]),
        Edge([12, 10]),
    }


def test_graphviz() -> None:
    grid = grid_replace_nodes(
        itertools.product(range(3, 5), range(3, 5)),
        np.uint(99),
        grid_replace_nodes(
            itertools.product(range(1, 4), range(1, 4)),
            np.uint(8),
            grid_sequential((7, 7))))
    graph = graph_remove_nodes(neighbourhood_graph(grid), {np.uint(99)})
    nodes = graph_nodes(graph)
    names = {node: str(node) for node in nodes}
    positions = {
        node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    gv = render_graphviz(graph, names, positions)
    assert '0 [pos="0,0!"]' in gv.source
    assert '0 -- 1' in gv.source
    # for debugging:
    # gv.render("gv", view=True)
    gv2 = render_graphviz(graph, names)
    assert '0 [pos="0,0!"]' not in gv2.source
    assert '0 -- 1' in gv2.source
    # for debugging:
    # gv2.render("gv2", view=True)


def test_graphviz_2(tmp_path) -> None:
    grid = grid_sequential((2, 2))
    graph = neighbourhood_graph(grid)
    nodes = graph_nodes(graph)
    names = {node: "oops" for node in nodes}
    with pytest.raises(ValueError):  # names not unique
        render_graphviz(graph, names)


def test_bitmap_1() -> None:
    grid = grid_sequential((2, 2))
    graph = neighbourhood_graph(grid)
    bitmap = render_bitmap(grid, graph)
    assert (bitmap == np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]).astype(bool)).all()
    assert (bitmap_remove_dots(bitmap) == np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]).astype(bool)).all()


def test_bitmap_2() -> None:
    grid = grid_sequential((3, 3))
    graph: Graph[np.uint] = {Edge([0, 2])}  # not neighbours
    with pytest.raises(ValueError, match="not neighbours"):
        render_bitmap(grid, graph)


def test_bitmap_3() -> None:
    grid = np.array([[0, 1, 1]])
    graph = neighbourhood_graph(grid)
    bitmap = render_bitmap(grid, graph)
    assert (bitmap == np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]).astype(bool)).all()


def test_bitmap_4() -> None:
    grid = np.array([[0, 1, 2, 2]])
    graph = {Edge([0, 1])}
    bitmap = render_bitmap(grid, graph)
    assert (bitmap == np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]).astype(bool)).all()
