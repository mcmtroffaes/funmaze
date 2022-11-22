import pytest

from funmaze.graph import graph_remove_nodes, Edge, \
    graph_merge_nodes, graph_nodes
from funmaze.graph.grid import grid_graph, grid_rectangle
from funmaze.visualize.graphviz import graph_graphviz


def test_edge():
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


def test_grid_graph():
    # 00 01
    assert grid_graph(1, 2) == {Edge([(0, 0), (0, 1)])}
    # 00 01
    # 10 11
    # 20 21
    assert grid_graph(3, 2) == {
        # horizontal
        Edge([(0, 0), (0, 1)]),
        Edge([(1, 0), (1, 1)]),
        Edge([(2, 0), (2, 1)]),
        # vertical
        Edge([(0, 0), (1, 0)]), Edge([(1, 0), (2, 0)]),
        Edge([(0, 1), (1, 1)]), Edge([(1, 1), (2, 1)]),
    }


def test_grid_mask():
    rec = grid_rectangle(range(2, 5), range(1, 3))
    assert rec == {(2, 1), (3, 1), (4, 1), (2, 2), (3, 2), (4, 2)}


def test_grid_graph_mask():
    grid = grid_graph(4, 4)
    rec = grid_rectangle(range(1, 4), range(1, 4))
    grid2 = graph_remove_nodes(grid, rec)
    # 00 01 02 03
    # 10 ** ** **
    # 20 ** ** **
    # 30 ** ** **
    assert grid2 == {
        Edge([(0, 0), (0, 1)]),
        Edge([(0, 1), (0, 2)]),
        Edge([(0, 2), (0, 3)]),
        Edge([(0, 0), (1, 0)]),
        Edge([(1, 0), (2, 0)]),
        Edge([(2, 0), (3, 0)]),
    }


def test_graph_merge_nodes():
    grid = grid_graph(4, 4)
    rec = grid_rectangle(range(1, 4), range(1, 4))
    grid2 = graph_merge_nodes(grid, rec, (2, 2))
    # 00 01 02 03
    # 10 ** ** **
    # 20 ** 22 **
    # 30 ** ** **
    assert grid2 == {
        Edge([(0, 0), (0, 1)]),
        Edge([(0, 1), (0, 2)]),
        Edge([(0, 2), (0, 3)]),
        Edge([(0, 0), (1, 0)]),
        Edge([(1, 0), (2, 0)]),
        Edge([(2, 0), (3, 0)]),
        Edge([(0, 1), (2, 2)]),
        Edge([(0, 2), (2, 2)]),
        Edge([(0, 3), (2, 2)]),
        Edge([(1, 0), (2, 2)]),
        Edge([(2, 0), (2, 2)]),
        Edge([(3, 0), (2, 2)]),
    }


def test_graphviz(tmp_path):
    rec1 = grid_rectangle(range(1, 4), range(1, 4))
    rec2 = grid_rectangle(range(3, 5), range(3, 5))
    top1 = grid_graph(7, 7)
    top2 = graph_merge_nodes(top1, rec1, (2, 2))
    top = graph_remove_nodes(top2, rec2)
    nodes = graph_nodes(top)
    names = {node: f"{i}" for i, node in enumerate(sorted(nodes))}
    positions = {node: (node[1], -node[0]) for node in nodes}
    graph = graph_graphviz(top, names, positions)
    graph.render(tmp_path / "grid", view=True)
    assert '0 [pos="0,0!"]' in graph.source
    assert '0 -- 1' in graph.source
    graph2 = graph_graphviz(top, names)
    assert '0 [pos="0,0!"]' not in graph2.source
    assert '0 -- 1' in graph2.source


def test_graphviz_2(tmp_path):
    top = grid_graph(2, 2)
    nodes = graph_nodes(top)
    names = {node: "oops" for node in nodes}
    with pytest.raises(ValueError):  # names not unique
        graph_graphviz(top, names)
