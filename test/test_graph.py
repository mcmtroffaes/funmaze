import numpy as np

from funmaze.graph import Graph, graph_merge_nodes, graph_remove_nodes
from funmaze.graph.grid import grid_squares, neighbourhood_graph


def test_graph_remove_nodes() -> None:
    grid = grid_squares((4, 4))
    graph = neighbourhood_graph(grid, steps=(1,))
    rec = {np.uint(i) for i in [5, 6, 7, 9, 10, 11, 13, 14, 15]}
    graph2 = set(graph_remove_nodes(graph, rec))
    # 0  1  2  3
    # 4  ** ** **
    # 8  ** ** **
    # 12 ** ** **
    assert graph2 == {
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 8), (8, 12),
    }


def test_graph_merge_nodes() -> None:
    grid = grid_squares((4, 4))
    graph = neighbourhood_graph(grid, steps=(1,))
    rec = {np.uint(i) for i in [5, 6, 7, 9, 10, 11, 13, 14, 15]}
    graph2 = set(graph_merge_nodes(graph, rec, np.uint(10)))
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
    assert set(graph_merge_nodes(graph, {0, 1}, 0)) == {(0, 0), (0, 2)}
    assert set(graph_merge_nodes(graph, {0, 1}, 1)) == {(1, 1), (1, 2)}
    assert set(graph_merge_nodes(graph, {1, 2}, 1)) == {(0, 1), (1, 1)}
    assert set(graph_merge_nodes(graph, {1, 2}, 2)) == {(0, 2), (2, 2)}
    assert set(graph_merge_nodes(graph, {0, 2}, 0)) == {(0, 0), (0, 1), (1, 0)}
    assert set(graph_merge_nodes(graph, {0, 2}, 2)) == {(1, 2), (2, 1), (2, 2)}
