from collections.abc import Set

import pytest

from funmaze.graph.grid import grid_squares, neighbourhood_graph, grid_triangles


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
    grid = grid_squares(shape)
    assert frozenset(neighbourhood_graph(grid, steps=(1,))) == graph


@pytest.mark.parametrize("ver,hor,graph", [
    #   111
    # 222 333
    (2, 2, {(1, 2), (1, 3)}),
    #   111 222
    # 333 444 555
    # 666 777 888
    #   999 000
    (4, 3, {(1, 3), (1, 4), (2, 4), (2, 5),
            (3, 6), (4, 7), (5, 8),
            (6, 9), (7, 9), (7, 10), (8, 10)}),
])
def test_grid_triangles_graph(
        ver: int, hor: int, graph: Set[tuple[int, int]]) -> None:
    grid, mask = grid_triangles(ver, hor)
    assert frozenset(neighbourhood_graph(grid, mask=mask, steps=(1,))) == graph
