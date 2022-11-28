from collections.abc import Set

import numpy as np
import pytest

from funmaze.graph.grid import grid_squares, neighbourhood_graph, \
    grid_triangles, grid_hexagons


@pytest.mark.parametrize("shape,graph", [
    # 0 1
    ((1, 2), {(0, 1)}),
    # 0 1
    # 2 3
    # 4 5
    ((3, 2), {(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)}),
])
def test_grid_squares(
        shape: tuple[int, ...], graph: Set[tuple[int, int]]) -> None:
    grid = grid_squares(shape)
    assert frozenset(neighbourhood_graph(grid, steps=(1,))) == graph


@pytest.mark.parametrize("ver,hor,arr,graph", [
    (2, 2,
     [[0, 0, 1, 1, 1, 0, 0],
      [2, 2, 2, 0, 3, 3, 3]],
     {(1, 2), (1, 3)}),
    (4, 3,
     [[0, 0, 1, 1, 1, 0, 2, 2, 2, 0, 0],
      [3, 3, 3, 0, 4, 4, 4, 0, 5, 5, 5],
      [6, 6, 6, 0, 7, 7, 7, 0, 8, 8, 8],
      [0, 0, 9, 9, 9, 0, 10, 10, 10, 0, 0]],
     {(1, 3), (1, 4), (2, 4), (2, 5),
      (3, 6), (4, 7), (5, 8),
      (6, 9), (7, 9), (7, 10), (8, 10)}),
])
def test_grid_triangles(
        ver: int, hor: int, arr: list[list[int]], graph: Set[tuple[int, int]]
) -> None:
    grid, mask = grid_triangles(ver, hor)
    np.testing.assert_array_equal(grid, arr)
    assert frozenset(neighbourhood_graph(grid, mask=mask, steps=(1,))) == graph


@pytest.mark.parametrize("ver,hor,graph", [
    # TODO test these grids
    # 1122
    #  3344
    (2, 2,
     {(1, 3), (1, 2), (2, 3), (2, 4), (3, 4)}),
    # 112233
    #  445566
    # 778899
    (3, 3,
     {(1, 4), (1, 2), (2, 4), (2, 5), (2, 3), (3, 5), (3, 6),
      (4, 7), (4, 8), (4, 5), (5, 8), (5, 9), (5, 6), (6, 9),
      (7, 8), (8, 9)}),
])
def test_grid_hexagons(
        ver: int, hor: int, graph: Set[tuple[int, int]]) -> None:
    grid, mask = grid_hexagons(ver, hor)
    assert frozenset(neighbourhood_graph(grid, mask=mask, steps=(1,))) == graph


def test_grid_hexagons_2() -> None:
    grid, mask = grid_hexagons(3, 3, mask=False)
    assert mask is None
    np.testing.assert_array_equal(grid, [
        [1, 1, 2, 2, 3, 3, 3],
        [4, 4, 4, 5, 5, 6, 6],
        [7, 7, 8, 8, 9, 9, 9],
    ])
