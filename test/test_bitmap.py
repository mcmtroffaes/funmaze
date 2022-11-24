import numpy as np
import pytest

from funmaze.graph import Graph
from funmaze.graph.grid import grid_sequential, neighbourhood_graph
from funmaze.render.bitmap import render_bitmap, bitmap_remove_dots, \
    bitmap_scale


def test_bitmap_1() -> None:
    grid = grid_sequential((2, 2))
    graph = neighbourhood_graph(grid)
    bitmap = render_bitmap(grid, graph)
    np.testing.assert_array_equal(bitmap, np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]))
    np.testing.assert_array_equal(bitmap_remove_dots(bitmap), np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ]))


def test_bitmap_2() -> None:
    grid = grid_sequential((3, 3))
    graph: Graph[np.uint] = {(np.uint(0), np.uint(2))}  # not neighbours
    with pytest.raises(ValueError, match="not neighbours"):
        render_bitmap(grid, graph)


def test_bitmap_3() -> None:
    grid = np.array([[0, 1, 1]])
    graph = neighbourhood_graph(grid)
    bitmap = render_bitmap(grid, graph)
    np.testing.assert_array_equal(bitmap, np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ]))


def test_bitmap_4() -> None:
    grid = np.array([[0, 1, 2, 2]])
    graph = {(0, 1)}
    bitmap = render_bitmap(grid, graph)
    np.testing.assert_array_equal(bitmap, np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]))


def test_bitmap_scale_1() -> None:
    bitmap = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 9, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ])
    bitmap2 = bitmap_scale(bitmap, 2, 3)
    np.testing.assert_array_equal(bitmap2, np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 9, 9, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 9, 9, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]))


def test_bitmap_scale_2() -> None:
    bitmap = np.array([
        [1, 2],
        [3, 0],
    ])
    bitmap2 = bitmap_scale(bitmap, 4, 2)
    np.testing.assert_array_equal(bitmap2, np.array([
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 2, 2],
        [1, 1, 1, 1, 2, 2],
        [3, 3, 3, 3, 0, 0],
        [3, 3, 3, 3, 0, 0],
    ]))
