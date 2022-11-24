import numpy as np
import pytest

from funmaze.graph import Graph, Edge
from funmaze.graph.grid import grid_sequential, neighbourhood_graph
from funmaze.render.bitmap import render_bitmap, bitmap_remove_dots


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
    graph: Graph[np.uint] = {Edge([np.uint(0), np.uint(2)])}  # not neighbours
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
