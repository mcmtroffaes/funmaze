import numpy as np
import pytest

from funmaze.generate.dfs import generate_dfs_maze
from funmaze.generate.wilson import generate_wilson_maze
from funmaze.graph.grid import grid_squares, neighbourhood_graph, graph_grid
from funmaze.render.bitmap import render_bitmap


def test_generate_simple() -> None:
    grid = grid_squares((5, 5))
    graph = frozenset(neighbourhood_graph(grid))
    maze1 = frozenset(generate_dfs_maze(graph, grid[0, 0]))
    maze2 = frozenset(generate_wilson_maze(graph, grid[0, 0]))
    bitmap1 = render_bitmap(grid, maze1).astype(int)
    bitmap2 = render_bitmap(grid, maze2).astype(int)
    assert bitmap1.sum() == 72
    assert bitmap2.sum() == 72
    # for debugging:
    # print(bitmap)
    # names = {node: str(node) for _, node in np.ndenumerate(grid)}
    # positions = {
    #     node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    # from funmaze.render.graphviz import render_graphviz
    # render_graphviz(maze, names, positions).render(view=True)


def test_generate_grid_with_room() -> None:
    grid = grid_squares((9, 9))
    grid[slice(3, 6), slice(3, 6)] = np.uint(99)
    graph = frozenset(neighbourhood_graph(grid))
    maze1 = frozenset(generate_dfs_maze(graph, grid[0, 0]))
    maze2 = frozenset(generate_wilson_maze(graph, grid[0, 0]))
    bitmap1 = render_bitmap(grid, maze1).astype(int)
    bitmap2 = render_bitmap(grid, maze2).astype(int)
    assert bitmap1.sum() == 196
    assert bitmap2.sum() == 196
    # from funmaze.render.bitmap import bitmap_remove_dots
    # print(bitmap_remove_dots(bitmap1).astype(int))
    # print(bitmap_remove_dots(bitmap2).astype(int))
    # names = {node: str(node) for _, node in np.ndenumerate(grid)}
    # positions = {
    #    node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    # from funmaze.render.graphviz import render_graphviz
    # render_graphviz(maze1, names, positions).render(view=True)
    # render_graphviz(maze2, names, positions).render(view=True)


def test_generate_empty() -> None:
    with pytest.raises(KeyError):
        assert set(generate_dfs_maze(set(), 0))
    with pytest.raises(KeyError):
        assert set(generate_wilson_maze(set(), 0))


def test_wilson_bad_1() -> None:
    with pytest.raises(IndexError):
        set(generate_wilson_maze([(1, 0), (2, 3)], 0))


def test_wilson_bad_2() -> None:
    with pytest.raises(IndexError):
        set(generate_wilson_maze([(0, 1)], 0))


def test_generate_3d() -> None:
    graph = frozenset(graph_grid((3, 3, 3)))
    assert len(set(generate_dfs_maze(graph, (0, 0, 0)))) == 52
    assert len(set(generate_wilson_maze(graph, (0, 0, 0)))) == 52


def test_generate_large() -> None:
    """Larger test to help profiling."""
    graph = frozenset(graph_grid((100, 100)))
    assert len(set(generate_dfs_maze(graph, (0, 0)))) == 19998
    assert len(set(generate_wilson_maze(graph, (0, 0)))) == 19998
