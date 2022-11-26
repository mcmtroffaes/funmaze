import itertools

import numpy as np

from funmaze.generate.backtracking import generate_backtracking_maze
from funmaze.generate.wilson import generate_wilson_maze
from funmaze.graph.grid import grid_squares, neighbourhood_graph, \
    grid_replace_nodes, graph_grid
from funmaze.render.bitmap import render_bitmap


def test_generate_simple():
    grid = grid_squares((5, 5))
    graph = frozenset(neighbourhood_graph(grid))
    maze1 = frozenset(generate_backtracking_maze(graph, grid[0, 0]))
    maze2 = frozenset(generate_wilson_maze(graph))
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


def test_generate_grid_with_room():
    grid = grid_replace_nodes(
        itertools.product(range(3, 6), range(3, 6)),
        np.uint(99),
        grid_squares((9, 9)))
    graph = frozenset(neighbourhood_graph(grid))
    maze1 = frozenset(generate_backtracking_maze(graph, grid[0, 0]))
    maze2 = frozenset(generate_wilson_maze(graph))
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


def test_generate_empty():
    assert set(generate_backtracking_maze(set(), 0)) == set()
    assert set(generate_wilson_maze(set())) == set()


def test_generate_3d():
    graph = frozenset(graph_grid((3, 3, 3)))
    set(generate_backtracking_maze(graph, (0, 0)))
    set(generate_wilson_maze(graph))


def test_generate_large():
    """Larger test to help profiling."""
    graph = frozenset(graph_grid((100, 100)))
    set(generate_backtracking_maze(graph, (0, 0)))
    set(generate_wilson_maze(graph))
