import itertools

import numpy as np

from funmaze.generate.backtracking import generate_backtracking_maze
from funmaze.graph.grid import grid_squares, neighbourhood_graph, \
    grid_replace_nodes, graph_grid
from funmaze.render.bitmap import render_bitmap


def test_backtracking_simple():
    grid = grid_squares((5, 5))
    graph = neighbourhood_graph(grid)
    maze = frozenset(generate_backtracking_maze(graph, grid[0, 0]))
    bitmap = render_bitmap(grid, maze).astype(int)
    assert sum(bitmap.flat) == 72  # perfect 5x5 maze has fixed number of walls
    # for debugging:
    # print(bitmap)
    # names = {node: str(node) for _, node in np.ndenumerate(grid)}
    # positions = {
    #     node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    # from funmaze.render.graphviz import render_graphviz
    # render_graphviz(maze, names, positions).render(view=True)


def test_backtracking_grid_with_room():
    grid = grid_replace_nodes(
        itertools.product(range(3, 6), range(3, 6)),
        np.uint(99),
        grid_squares((9, 9)))
    graph = neighbourhood_graph(grid)
    maze = frozenset(generate_backtracking_maze(graph, grid[0, 0]))
    render_bitmap(grid, maze)
    # for debugging:
    # bitmap = render_bitmap(grid, maze)
    # from funmaze.render.bitmap import bitmap_remove_dots
    # print(bitmap_remove_dots(bitmap).astype(int))
    # names = {node: str(node) for _, node in np.ndenumerate(grid)}
    # positions = {
    #    node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    # from funmaze.render.graphviz import render_graphviz
    # render_graphviz(maze, names, positions).render(view=True)
    # raise


def test_backtracking_empty():
    assert set(generate_backtracking_maze(set(), 0)) == set()


def test_backtracking_3d():
    graph = graph_grid((3, 3, 3))
    set(generate_backtracking_maze(graph, (0, 0)))


def test_backtracking_large():
    """Larger test to help profiling."""
    graph = graph_grid((100, 100))
    set(generate_backtracking_maze(graph, (0, 0)))
