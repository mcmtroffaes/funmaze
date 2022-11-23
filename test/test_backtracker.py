import itertools

import numpy as np

from funmaze.generate.backtracker import generate_backtracker
from funmaze.graph.grid import grid_sequential, neighbourhood_graph, \
    grid_replace_nodes
from funmaze.render.bitmap import render_bitmap


def test_recursive_backtracker_simple():
    grid = grid_sequential((5, 5))
    graph = neighbourhood_graph(grid)
    maze = generate_backtracker(graph)
    bitmap = render_bitmap(grid, maze).astype(int)
    assert sum(bitmap.flat) == 72  # perfect 5x5 maze has fixed number of walls
    # for debugging:
    # print(bitmap)
    # names = {node: str(node) for _, node in np.ndenumerate(grid)}
    # positions = {
    #     node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    # from funmaze.render.graphviz import render_graphviz
    # render_graphviz(maze, names, positions).render(view=True)


def test_recursive_backtracker_grid_with_room():
    grid = grid_replace_nodes(
        itertools.product(range(3, 6), range(3, 6)),
        np.uint(99),
        grid_sequential((9, 9)))
    graph = neighbourhood_graph(grid)
    maze = generate_backtracker(graph)
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


def test_recursive_backtracker_empty():
    assert generate_backtracker(set()) == set()


def test_recursive_backtracker_large():
    """Larger test to help profiling."""
    grid = grid_sequential((40, 40))
    graph = neighbourhood_graph(grid)
    generate_backtracker(graph)
