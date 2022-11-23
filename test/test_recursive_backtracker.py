import sys

import pytest

from funmaze.generate.recursive_backtracker import \
    generate_recursive_backtracker
from funmaze.graph.grid import grid_sequential, neighbourhood_graph
from funmaze.render.bitmap import render_bitmap


@pytest.fixture
def recursion_limit_50():
    """Fixture to lower the recursion limit, to speed up tests."""
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(50)
    try:
        yield
    finally:
        sys.setrecursionlimit(old_limit)


def test_recursive_backtracker_simple():
    grid = grid_sequential((5, 5))
    graph = neighbourhood_graph(grid)
    maze = generate_recursive_backtracker(graph)
    bitmap = render_bitmap(grid, maze).astype(int)
    assert sum(bitmap.flat) == 72  # perfect 5x5 maze has fixed number of walls
    # for debugging:
    # print(bitmap)
    # names = {node: str(node) for _, node in np.ndenumerate(grid)}
    # positions = {
    #     node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    # from funmaze.render.graphviz import render_graphviz
    # render_graphviz(maze, names, positions).render(view=True)


def test_recursive_backtracker_recursion_depth(recursion_limit_50):
    grid = grid_sequential((100, 1))
    graph = neighbourhood_graph(grid)
    with pytest.raises(RecursionError):
        generate_recursive_backtracker(graph)
