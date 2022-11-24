import numpy as np

from funmaze.generate.backtracking import generate_backtracking
from funmaze.graph.grid import grid_sequential, neighbourhood_graph
from funmaze.render.bitmap import render_bitmap
from funmaze.solve.backtracking import solve_backtracking


def test_backtracking_simple():
    grid = grid_sequential((10, 10))
    graph = neighbourhood_graph(grid)
    maze = generate_backtracking(graph)
    bitmap = render_bitmap(grid, maze).astype(int)
    for solution in solve_backtracking(maze, np.uint(0), np.uint(99)):
        bitmap2 = render_bitmap(grid, solution).astype(int)
        # for debugging
        # print(bitmap + (1 - bitmap2) * 2)
