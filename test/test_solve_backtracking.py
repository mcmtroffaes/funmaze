import numpy as np

from funmaze.generate.backtracking import generate_backtracking
from funmaze.graph import Graph
from funmaze.graph.grid import grid_sequential, neighbourhood_graph
from funmaze.solve.backtracking import solve_backtracking


def test_backtracking_1() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 2)}
    sols = list(frozenset(sol) for sol in solve_backtracking(maze, 0, 2))
    assert len(sols) == 1
    assert sols[0] in {
        frozenset([(0, 1), (1, 2)]),
        frozenset([(0, 2)]),
    }


def test_backtracking_2() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 3), (3, 2)}
    sols = list(frozenset(sol) for sol in solve_backtracking(maze, 0, 2))
    assert len(sols) == 1
    assert sols[0] in {
        frozenset([(0, 1), (1, 2)]),
        frozenset([(0, 3), (3, 2)]),
    }


def test_backtracking_3() -> None:
    return
    grid = grid_sequential((10, 10))
    graph = neighbourhood_graph(grid)
    maze = generate_backtracking(graph)
    solutions = list(solve_backtracking(maze, np.uint(0), np.uint(99)))
    assert len(solutions) == 1  # perfect maze only has one solution
    # for debugging
    # from funmaze.render.bitmap import render_bitmap
    # bitmap = render_bitmap(grid, maze).astype(int)
    # bitmap2 = render_bitmap(grid, set(solutions[0])).astype(int)
    # print(bitmap + (1 - bitmap2) * 2)


def test_backtracking_4() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (2, 3), (4, 5)}
    sols = list(frozenset(sol) for sol in solve_backtracking(maze, 0, 5))
    assert not sols
