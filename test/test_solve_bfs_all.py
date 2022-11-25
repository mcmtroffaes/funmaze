import pytest

from funmaze.generate.backtracking import generate_backtracking_maze
from funmaze.graph import Graph
from funmaze.graph.grid import grid_squares, neighbourhood_graph
from funmaze.solve.bfs import solve_bfs_all


def test_bfs_all_1() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 2)}
    sols = {tuple(sol) for sol in solve_bfs_all(maze, 0, 2)}
    assert sols == {(0, 1, 2), (0, 2)}


def test_bfs_all_2() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 3), (3, 2)}
    sols = {tuple(sol) for sol in solve_bfs_all(maze, 0, 2)}
    assert sols == {(0, 1, 2), (0, 3, 2)}


@pytest.mark.parametrize("shape", [
    (2, 2), (5, 5), (2, 5), (5, 2), (10, 10),
])
def test_bfs_all_3(shape: tuple[int, ...]) -> None:
    grid = grid_squares(shape)
    start = grid[0, 0]
    end = grid[shape[0] - 1, shape[1] - 1]
    graph = neighbourhood_graph(grid)
    maze = set(generate_backtracking_maze(graph, start))
    solutions = list(solve_bfs_all(maze, start, end))
    assert len(solutions) == 1  # perfect maze only has one solution
    # for debugging
    # from funmaze.render.bitmap import render_bitmap
    # bitmap = render_bitmap(grid, maze).astype(int)
    # bitmap2 = render_bitmap(grid, set(solutions[0])).astype(int)
    # print(bitmap + (1 - bitmap2) * 2)


def test_bfs_all_4() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (2, 3), (4, 5)}
    sols = list(solve_bfs_all(maze, 0, 5))
    assert not sols