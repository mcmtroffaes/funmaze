import pytest

from funmaze.generate.backtracking import generate_backtracking_maze
from funmaze.graph import Graph
from funmaze.graph.grid import grid_squares, neighbourhood_graph
from funmaze.solve.bfs import solve_bfs_one_shortest


def test_shortest_1() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 2)}
    sol = solve_bfs_one_shortest(maze, 0, 2)
    assert sol is not None
    assert tuple(sol) == (0, 2)


def test_shortest_2() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 3), (3, 2)}
    sol = solve_bfs_one_shortest(maze, 0, 2)
    assert sol is not None
    assert tuple(sol) in {(0, 1, 2), (0, 3, 2)}


@pytest.mark.parametrize("shape", [
    (2, 2), (5, 5), (2, 5), (5, 2), (10, 10), (50, 100), (200, 20)
])
def test_shortest_3(shape: tuple[int, ...]) -> None:
    grid = grid_squares(shape)
    start = grid[0, 0]
    end = grid[shape[0] - 1, shape[1] - 1]
    graph = neighbourhood_graph(grid)
    maze = generate_backtracking_maze(graph, start)
    sol = solve_bfs_one_shortest(maze, start, end)
    assert sol is not None  # perfect maze only has one solution
    # for debugging
    # from funmaze.render.bitmap import render_bitmap
    # bitmap = render_bitmap(grid, maze).astype(int)
    # bitmap2 = render_bitmap(grid, {sol}).astype(int)
    # print(bitmap + (1 - bitmap2) * 2)


def test_shortest_4() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (2, 3), (4, 5)}
    assert solve_bfs_one_shortest(maze, 0, 5) is None
