import pytest

from funmaze.generate.dfs import generate_dfs_maze
from funmaze.graph import Graph
from funmaze.graph.grid import grid_squares, neighbourhood_graph
from funmaze.solve.bfs import solve_bfs_all, solve_bfs_one_shortest, \
    solve_bfs_all_shortest
from funmaze.solve.dfs import solve_dfs_one


def test_bfs_1() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 2)}
    sols1 = [tuple(sol) for sol in solve_bfs_all(maze, 0, 2)]
    assert sols1 == [(0, 2), (0, 1, 2)]
    sols2 = [tuple(sol) for sol in solve_bfs_all_shortest(maze, 0, 2)]
    assert sols2 == [(0, 2)]
    sol3 = solve_bfs_one_shortest(maze, 0, 2)
    assert sol3 is not None
    assert tuple(sol3) == (0, 2)
    sol4 = solve_dfs_one(maze, 0, 2)
    assert sol4 is not None
    assert tuple(sol4) in {(0, 1, 2), (0, 2)}


def test_bfs_2() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 3), (3, 2)}
    sols1 = set(tuple(sol) for sol in solve_bfs_all(maze, 0, 2))
    assert sols1 == {(0, 1, 2), (0, 3, 2)}
    sols2 = set(tuple(sol) for sol in solve_bfs_all_shortest(maze, 0, 2))
    assert sols2 == {(0, 1, 2), (0, 3, 2)}
    sol3 = solve_bfs_one_shortest(maze, 0, 2)
    assert sol3 is not None
    assert tuple(sol3) in {(0, 1, 2), (0, 3, 2)}
    sol4 = solve_dfs_one(maze, 0, 2)
    assert sol4 is not None
    assert tuple(sol4) in {(0, 1, 2), (0, 3, 2)}


def test_bfs_3() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (2, 3), (4, 5)}
    sols1 = list(solve_bfs_all(maze, 0, 5))
    assert not sols1
    sols2 = list(solve_bfs_all_shortest(maze, 0, 5))
    assert not sols2
    sol3 = solve_bfs_one_shortest(maze, 0, 5)
    assert sol3 is None
    sol4 = solve_dfs_one(maze, 0, 5)
    assert sol4 is None


@pytest.mark.parametrize("shape", [
    (2, 2), (5, 5), (2, 5), (5, 2), (10, 10),
])
def test_bfs_all_4(shape: tuple[int, ...]) -> None:
    grid = grid_squares(shape)
    start = grid[0, 0]
    end = grid[shape[0] - 1, shape[1] - 1]
    graph = neighbourhood_graph(grid)
    maze = set(generate_dfs_maze(graph, start))
    # perfect maze has only one solution
    sols1 = list(solve_bfs_all(maze, start, end))
    assert len(sols1) == 1
    sols2 = list(solve_bfs_all_shortest(maze, start, end))
    assert len(sols2) == 1
    sol3 = solve_bfs_one_shortest(maze, start, end)
    assert sol3 is not None
    sol4 = solve_dfs_one(maze, start, end)
    assert sol4 is not None
    # for debugging
    # from funmaze.render.bitmap import render_bitmap
    # bitmap = render_bitmap(grid, maze).astype(int)
    # bitmap2 = render_bitmap(grid, set(solutions[0])).astype(int)
    # print(bitmap + (1 - bitmap2) * 2)
