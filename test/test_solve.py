import itertools

import pytest

from funmaze.generate.dfs import generate_dfs_maze
from funmaze.graph import Graph, Node, graph_undirected
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
    _assert_sol_imperfect(sols1, sols2, tuple(sol3), tuple(sol4))


def test_bfs_2() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (0, 3), (3, 2)}
    sols1 = list(tuple(sol) for sol in solve_bfs_all(maze, 0, 2))
    assert set(sols1) == {(0, 1, 2), (0, 3, 2)}
    sols2 = list(tuple(sol) for sol in solve_bfs_all_shortest(maze, 0, 2))
    assert set(sols2) == {(0, 1, 2), (0, 3, 2)}
    sol3 = solve_bfs_one_shortest(maze, 0, 2)
    assert sol3 is not None
    assert tuple(sol3) in {(0, 1, 2), (0, 3, 2)}
    sol4 = solve_dfs_one(maze, 0, 2)
    assert sol4 is not None
    assert tuple(sol4) in {(0, 1, 2), (0, 3, 2)}
    _assert_sol_imperfect(sols1, sols2, tuple(sol3), tuple(sol4))


def test_bfs_3() -> None:
    maze: Graph[int] = {(0, 1), (1, 2), (2, 3), (4, 5)}
    sols1 = list(tuple(sol) for sol in solve_bfs_all(maze, 0, 5))
    assert not sols1
    sols2 = list(tuple(sol) for sol in solve_bfs_all_shortest(maze, 0, 5))
    assert not sols2
    sol3 = solve_bfs_one_shortest(maze, 0, 5)
    assert sol3 is None
    sol4 = solve_dfs_one(maze, 0, 5)
    assert sol4 is None


@pytest.mark.parametrize("shape", [
    (2, 2), (5, 5), (2, 5), (5, 2), (10, 10),
])
def test_bfs_4(shape: tuple[int, ...]) -> None:
    grid = grid_squares(shape)
    start = grid[0, 0]
    end = grid[shape[0] - 1, shape[1] - 1]
    graph = neighbourhood_graph(grid)
    maze = set(generate_dfs_maze(graph, start))
    # perfect maze has only one solution
    sols1 = list(tuple(sol) for sol in solve_bfs_all(maze, start, end))
    assert len(sols1) == 1
    sols2 = list(
        tuple(sol) for sol in solve_bfs_all_shortest(maze, start, end))
    assert len(sols2) == 1
    sol3 = solve_bfs_one_shortest(maze, start, end)
    assert sol3 is not None
    sol4 = solve_dfs_one(maze, start, end)
    assert sol4 is not None
    _assert_sol_perfect(sols1, sols2, tuple(sol3), tuple(sol4))
    # for debugging
    # from funmaze.render.bitmap import render_bitmap
    # bitmap = render_bitmap(grid, maze).astype(int)
    # bitmap2 = render_bitmap(grid, set(solutions[0])).astype(int)
    # print(bitmap + (1 - bitmap2) * 2)


def test_bfs_5() -> None:
    # 0-1-2-3-4-5
    # | |     | |
    # 6-7-8-9-1011
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
             (6, 7), (7, 8), (8, 9), (9, 10), (10, 11),
             (0, 6), (1, 7), (4, 10), (5, 11)]
    maze = frozenset(graph_undirected(edges))
    sols1 = list(tuple(sol) for sol in solve_bfs_all(maze, 0, 11))
    assert sorted(sols1) == [
        (0, 1, 2, 3, 4, 5, 11),
        (0, 1, 2, 3, 4, 10, 11),
        (0, 1, 7, 8, 9, 10, 4, 5, 11),
        (0, 1, 7, 8, 9, 10, 11),
        (0, 6, 7, 1, 2, 3, 4, 5, 11),
        (0, 6, 7, 1, 2, 3, 4, 10, 11),
        (0, 6, 7, 8, 9, 10, 4, 5, 11),
        (0, 6, 7, 8, 9, 10, 11)]
    sols2 = list(
        tuple(sol) for sol in solve_bfs_all_shortest(maze, 0, 11))
    assert sorted(sols2) == [
        (0, 1, 2, 3, 4, 5, 11),
        (0, 1, 2, 3, 4, 10, 11),
        (0, 1, 7, 8, 9, 10, 11),
        (0, 6, 7, 8, 9, 10, 11)]
    sol3 = solve_bfs_one_shortest(maze, 0, 11)
    assert tuple(sol3) in sols2
    sol4 = solve_dfs_one(maze, 0, 11)
    assert tuple(sol4) in sols1
    _assert_sol_perfect(sols1, sols2, tuple(sol3), tuple(sol4))


def _assert_sol_perfect(
        sols_all: list[tuple[Node, ...]],
        sols_all_shortest: list[tuple[Node, ...]],
        sol_one_shortest: tuple[Node, ...],
        sol_dfs_one: tuple[Node, ...]
        ) -> None:
    """Check solution of a perfect maze."""
    shortest_length = len(sols_all[0])
    assert set(sols_all_shortest) == set(
        itertools.takewhile(lambda sol: len(sol) == shortest_length, sols_all))
    assert sol_one_shortest in set(sols_all_shortest)
    assert sol_one_shortest == sol_dfs_one


def _assert_sol_imperfect(
        sols_all: list[tuple[Node, ...]],
        sols_all_shortest: list[tuple[Node, ...]],
        sol_one_shortest: tuple[Node, ...],
        sol_dfs_one: tuple[Node, ...]
        ) -> None:
    """Check solution of an imperfect maze."""
    shortest_length = len(sols_all[0])
    assert set(sols_all_shortest) == set(
        itertools.takewhile(lambda sol: len(sol) == shortest_length, sols_all))
    assert sol_one_shortest in set(sols_all_shortest)
    assert sol_dfs_one in sols_all
