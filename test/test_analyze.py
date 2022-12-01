from collections.abc import Sequence

import numpy as np
import pytest

from funmaze.analyze.bfs import analyze_bfs_branches, analyze_bfs_branches_many
from funmaze.generate.dfs import generate_dfs_maze
from funmaze.graph.grid import grid_squares, neighbourhood_graph


def mean_and_error(xs: Sequence[float]):
    n = len(xs)
    mean = sum(xs) / n
    err = (sum((x - mean) ** 2 for x in xs) / (n * (n - 1))) ** 0.5
    return mean, err


@pytest.mark.parametrize("shape", [(2, 2), (5, 5), (10, 10), (20, 20)])
def test_analyze_1(shape: tuple[int, ...]):
    grid = grid_squares(shape)
    graph = frozenset(neighbourhood_graph(grid))
    maze = frozenset(generate_dfs_maze(graph, grid[0, 0]))
    assert sum(analyze_bfs_branches(maze, grid[0, 0])) == len(maze) // 2


@pytest.mark.parametrize("shape", [(2, 2), (5, 5), (10, 10), (20, 20)])
def test_analyze_2(shape: tuple[int, ...]):
    grid = grid_squares(shape)
    graph = frozenset(neighbourhood_graph(grid))
    counter = analyze_bfs_branches_many(
        200, graph, grid[0, 0], generate_dfs_maze)
    avg, err = mean_and_error(list(counter.elements()))
    if shape == (2, 2):  # dfs on 2x2 is always a single branch of length 3
        assert avg == pytest.approx(3.0)
        assert err == pytest.approx(0.0)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.bar(counter.keys(), counter.values(), color="blue")
    # print(avg - 2 * err, avg + 2 * err)
    # plt.vlines(avg, 0, max(counter.values()), colors="red")
    # plt.show()
