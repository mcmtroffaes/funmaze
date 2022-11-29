import pytest

from funmaze.analyze.bfs import analyze_bfs_branches, analyze_bfs_branches_many
from funmaze.generate.dfs import generate_dfs_maze
from funmaze.graph.grid import grid_squares, neighbourhood_graph


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
    analyze_bfs_branches_many(
        200, graph, grid[0, 0], generate_dfs_maze)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.bar(counter.keys(), counter.values(), color="blue")
    # avg = sum(number * count for number, count in counter.items())
    # avg /= sum(count for number, count in counter.items())
    # print(avg)
    # plt.vlines(avg, 0, max(counter.values()), colors="red")
    # plt.show()
