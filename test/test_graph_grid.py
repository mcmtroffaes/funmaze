from collections.abc import Set

import pytest

from funmaze.graph.grid import grid_sequential, neighbourhood_graph


@pytest.mark.parametrize("shape,graph", [
    # 0 1
    ((1, 2), {(0, 1)}),
    # 0 1
    # 2 3
    # 4 5
    ((3, 2), {(0, 1), (2, 3), (4, 5), (0, 2), (2, 4), (1, 3), (3, 5)}),
])
def test_grid_neighbourhood_graph(
        shape: tuple[int, ...], graph: Set[tuple[int, int]]) -> None:
    grid = grid_sequential(shape)
    assert neighbourhood_graph(grid, steps=(1,)) == graph
