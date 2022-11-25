import itertools

import numpy as np
import pytest

from funmaze.graph import graph_remove_nodes, graph_nodes
from funmaze.graph.grid import grid_replace_nodes, grid_sequential, \
    neighbourhood_graph
from funmaze.render.graphviz import render_graphviz


def test_graphviz() -> None:
    grid = grid_replace_nodes(
        itertools.product(range(3, 5), range(3, 5)),
        np.uint(99),
        grid_replace_nodes(
            itertools.product(range(1, 4), range(1, 4)),
            np.uint(8),
            grid_sequential((7, 7))))
    graph = set(graph_remove_nodes(neighbourhood_graph(grid), {np.uint(99)}))
    nodes = frozenset(graph_nodes(graph))
    names = {node: str(node) for node in nodes}
    positions = {
        node: (pos[1], -pos[0]) for pos, node in np.ndenumerate(grid)}
    gv = render_graphviz(graph, names, positions)
    assert '0 [pos="0,0!"]' in gv.source
    assert '0 -- 1' in gv.source
    # for debugging:
    # gv.render("gv", view=True)
    gv2 = render_graphviz(graph, names)
    assert '0 [pos="0,0!"]' not in gv2.source
    assert '0 -- 1' in gv2.source
    # for debugging:
    # gv2.render("gv2", view=True)


def test_graphviz_2(tmp_path) -> None:
    grid = grid_sequential((2, 2))
    graph = frozenset(neighbourhood_graph(grid))
    nodes = frozenset(graph_nodes(graph))
    names = {node: "oops" for node in nodes}
    with pytest.raises(ValueError):  # names not unique
        render_graphviz(graph, names)
