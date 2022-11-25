from collections.abc import Mapping

import graphviz

from funmaze.graph import Graph, Node, graph_nodes


def render_graphviz(
        graph: Graph[Node],
        names: Mapping[Node, str],
        positions: Mapping[Node, tuple[float, float]] | None = None,
        engine: str = 'neato',
        fmt: str = 'pdf',
) -> graphviz.Digraph:
    # note: default engine set to 'neato' to support node positions
    if len(frozenset(names.values())) != len(names):
        raise ValueError("names must be unique")
    gv = graphviz.Digraph(strict=True, engine=engine, format=fmt)
    if positions is not None:
        for node in frozenset(graph_nodes(graph)):
            pos = positions[node]
            name = names[node]
            gv.node(name, pos=f"{pos[0]},{pos[1]}!")
    for node1, node2 in graph:
        gv.edge(names[node1], names[node2])
    return gv
