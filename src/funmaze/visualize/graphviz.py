from collections.abc import Mapping

import graphviz

from funmaze.graph import Graph, Node, graph_nodes


def graph_graphviz(
        graph: Graph[Node],
        names: Mapping[Node, str],
        positions: Mapping[Node, tuple[float, float]] | None = None,
        engine: str = 'neato',
        fmt: str = 'pdf',
) -> graphviz.Graph:
    # note: default engine set to 'neato' to support node positions
    if len(set(names.values())) != len(names.values()):
        raise ValueError("names must be unique")
    gv = graphviz.Graph(strict=True, engine=engine, format=fmt)
    if positions is not None:
        for node in graph_nodes(graph):
            pos = positions[node]
            name = names[node]
            gv.node(name, pos=f"{pos[0]},{pos[1]}!")
    for edge in graph:
        edge_names = sorted(names[node] for node in edge)
        assert len(edge_names) == 2
        gv.edge(*edge_names)
    return gv
