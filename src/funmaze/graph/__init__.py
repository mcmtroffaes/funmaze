import itertools
from collections.abc import Set, Mapping, Iterable, Sequence
from typing import TypeVar

Node = TypeVar("Node")


Edge = tuple[Node, Node]
"""Directed edge."""

Graph = Set[Edge[Node]]
"""Set of directed edges."""


IGraph = Iterable[Edge[Node]]
"""Iterable of directed edges (identical edges may appear more than once)."""


def graph_nodes(graph: IGraph[Node]) -> Iterable[Node]:
    """Iterate over all nodes of *graph*. May contain duplicates.
    Wrap the result into a set to get rid of those.
    """
    return itertools.chain.from_iterable(graph)


def graph_from_path(path: Sequence[Node]) -> IGraph[Node]:
    return zip(path[:-1], path[1:])


def graph_neighbours(graph: IGraph[Node]) -> Mapping[Node, Set[Node]]:
    """Find neighbours ("forward star") of every node."""
    neighbours: dict[Node, set[Node]] = {}
    for node1, node2 in graph:
        neighbours.setdefault(node1, set()).add(node2)
    return neighbours


def graph_remove_nodes(graph: IGraph, nodes: Set[Node]) -> IGraph[Node]:
    """Remove *nodes* from *graph*."""
    return (
        edge for edge in graph if all(node not in nodes for node in edge))


def graph_merge_nodes(
        graph: IGraph, nodes: Set[Node], target: Node) -> IGraph[Node]:
    """Remove and merge *nodes* from *graph* into a *target* node."""
    def _update(edge: Edge[Node]) -> Edge[Node]:
        node1, node2 = edge
        if node1 not in nodes:
            if node2 not in nodes:
                return edge
            else:
                return node1, target
        else:
            if node2 not in nodes:
                return target, node2
            else:
                return target, target
    return (
        edge2 for edge in graph if (edge2 := _update(edge)) is not None)


def graph_undirected(graph: IGraph[Node]) -> IGraph[Node]:
    """Convert graph into an undirected graph.
    Edges may appear more than once in the resulting iterable.
    Wrap the result into a set if you need to get rid of those.
    """
    for node1, node2 in graph:
        yield node1, node2
        yield node2, node1
