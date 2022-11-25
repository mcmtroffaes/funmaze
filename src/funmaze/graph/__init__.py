import itertools
from collections.abc import Set, Mapping
from typing import TypeVar

Node = TypeVar("Node")


Edge = tuple[Node, Node]
"""Directed edge."""

Graph = Set[Edge[Node]]
"""Set of directed edges."""


def graph_nodes(graph: Graph[Node]) -> Set[Node]:
    """Set of all nodes of *graph*."""
    return frozenset(itertools.chain.from_iterable(graph))


def graph_neighbours(graph: Graph[Node]) -> Mapping[Node, Set[Node]]:
    neighbours: dict[Node, set[Node]] = {}
    for node1, node2 in graph:
        neighbours.setdefault(node1, set()).add(node2)
    return neighbours


def graph_remove_nodes(graph: Graph, nodes: Set[Node]) -> Graph[Node]:
    """Remove *nodes* from *graph*."""
    return frozenset(
        edge for edge in graph if all(node not in nodes for node in edge))


def graph_merge_nodes(
        graph: Graph, nodes: Set[Node], target: Node) -> Graph[Node]:
    """Remove and merge *nodes* from *graph* into a *target* node.
    """
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
    return frozenset(
        edge2 for edge in graph if (edge2 := _update(edge)) is not None)
