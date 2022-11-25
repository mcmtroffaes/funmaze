import itertools
from collections.abc import Set, Mapping
from typing import TypeVar, Iterable, Collection

import numpy as np

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


def neighbourhood_positions(
        shape: tuple[int, ...], pos: tuple[int, ...],
        steps: Collection[int] = (-1, 1),
) -> Iterable[tuple[int, ...]]:
    """List all positions neighbouring *pos* on a grid of the given *shape*,
    assuming we can move in *steps* along a single axis.
    """
    for i, (x, x_max) in enumerate(zip(pos, shape)):
        for step in steps:
            if 0 <= x + step < x_max:
                yield tuple((x2 + step) if i == j else x2
                            for j, x2 in enumerate(pos))


def graph_grid(shape: tuple[int, ...], steps: Collection[int] = (-1, 1)
               ) -> Graph[tuple[int, ...]]:
    """Construct a grid shaped graph."""
    def _edges():
        for pos1 in np.ndindex(*shape):
            for pos2 in neighbourhood_positions(shape, pos1, steps):
                yield pos1, pos2
    return frozenset(_edges())


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
