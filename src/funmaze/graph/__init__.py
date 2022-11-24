import itertools
from collections.abc import Set
from dataclasses import dataclass, InitVar, field
from typing import TypeVar, Iterator, Iterable

import numpy as np

Node = TypeVar("Node")


@dataclass(eq=True, frozen=True)
class Edge(Set[Node]):
    """Undirected edge."""

    _nodes: Set[Node] = field(init=False)
    nodes: InitVar[Iterable[Node]]

    def __post_init__(self, nodes: Iterable[Node]) -> None:
        object.__setattr__(self, '_nodes', frozenset(nodes))
        if len(self._nodes) != 2:
            raise ValueError("edge must have exactly two distinct nodes")

    def __contains__(self, other: object) -> bool:
        return other in self._nodes

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[Node]:
        return self._nodes.__iter__()


Graph = Set[Edge[Node]]
"""Set of undirected edges."""


def graph_nodes(graph: Graph[Node]) -> Set[Node]:
    """Set of all nodes of *graph*."""
    return frozenset(itertools.chain.from_iterable(graph))


def graph_grid(shape: tuple[int, ...]) -> Graph[tuple[int, int]]:
    """Construct a grid shaped graph."""
    def _edges():
        for pos in np.ndindex(*shape):
            for i, (x, max_x) in enumerate(zip(pos, shape)):
                if x + 1 < max_x:
                    pos2 = tuple((xx + 1) if i == j else xx
                                 for j, xx in enumerate(pos))
                    yield Edge((pos, pos2))
    return frozenset(_edges())


def graph_remove_nodes(graph: Graph, nodes: Set[Node]) -> Graph[Node]:
    """Remove *nodes* from *graph*."""
    return frozenset(
        edge for edge in graph if all(node not in nodes for node in edge))


def graph_merge_nodes(
        graph: Graph, nodes: Set[Node], target: Node) -> Graph[Node]:
    """Remove and merge *nodes* from *graph* into a *target* node.
    The *nodes* must contain the *target*.
    """
    def _update(edge: Edge[Node]) -> Edge[Node] | None:
        node1, node2 = edge
        if node1 not in nodes:
            if node2 not in nodes:
                return edge
            else:
                return Edge((node1, target))
        else:
            if node2 not in nodes:
                return Edge((target, node2))
            else:
                return None
    if target not in nodes:
        raise ValueError("nodes must contain target")
    return frozenset(
        edge2 for edge in graph if (edge2 := _update(edge)) is not None)
