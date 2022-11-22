import itertools
from collections.abc import Set
from dataclasses import dataclass, InitVar, field
from typing import TypeVar, Iterator, Iterable

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


def graph_remove_nodes(graph: Graph, nodes: Set[Node]
                       ) -> Graph[Node]:
    """Remove *nodes* from *graph*."""
    return {edge for edge in graph if all(node not in nodes for node in edge)}


def graph_merge_nodes(graph: Graph, nodes: Set[Node], target: Node
                      ) -> Graph[Node]:
    """Remove and merge *nodes* from *graph* into a *target* node.
    The *target* can be contained in *nodes*, can be a completely
    new node, or can be some other node in *graph*.
    """
    def _update(edge: Edge[Node]) -> Edge[Node] | None:
        good_nodes = frozenset(edge) - nodes
        if len(good_nodes) == 2:
            return edge
        elif len(good_nodes | {target}) == 2:
            return Edge(good_nodes | {target})
        else:
            return None
    return frozenset(
        edge2 for edge in graph if (edge2 := _update(edge)) is not None)
