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


def graph_nodes(top: Graph[Node]) -> Set[Node]:
    """Set of all nodes."""
    nodes: set[Node] = set()
    for edge in top:
        nodes.update(edge)
    return nodes


def graph_remove_nodes(graph: Graph, nodes: Set[Node]
                       ) -> Graph[Node]:
    """Remove a node from a graph.

    :param EdgeTopology graph: Base graph.
    :param Set[Cell] nodes: Set of nodes to be removed.
    """
    return {edge for edge in graph if all(node not in nodes for node in edge)}


def graph_merge_nodes(graph: Graph, nodes: Set[Node], target: Node
                      ) -> Graph[Node]:
    """Merge nodes into a new node.

    :param EdgeTopology graph: Base graph.
    :param Cell nodes: Cell to remove.
    :param Cell target: Target node.
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
