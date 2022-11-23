import random
from collections.abc import Mapping, Set, MutableSet

from funmaze.graph import Graph, Node, Edge, graph_nodes


def _recursive_backtracker(
        neighbours: Mapping[Node, Set[Node]], maze: MutableSet[Edge[Node]],
        visited: MutableSet[Node], node: Node
) -> None:
    visited.add(node)
    while good_neighbours := [
            node2 for node2 in neighbours[node] if node2 not in visited]:
        node2 = random.choice(good_neighbours)
        maze.add(Edge([node, node2]))
        _recursive_backtracker(neighbours, maze, visited, node2)


def generate_backtracker(graph: Graph[Node]) -> Graph[Node]:
    """Return a subgraph of *graph* representing a perfect maze on the graph,
    through the recursive backtracker algorithm.
    """
    maze: set[Edge[Node]] = set()
    nodes = graph_nodes(graph)
    neighbours: dict[Node, set[Node]] = {}
    for edge in graph:
        node1, node2 = tuple(edge)
        neighbours.setdefault(node1, set()).add(node2)
        neighbours.setdefault(node2, set()).add(node1)
    if nodes:
        initial_node = random.choice(list(nodes))
        visited: set[Node] = set()
        _recursive_backtracker(neighbours, maze, visited, initial_node)
    return maze


# TODO add a stack implementation which does not break recursion depth
