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


def generate_recursive_backtracker(graph: Graph[Node]) -> Graph[Node]:
    """Return a subgraph of *graph* representing a perfect maze on the graph,
    through the recursive backtracker algorithm.
    """
    maze: set[Edge[Node]] = set()
    nodes = graph_nodes(graph)
    neighbours = {
        node: set(
            tuple(node2)[0] for edge in graph
            if len(node2 := set(edge) - {node}) == 1)
        for node in nodes
    }
    if nodes:
        initial_node = random.choice(list(nodes))
        visited: set[Node] = set()
        _recursive_backtracker(neighbours, maze, visited, initial_node)
    return maze


# TODO add a stack implementation which does not break recursion depth
