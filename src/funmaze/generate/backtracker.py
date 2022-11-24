import random
from collections.abc import Mapping, Set, MutableSet

from funmaze.graph import Graph, Node, Edge


def _backtracker(
        neighbours: Mapping[Node, Set[Node]], maze: MutableSet[Edge[Node]],
        visited: MutableSet[Node], node: Node
) -> None:
    stack: list[Node] = []
    visited.add(node)
    stack.append(node)
    while stack:
        node = stack.pop()
        if good_neighbours := [
                node2 for node2 in neighbours[node] if node2 not in visited]:
            stack.append(node)
            node2 = random.choice(good_neighbours)
            maze.add((node, node2))
            visited.add(node2)
            stack.append(node2)


def generate_backtracker(graph: Graph[Node]) -> Graph[Node]:
    """Return a subgraph of *graph* representing a perfect maze on the graph,
    through the recursive backtracker algorithm.
    """
    maze: set[Edge[Node]] = set()
    neighbours: dict[Node, set[Node]] = {}
    for node1, node2 in graph:
        neighbours.setdefault(node1, set()).add(node2)
        neighbours.setdefault(node2, set()).add(node1)
    if graph:
        initial_node = random.choice(list(random.choice(list(graph))))
        visited: set[Node] = set()
        _backtracker(neighbours, maze, visited, initial_node)
    return maze


# TODO add a stack implementation which does not break recursion depth
