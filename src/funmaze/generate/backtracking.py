import random
from collections.abc import Mapping, Set

from funmaze.graph import Graph, Node, Edge, graph_neighbours


def generate_backtracking(graph: Graph[Node]) -> Graph[Node]:
    """Return a subgraph of *graph* representing a perfect maze on the graph,
    through the recursive backtracker algorithm.
    """
    maze: set[Edge[Node]] = set()
    if graph:
        neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
        node = random.choice(list(random.choice(list(graph))))
        visited: set[Node] = {node}
        stack: list[Node] = [node]
        while stack:
            node = stack.pop()
            if good_neighbours := [
                    node2 for node2 in neighbours.get(node, set())
                    if node2 not in visited]:
                stack.append(node)
                node2 = random.choice(good_neighbours)
                maze.add((node, node2))
                visited.add(node2)
                stack.append(node2)
    return maze
