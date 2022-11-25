import random
from collections.abc import Mapping, Set, Iterable

from funmaze.graph import IGraph, Node, Edge, graph_neighbours, \
    graph_undirected


def generate_backtracking_tree(graph: IGraph[Node], start: Node
                               ) -> Iterable[Edge[Node]]:
    """Return a random subgraph of *graph* representing a tree on the graph,
    rooted at *root*, through the recursive backtracker algorithm.
    The tree will be a spanning tree of the connected component of the graph
    that contains *root*.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    stack: list[Node] = [start]
    visited: set[Node] = {start}
    while stack:
        node = stack.pop()
        if good_neighbours := [
                node2 for node2 in neighbours.get(node, set())
                if node2 not in visited]:
            stack.append(node)
            node2 = random.choice(good_neighbours)
            visited.add(node2)
            stack.append(node2)
            yield node, node2


def generate_backtracking_maze(
        graph: IGraph[Node], start: Node) -> IGraph[Node]:
    """Return a perfect maze on the graph.
    First, we generate a tree through the recursive backtracker algorithm.
    Then, we complete the maze by adding all reverse edges.
    """
    return graph_undirected(generate_backtracking_tree(graph, start))
