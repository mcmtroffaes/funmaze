import random
from collections.abc import Mapping, Set
from typing import Iterable

from funmaze.graph import Graph, Node, Edge, graph_neighbours


def generate_backtracking_tree(graph: Graph[Node], start: Node
                               ) -> Iterable[Edge[Node]]:
    """Return a random subgraph of *graph* representing a tree on the graph,
    rooted at *root*, through the recursive backtracker algorithm.
    The tree will be a spanning tree of the connected component of the graph
    that contains *root*.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    visited: set[Node] = {start}
    stack: list[Node] = [start]
    while stack:
        node = stack.pop()
        if good_neighbours := [
                node2 for node2 in neighbours.get(node, set())
                if node2 not in visited]:
            stack.append(node)
            node2 = random.choice(good_neighbours)
            yield node, node2
            visited.add(node2)
            stack.append(node2)


def generate_backtracking_maze(graph: Graph[Node], start: Node) -> Graph[Node]:
    """Return a perfect maze on the graph.
    First, we generate a tree through the recursive backtracker algorithm.
    Then, we complete the maze by adding all reverse edges.
    """
    maze: set[Edge[Node]] = set()
    for node1, node2 in generate_backtracking_tree(graph, start):
        maze.add((node1, node2))
        maze.add((node2, node1))
    return maze
