"""Standard depth-first-search "backtracking" algorithms."""

import random
from collections.abc import Mapping, Set

from funmaze.graph import IGraph, Node, graph_neighbours, graph_undirected


def generate_dfs_tree(graph: IGraph[Node], start: Node
                      ) -> IGraph[Node]:
    """Return a random out-tree on *graph* rooted at *root*,
    through depth-first-search.

    The out-tree will be a spanning tree of the connected component
    of the graph that contains *root*.

    This algorithm is also known as the recursive backtracker.
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


def generate_dfs_maze(
        graph: IGraph[Node], start: Node) -> IGraph[Node]:
    """Return a perfect maze on the graph.
    First, we generate a tree through the recursive backtracker algorithm.
    Then, we complete the maze by adding all reverse edges.
    """
    return graph_undirected(generate_dfs_tree(graph, start))
