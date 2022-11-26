"""Wilson's algorithm for generating uniform spanning trees."""
# see https://www.kurims.kyoto-u.ac.jp/~kumagai/LN-barlow1.pdf for some notes

import random
from collections.abc import Iterable, Set, Mapping, Sequence

from funmaze.graph import IGraph, Node, \
    graph_from_path, graph_undirected, graph_nodes_neighbours


def simple_random_walk(
        neighbours: Mapping[Node, Set[Node]], start: Node) -> Iterable[Node]:
    """Simple random walk from *start*."""
    node = start
    yield node
    while good_neighbours := list(neighbours.get(node, set())):
        node = random.choice(good_neighbours)
        yield node


def loop_erased_random_walk(
    neighbours: Mapping[Node, Set[Node]], start: Node, end: Set[Node]
) -> Sequence[Node]:
    """Loop erased random walk from *start* to *end*."""
    path: list[Node] = []
    path_set: set[Node] = set()
    for node in simple_random_walk(neighbours, start):
        while node in path_set:
            path_set.remove(path.pop())
        path.append(node)
        path_set.add(node)
        if node in end:
            return path
    raise ValueError(f"unable to go from {start} to {end}")


def generate_wilson_forest(graph: IGraph[Node]) -> IGraph[Node]:
    """Generate a spanning forest subgraph of *graph*, through Wilson's
    algorithm.

    If the graph is connected, this will be a single spanning tree.
    Otherwise, it will be a spanning forest with one spanning tree in each
    connected component of the graph.
    """
    nodes, neighbours = graph_nodes_neighbours(graph)
    unvisited = set(nodes)
    visited: set[Node] = {unvisited.pop()}
    while unvisited:
        node = unvisited.pop()
        path = loop_erased_random_walk(neighbours, node, visited)
        path_set = set(path)
        visited |= path_set
        unvisited -= path_set
        yield from graph_from_path(path)


def generate_wilson_maze(graph: IGraph[Node]) -> IGraph[Node]:
    """Return a perfect maze on the graph.
    First, we generate a spanning forest through the Wilson algorithm.
    Then, we complete the maze by adding all reverse edges.
    """
    return graph_undirected(generate_wilson_forest(graph))
