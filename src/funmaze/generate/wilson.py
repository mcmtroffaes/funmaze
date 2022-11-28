"""Wilson's algorithm for generating uniform spanning trees."""
# see https://doi.org/10.1145/237814.237880 for original algorithm
# see https://www.kurims.kyoto-u.ac.jp/~kumagai/LN-barlow1.pdf for some notes

import random
from collections.abc import Iterable, Set, Mapping, Sequence

from funmaze.graph import IGraph, Node, \
    graph_from_path, graph_undirected, graph_neighbours


def simple_random_walk(
        neighbours: Mapping[Node, Sequence[Node]], start: Node
) -> Iterable[Node]:
    """Infinite simple uniform random walk from *start*.
    Raises :exc:`KeyError` if a node is not present in *neighbours*.
    Raises :exc:`IndexError` if a node has no neighbours.
    """
    node = start
    while True:
        yield node
        node = random.choice(neighbours[node])


def loop_erased_random_walk(
    neighbours: Mapping[Node, Sequence[Node]], start: Node, end: Set[Node]
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
    assert False, "code unreachable"  # pragma: no cover


def generate_wilson_tree(graph: IGraph[Node], start: Node) -> IGraph[Node]:
    """Return a random in-tree on *graph* rooted at *start*,
    through Wilson's algorithm.

    The tree will be a spanning tree of the graph, uniformly sampled from
    the set of all spanning in-trees rooted at *start*.

    Important: for every node in *graph*, there must be at least one path
    to *start*, otherwise the algorithm will throw an :exc:`IndexError`
    (if it finds a terminal node before reaching *start*),
    or will not terminate (if it gets stuck in an absorbing class).
    """
    neighbours = graph_neighbours(graph)
    unvisited = set(neighbours)
    unvisited.remove(start)
    visited: set[Node] = {start}
    while unvisited:
        node = unvisited.pop()
        path = loop_erased_random_walk(neighbours, node, visited)
        path_set = frozenset(path)
        visited |= path_set
        unvisited -= path_set
        yield from graph_from_path(path)


def generate_wilson_maze(graph: IGraph[Node], start: Node) -> IGraph[Node]:
    """Return a perfect maze on the graph.
    First, we generate a spanning forest through the Wilson algorithm.
    Then, we complete the maze by adding all reverse edges.
    """
    return graph_undirected(generate_wilson_tree(graph, start))
