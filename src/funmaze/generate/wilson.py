"""Wilson's algorithm for generating uniform spanning trees."""
import itertools
# see https://www.kurims.kyoto-u.ac.jp/~kumagai/LN-barlow1.pdf for some notes

import random
from collections.abc import Iterable, Set, Mapping, Sequence

from funmaze.graph import IGraph, Node, Edge, graph_neighbours, \
    graph_from_path, graph_undirected


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
    path = itertools.takewhile(
        lambda node_: node_ not in end,
        simple_random_walk(neighbours, start))
    erased_path: list[Node] = []
    erased_visited: set[Node] = set()
    for node in path:
        if node in erased_visited:
            while node in erased_visited:
                erased_visited.remove(erased_path.pop())
        erased_visited.add(node)
        erased_path.append(node)
    return erased_path


def generate_wilson_tree(graph: IGraph[Node]) -> IGraph[Node]:
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    unvisited_set: set[Node] = set(neighbours)
    node = unvisited_set.pop()
    visited_set: set[Node] = {node}
    while unvisited_set:
        node = unvisited_set.pop()
        path = loop_erased_random_walk(neighbours, node, visited_set)
        for node2 in path:
            visited_set.add(node2)
            unvisited_set.remove(node2)
        yield from graph_from_path(path)


def generate_wilson_maze(graph: IGraph[Node]) -> IGraph[Node]:
    """Return a perfect maze on the graph.
    First, we generate a tree through the recursive backtracker algorithm.
    Then, we complete the maze by adding all reverse edges.
    """
    return graph_undirected(generate_wilson_tree(graph))
