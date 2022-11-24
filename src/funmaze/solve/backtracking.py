import random
from collections.abc import Set, Mapping, Iterable

from funmaze.graph import Graph, Node


def _backtracking(
        neighbours: Mapping[Node, Set[Node]], start: Node, end: Node,
        visited: Set[Node],
) -> Iterable[Graph[Node]]:
    # TODO use stack implementation?
    neighbours_start = neighbours[start]
    if end in neighbours_start:
        yield {(start, end)}
    else:
        good_neighbours = [
            node for node in neighbours_start if node not in visited]
        for node in good_neighbours:
            for solution in _backtracking(
                    neighbours, node, end, visited | {start}):
                yield {(start, node)} | solution


def solve_backtracking(graph: Graph[Node], start: Node, end: Node
                       ) -> Iterable[Graph[Node]]:
    neighbours: dict[Node, set[Node]] = {}
    for node1, node2 in graph:
        neighbours.setdefault(node1, set()).add(node2)
        neighbours.setdefault(node2, set()).add(node1)
    yield from _backtracking(neighbours, start, end, set())
