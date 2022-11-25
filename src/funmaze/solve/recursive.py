from collections.abc import Set, Mapping, Iterable

from funmaze.graph import Graph, Node, Edge


def _combine(x, xs: Iterable) -> Iterable:
    yield x
    yield from xs


def _recursive(
        neighbours: Mapping[Node, Set[Node]], start: Node, end: Node,
        visited: Set[Node],
) -> Iterable[Iterable[Edge]]:
    if start == end:
        yield []
    else:
        neighbours_start: Set[Node] = neighbours.get(start, set())
        good_neighbours = [
            node for node in neighbours_start if node not in visited]
        for node in good_neighbours:
            for solution in _recursive(
                    neighbours, node, end, visited | {start}):
                yield _combine((start, node), solution)


def solve_recursive(graph: Graph[Node], start: Node, end: Node
                    ) -> Iterable[Iterable[Edge]]:
    """Recursively find all solutions without cycles.

    Warning: currently not suitable for large mazes as function recursion
    depth is likely to be exceeded.
    """
    neighbours: dict[Node, set[Node]] = {}
    for node1, node2 in graph:
        neighbours.setdefault(node1, set()).add(node2)
        neighbours.setdefault(node2, set()).add(node1)
    yield from _recursive(neighbours, start, end, set())
