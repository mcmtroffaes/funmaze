from collections.abc import Set, Mapping, Iterable

from funmaze.graph import Graph, Node, graph_neighbours


def _combine(x: Node, xs: Iterable[Node]) -> Iterable[Node]:
    yield x
    yield from xs


def _recursive(
        neighbours: Mapping[Node, Set[Node]], start: Node, end: Node,
        visited: Set[Node],
) -> Iterable[Iterable[Node]]:
    if start == end:
        yield [start]
    else:
        neighbours_start: Set[Node] = neighbours.get(start, set())
        good_neighbours = [
            node for node in neighbours_start if node not in visited]
        for node in good_neighbours:
            for solution in _recursive(
                    neighbours, node, end, visited | {start}):
                yield _combine(start, solution)


def solve_recursive(graph: Graph[Node], start: Node, end: Node
                    ) -> Iterable[Iterable[Node]]:
    """Recursively find all solutions without cycles.

    Warning: currently not suitable for large mazes as function recursion
    depth is likely to be exceeded.
    """
    return _recursive(graph_neighbours(graph), start, end, set())
