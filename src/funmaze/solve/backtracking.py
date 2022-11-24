import random
from collections.abc import Iterable

from funmaze.graph import Graph, Node, Edge


def solve_backtracking(graph: Graph[Node], start: Node, end: Node
                       ) -> Iterable[Iterable[Edge]]:
    """Use the backtracking algorithm for generating mazes to solve
    the maze. We do this by returning the stack as soon as the generator
    reaches the end state. This routine will return at most one solution.
    """
    neighbours: dict[Node, set[Node]] = {}
    for node1, node2 in graph:
        neighbours.setdefault(node1, set()).add(node2)
        neighbours.setdefault(node2, set()).add(node1)
    stack: list[Node] = [start]
    visited: set[Node] = {start}
    while stack and stack[-1] != end:
        node = stack.pop()
        if good_neighbours := [
                node2 for node2 in neighbours[node] if node2 not in visited]:
            stack.append(node)
            node2 = random.choice(good_neighbours)
            visited.add(node2)
            stack.append(node2)
    return [zip(stack[:-1], stack[1:])]
