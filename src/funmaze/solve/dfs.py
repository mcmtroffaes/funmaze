import random
from collections.abc import Mapping, Set, Iterable, Sequence

from funmaze.graph import IGraph, Node, graph_neighbours


def solve_dfs_one(graph: IGraph[Node], start: Node, end: Node
                  ) -> Sequence[Node] | None:
    """Use the backtracking algorithm for generating trees to find a solution
    to the maze. We do this by returning the stack as soon as the generator
    reaches the end state. This routine will return at most one solution.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    stack: list[Node] = [start]
    visited: set[Node] = {start}
    while stack and stack[-1] != end:
        node = stack.pop()
        if good_neighbours := [
                node2 for node2 in neighbours.get(node, set())
                if node2 not in visited]:
            stack.append(node)
            node2 = random.choice(good_neighbours)
            visited.add(node2)
            stack.append(node2)
    return stack if stack else None
