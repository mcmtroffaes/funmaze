import random
from collections.abc import Mapping, Set, Sequence

from funmaze.graph import IGraph, Node, graph_neighbours


def solve_dfs_one(graph: IGraph[Node], start: Node, end: Node
                  ) -> Sequence[Node] | None:
    """Find one path on the *graph* from *start* to *end*
    by depth-first-search.

    This is the same as using the backtracking algorithm for generating trees
    to find a solution to the maze, but returning the stack as soon
    as it reaches the end state rather than waiting until it becomes
    empty again.
    """
    neighbours: Mapping[Node, Sequence[Node]] = graph_neighbours(graph)
    stack: list[Node] = [start]
    visited: set[Node] = {start}
    while stack and stack[-1] != end:
        node = stack.pop()
        if good_neighbours := [
                node2 for node2 in neighbours[node]
                if node2 not in visited]:
            stack.append(node)
            node2 = random.choice(good_neighbours)
            visited.add(node2)
            stack.append(node2)
    return stack if stack else None
