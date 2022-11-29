from collections import deque, Counter
from collections.abc import Iterable, Sequence, Mapping, Callable

from funmaze.graph import IGraph, Node, graph_neighbours, Graph


def analyze_bfs_branches(graph: IGraph[Node], start: Node,
                         ) -> Iterable[int]:
    """Find lengths of all branches on the *graph* from *start*
    by breadth-first-search.

    A branch is defined as a non-looping section of the graph with no forks.
    The implementation is meant for trees (i.e. perfect mazes).
    If there are cycles, they will be counted as two branches.
    """
    neighbours: Mapping[Node, Sequence[Node]] = graph_neighbours(graph)
    visited: set[Node] = {start}
    queue: deque[tuple[Node, int]] = deque()  # node & length since last fork
    queue.append((start, 0))
    while queue:
        node, branch_length = queue.popleft()
        neigh = [node2 for node2 in neighbours[node] if node2 not in visited]
        if len(neigh) > 1:
            yield branch_length
            branch_length = 0
        if neigh:
            for node2 in neigh:
                visited.add(node2)
                queue.append((node2, branch_length + 1))
        else:
            yield branch_length


def analyze_bfs_branches_many(
        num_samples: int,
        graph: Graph[Node], start: Node,
        generate: Callable[[IGraph, Node], IGraph]) -> Counter[int]:
    counter: Counter[int] = Counter()
    for _ in range(num_samples):
        maze = generate(graph, start)
        counter.update(analyze_bfs_branches(maze, start))
    return counter
