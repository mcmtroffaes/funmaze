from collections import deque
from collections.abc import Iterable, Mapping, Sequence

from funmaze.graph import IGraph, Node, graph_neighbours


# https://www.geeksforgeeks.org/print-paths-given-source-destination-using-bfs/
# https://stackoverflow.com/a/64667117/2863746
def solve_bfs_all(graph: IGraph[Node], start: Node, end: Node
                  ) -> Iterable[Sequence[Node]]:
    """Find all paths on the *graph* from *start* to *end*
    by breadth-first-search.

    .. warning:: Implementation consumes a lot of memory.
    """
    # TODO return the graph instead of all paths, this will be much faster
    #      (smaller queue) and consume a lot less memory (no need to store
    #      full paths); can then write another function that iterates over all
    #      paths in a graph from a given node
    neighbours: Mapping[Node, Sequence[Node]] = graph_neighbours(graph)
    queue: deque[list[Node]] = deque()
    queue.append([start])
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == end:
            yield path
        else:
            for node2 in neighbours[node]:
                if node2 not in path:
                    queue.append(path + [node2])


# https://en.wikipedia.org/wiki/Breadth-first_search#Pseudocode
# https://www.baeldung.com/cs/graph-algorithms-bfs-dijkstra
def solve_bfs_one_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> Sequence[Node] | None:
    """Use a breadth-first-search on the graph to find one shortest path
    between two nodes.

    This implementation is almost identical to Dijkstra's algorithm
    where we exploit that the distance along every edge is equal to one
    to make for a slightly faster algorithm.
    The difference is that, unlike Dijkstra, we do not need to use a priority
    queue ordered by distance, and instead can use a simple FIFO queue,
    since this will be ordered by distance automatically (as edges have the
    same distance).
    """
    neighbours: Mapping[Node, Sequence[Node]] = graph_neighbours(graph)
    parent: dict[Node, Node] = {}

    def _backtrack(node3) -> Sequence[Node]:
        path: deque[Node] = deque([node3])
        while (parent_node := parent[node3]) != start:
            node3 = parent_node
            path.appendleft(node3)
        path.appendleft(start)
        return path

    visited: set[Node] = {start}
    queue: deque[Node] = deque()
    queue.append(start)
    while queue:
        node = queue.popleft()
        if node == end:
            return _backtrack(node)
        else:
            for node2 in neighbours[node]:
                if node2 not in visited:
                    visited.add(node2)
                    parent[node2] = node
                    queue.append(node2)
    return None


def solve_bfs_all_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> Iterable[Sequence[Node]]:
    """Use a breadth-first-search on the graph to find all shortest paths
    between two nodes.

    .. warning:: Implementation consumes a lot of memory.
    """
    # TODO return graph instead of set of paths (see TODO in solve_bfs_all)
    # bfs returns shortest solutions first
    solution_iter = iter(solve_bfs_all(graph, start, end))
    solution1 = next(solution_iter, None)
    if solution1 is not None:
        yield solution1
        length = len(solution1)
        while ((solution2 := next(solution_iter, None)) is not None
               and len(solution2) == length):
            yield solution2
