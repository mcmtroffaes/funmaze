from collections.abc import Iterable, Mapping, Set, Sequence
from queue import SimpleQueue

from funmaze.graph import IGraph, Node, graph_neighbours


# https://www.geeksforgeeks.org/print-paths-given-source-destination-using-bfs/
# https://stackoverflow.com/a/64667117/2863746
def solve_bfs_all(graph: IGraph[Node], start: Node, end: Node
                  ) -> Iterable[Sequence[Node]]:
    """Find all paths on the *graph* from *start* to *end*
    by breadth-first-search.

    .. warning:: Implementation consumes a lot of memory.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    queue: SimpleQueue[list[Node]] = SimpleQueue()
    queue.put([start])
    while not queue.empty():
        path = queue.get()
        node = path[-1]
        if node == end:
            yield path
        else:
            for node2 in neighbours.get(node, set()):
                if node2 not in path:
                    queue.put(path + [node2])


# https://en.wikipedia.org/wiki/Breadth-first_search#Pseudocode
def solve_bfs_one_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> Sequence[Node] | None:
    """Use a breadth-first-search on the graph to find one shortest path
    between two nodes.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    parent: dict[Node, Node] = {}

    def _backtrack(node3) -> Iterable[Node]:
        yield node3
        while (parent_node := parent.get(node3)) is not None:
            node3 = parent_node
            yield node3

    visited: set[Node] = {start}
    queue: SimpleQueue[Node] = SimpleQueue()
    queue.put(start)
    while not queue.empty():
        node = queue.get()
        if node == end:
            return list(reversed(list(_backtrack(node))))
        else:
            for node2 in neighbours.get(node, set()):
                if node2 not in visited:
                    visited.add(node2)
                    parent[node2] = node
                    queue.put(node2)
    return None


def solve_bfs_all_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> Iterable[Sequence[Node]]:
    """Use a breadth-first-search on the graph to find all shortest paths
    between two nodes.

    .. warning:: Implementation consumes a lot of memory.
    """
    # bfs returns shortest solutions first
    solution_iter = iter(solve_bfs_all(graph, start, end))
    solution1 = next(solution_iter, None)
    if solution1 is not None:
        yield solution1
        length = len(solution1)
        while ((solution2 := next(solution_iter, None)) is not None
               and len(solution2) == length):
            yield solution2
