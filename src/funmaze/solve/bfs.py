from collections.abc import Iterable, Mapping, Set, Sequence

from funmaze.graph import IGraph, Node, graph_neighbours


def solve_bfs_all(graph: IGraph[Node], start: Node, end: Node
                  ) -> Iterable[Sequence[Node]]:
    """Use a breadth-first-search on the graph to find all paths
    between two nodes.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    solutions: list[list[Node]] = [[start]]
    while solutions:
        solutions2: list[list[Node]] = []
        for solution in solutions:
            node = solution[-1]
            if node == end:
                yield solution
            else:
                for node2 in neighbours.get(node, set()):
                    if node2 not in solution:
                        solutions2.append(solution + [node2])
        solutions = solutions2


def solve_bfs_shortest(graph: IGraph[Node], start: Node, end: Node
                       ) -> Iterable[Sequence[Node]]:
    """Use a breadth-first-search on the graph to find one shortest path
    between two nodes.
    """
    # bfs returns shortest solutions first
    solution = next(iter(solve_bfs_all(graph, start, end)), None)
    return [solution] if solution is not None else []


def solve_bfs_all_shortest(graph: IGraph[Node], start: Node, end: Node
                           ) -> Iterable[Sequence[Node]]:
    """Use a breadth-first-search on the graph to find all shortest paths
    between two nodes.
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
