from collections.abc import Iterable, Mapping, Set

from funmaze.graph import IGraph, Node, graph_neighbours


def solve_bfs_shortest(graph: IGraph[Node], start: Node, end: Node
                       ) -> Iterable[Iterable[Node]]:
    """Use a breadth-first-search on the graph to find one shortest path
    between two nodes.
    """
    # bfs returns shortest solutions first
    solution = next(iter(solve_bfs_all(graph, start, end)), None)
    return [solution] if solution is not None else []


def solve_bfs_all(graph: IGraph[Node], start: Node, end: Node
                  ) -> Iterable[Iterable[Node]]:
    """Use a breadth-first-search on the graph to find all paths
    between two nodes.
    """
    neighbours: Mapping[Node, Set[Node]] = graph_neighbours(graph)
    solutions: list[Node] = [[start]]
    while solutions:
        solutions2 = []
        for solution in solutions:
            node = solution[-1]
            if node == end:
                yield solution
            else:
                for node2 in neighbours.get(node, set()):
                    if node2 not in solution:
                        solutions2.append(solution + [node2])
        solutions = solutions2
