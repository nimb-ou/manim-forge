"""Graph primitives — traversal and shortest paths, recorded step by step.

Seventh in the verified-primitives library. Same rule: the animation computes
its own claims. A scene showing breadth-first search must *run* the search and
animate the order it actually visited, not a hand-picked order that looks
tidy — otherwise the animation teaches a traversal that the algorithm would
never produce.

Every algorithm here returns its full trace, so a scene can replay the real
sequence of events rather than reconstruct a plausible one.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field


@dataclass
class Graph:
    """An undirected weighted graph with positions, for drawing."""
    nodes: dict[str, tuple[float, float]]
    edges: list[tuple[str, str, float]] = field(default_factory=list)

    def neighbours(self, n: str) -> list[tuple[str, float]]:
        out = []
        for a, b, w in self.edges:
            if a == n:
                out.append((b, w))
            elif b == n:
                out.append((a, w))
        return sorted(out)

    def degree(self, n: str) -> int:
        return len(self.neighbours(n))


@dataclass
class Step:
    """One event in a traversal, enough to animate a single beat."""
    node: str
    depth: int
    via: str | None
    frontier: list[str]


def bfs(g: Graph, start: str) -> list[Step]:
    """Breadth-first traversal, recorded.

    The frontier is captured at each step because it is the thing worth
    animating — BFS is visually about the wave of "everything one step away"
    expanding, and a scene that only highlights visited nodes loses that.
    """
    seen = {start}
    depth = {start: 0}
    parent: dict[str, str | None] = {start: None}
    q = deque([start])
    trace: list[Step] = []

    while q:
        node = q.popleft()
        trace.append(Step(node, depth[node], parent[node], list(q)))
        for nxt, _ in g.neighbours(node):
            if nxt not in seen:
                seen.add(nxt)
                depth[nxt] = depth[node] + 1
                parent[nxt] = node
                q.append(nxt)
    return trace


def dijkstra(g: Graph, start: str) -> tuple[dict[str, float], dict[str, str | None]]:
    """Shortest distances and the tree of predecessors."""
    dist = {n: float("inf") for n in g.nodes}
    prev: dict[str, str | None] = {n: None for n in g.nodes}
    dist[start] = 0.0
    pq = [(0.0, start)]
    done = set()

    while pq:
        d, node = heapq.heappop(pq)
        if node in done:
            continue
        done.add(node)
        for nxt, w in g.neighbours(node):
            if d + w < dist[nxt]:
                dist[nxt] = d + w
                prev[nxt] = node
                heapq.heappush(pq, (dist[nxt], nxt))
    return dist, prev


def path_to(prev: dict[str, str | None], target: str) -> list[str]:
    out = [target]
    while prev.get(out[-1]) is not None:
        out.append(prev[out[-1]])
    return list(reversed(out))


def verify_bfs_is_shortest_unweighted(g: Graph, start: str) -> bool:
    """BFS depth equals Dijkstra distance when every edge weighs one.

    The claim a BFS scene usually makes — that it finds shortest paths — is
    only true on unweighted graphs. Checking it against Dijkstra keeps the
    scene from asserting something the picture cannot support.
    """
    unit = Graph(g.nodes, [(a, b, 1.0) for a, b, _ in g.edges])
    dist, _ = dijkstra(unit, start)
    return all(s.depth == dist[s.node] for s in bfs(unit, start))
