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
import math
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


# --------------------------------------------------------------- colouring


def greedy_colouring(g: Graph, order: list[str] | None = None) -> dict[str, int]:
    """Colour each node with the lowest colour none of its neighbours uses.

    Greedy, and therefore *not* guaranteed optimal -- which is the point of
    the scene that uses it. The result depends on the order nodes are taken
    in, and the caller may supply one to show that dependence.
    """
    colours: dict[str, int] = {}
    for n in (order or sorted(g.nodes)):
        used = {colours[m] for m, _ in g.neighbours(n) if m in colours}
        c = 0
        while c in used:
            c += 1
        colours[n] = c
    return colours


def verify_colouring(g: Graph, colours: dict[str, int]) -> bool:
    """No edge may join two nodes of the same colour."""
    return all(colours[a] != colours[b] for a, b, _ in g.edges if a != b)


def chromatic_number(g: Graph) -> int:
    """The true minimum number of colours, by exhaustive search.

    Exponential, and deliberately so: this is the honest answer against which
    the greedy result is compared, and a scene claiming greedy "sometimes uses
    one more than necessary" needs the necessary number to actually be known.
    Only safe for the small graphs these scenes draw.
    """
    nodes = sorted(g.nodes)
    n = len(nodes)
    for k in range(1, n + 1):
        assignment: dict[str, int] = {}

        def extend(i: int) -> bool:
            if i == n:
                return True
            node = nodes[i]
            for c in range(k):
                if all(assignment.get(m) != c for m, _ in g.neighbours(node)):
                    assignment[node] = c
                    if extend(i + 1):
                        return True
                    del assignment[node]
            return False

        if extend(0):
            return k
    return n


# ------------------------------------------------------------ euler paths


def odd_degree_nodes(g: Graph) -> list[str]:
    """Nodes touched by an odd number of edge-ends.

    Degree counts multiplicities, because two bridges between the same pair of
    banks are two ways across and Konigsberg has exactly that.
    """
    return sorted(n for n in g.nodes if g.degree(n) % 2 == 1)


def euler_path_exists(g: Graph) -> bool:
    """A walk using every edge exactly once exists iff 0 or 2 nodes are odd.

    Every visit to a node uses one edge in and one out, so an odd node can
    only be an end of the walk -- and a walk has two ends.
    """
    return len(odd_degree_nodes(g)) in (0, 2)


#: The seven bridges, as Euler posed them in 1736. N and S are the banks,
#: A is the island, B the far quarter. Four of the seven are doubled pairs.
KONIGSBERG = Graph(
    nodes={"N": (0.0, 1.8), "A": (-1.6, 0.0), "B": (1.9, 0.0), "S": (0.0, -1.8)},
    edges=[("N", "A", 1.0), ("N", "A", 1.0),      # two northern bridges
           ("S", "A", 1.0), ("S", "A", 1.0),      # two southern bridges
           ("N", "B", 1.0), ("S", "B", 1.0),      # one each to the far quarter
           ("A", "B", 1.0)],                      # the island to the quarter
)


def verify_konigsberg_has_no_walk() -> bool:
    """All four land masses are odd, so no such walk exists. Checked, not told."""
    return (len(odd_degree_nodes(KONIGSBERG)) == 4
            and not euler_path_exists(KONIGSBERG))


# --------------------------------------------------------- dijkstra, traced


@dataclass
class Relax:
    """One edge examined by Dijkstra, and what it changed."""
    frm: str
    to: str
    weight: float
    old: float
    new: float
    improved: bool


@dataclass
class Settle:
    """One node fixed for good, with the relaxations that followed."""
    node: str
    dist: float
    relaxations: list[Relax]


def dijkstra_steps(g: Graph, start: str) -> list[Settle]:
    """Dijkstra with its working shown.

    The plain version returns distances; an animation needs the order nodes
    were settled in and which edges failed to improve anything, because the
    edges that change nothing are what make the algorithm look greedy and
    still be correct.
    """
    dist = {n: math.inf for n in g.nodes}
    dist[start] = 0.0
    done: set[str] = set()
    out: list[Settle] = []
    heap = [(0.0, start)]
    while heap:
        d, n = heapq.heappop(heap)
        if n in done:
            continue
        done.add(n)
        relaxations: list[Relax] = []
        for m, w in g.neighbours(n):
            if m in done:
                continue
            old, new = dist[m], d + w
            better = new < old
            relaxations.append(Relax(n, m, w, old, new, better))
            if better:
                dist[m] = new
                heapq.heappush(heap, (new, m))
        out.append(Settle(n, d, relaxations))
    return out


def verify_dijkstra_matches_bruteforce(g: Graph, start: str,
                                       tol: float = 1e-9) -> bool:
    """Dijkstra's answer must equal the cheapest path found by enumeration.

    Greedy algorithms are exactly the kind that look right while being wrong
    on a case the drawing does not happen to contain, so the scene's claim
    that the greedy choice is safe is checked against every simple path.
    """
    best: dict[str, float] = {start: 0.0}

    def walk(node: str, cost: float, seen: frozenset[str]) -> None:
        for m, w in g.neighbours(node):
            if m in seen:
                continue
            c = cost + w
            if c < best.get(m, math.inf):
                best[m] = c
            walk(m, c, seen | {m})

    walk(start, 0.0, frozenset({start}))
    dist, _ = dijkstra(g, start)
    return all(abs(dist[n] - best.get(n, math.inf)) < tol for n in g.nodes)


@dataclass
class WalkStep:
    """One bridge crossed: which edge, and where it took you."""
    edge_index: int
    frm: str
    to: str


def walk_until_stuck(g: Graph, start: str,
                     preference: list[int] | None = None) -> list[WalkStep]:
    """Cross unused edges from ``start`` until no unused edge is incident.

    The walk is *maximal*, not optimal: at each node it takes the first
    unused edge in ``preference`` order (edge-list order by default). Vary the
    preference to get a different attempt.

    This exists because hand-written attempts are not attempts. Three routes
    written by eye for the Konigsberg scene turned out to be one invalid walk
    and two that could have carried on -- the narration said "stranded" and
    the data disagreed. A walk that really runs out has to be walked.
    """
    order = preference if preference is not None else list(range(len(g.edges)))
    used: set[int] = set()
    here = start
    out: list[WalkStep] = []
    while True:
        nxt = None
        for i in order:
            if i in used:
                continue
            a, b, _ = g.edges[i]
            if here == a:
                nxt = (i, b)
                break
            if here == b:
                nxt = (i, a)
                break
        if nxt is None:
            return out
        i, dest = nxt
        used.add(i)
        out.append(WalkStep(i, here, dest))
        here = dest


def verify_walk_strands(g: Graph, walk: list[WalkStep], start: str) -> bool:
    """The walk must be legal, and must genuinely have nowhere left to go."""
    here = start
    used: set[int] = set()
    for s in walk:
        a, b, _ = g.edges[s.edge_index]
        if {here, s.to} != {a, b} or s.frm != here or s.edge_index in used:
            return False
        used.add(s.edge_index)
        here = s.to
    remaining = [i for i in range(len(g.edges)) if i not in used]
    if not remaining:
        return False                      # it finished; that is not stranding
    return not any(here in g.edges[i][:2] for i in remaining)
