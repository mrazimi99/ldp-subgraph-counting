"""Exact (non-private) counts of the six connected 4-node patterns, non-induced."""

from itertools import combinations

PATTERNS = ["3-star", "3-path", "tailed-triangle", "4-cycle", "diamond", "4-clique"]

# On 4 nodes, a connected edge set is identified by its sorted degree sequence.
_DEGREES_TO_PATTERN = {
    (1, 1, 1, 3): "3-star",
    (1, 1, 2, 2): "3-path",
    (1, 2, 2, 3): "tailed-triangle",
    (2, 2, 2, 2): "4-cycle",
    (2, 2, 3, 3): "diamond",
    (3, 3, 3, 3): "4-clique",
}


def count_brute_force(G):
    """Count all six patterns by checking every 4-node set. O(n^4), tiny graphs only.

    For each 4-node set, every subset of its edges that matches a pattern is one
    (non-induced) copy. Edge subsets that are disconnected or miss a node are skipped.
    """
    counts = dict.fromkeys(PATTERNS, 0)
    for nodes in combinations(G.nodes(), 4):
        edges = list(G.subgraph(nodes).edges())
        for k in range(3, len(edges) + 1):
            for subset in combinations(edges, k):
                degree = dict.fromkeys(nodes, 0)
                for u, v in subset:
                    degree[u] += 1
                    degree[v] += 1
                pattern = _DEGREES_TO_PATTERN.get(tuple(sorted(degree.values())))
                if pattern:
                    counts[pattern] += 1
    return counts
