"""Exact (non-private) counts of the six connected 4-node patterns, non-induced.

Fast counts use the formulas of ESCAPE (Pinar, Seshadhri, Vishal, WWW 2017).
"""

from itertools import combinations

import networkx as nx
import numpy as np
from scipy import sparse

from ldpsub.local import neighbor_block, triangle_sum

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


def count_exact(G):
    """Fast exact counts with the ESCAPE formulas (nodes must be 0..n-1).

    d_i  = degree of node i
    t_i  = triangles containing node i
    t_ij = triangles containing edge (i, j) = common neighbors of i and j
    T    = total triangles
    W_ij = common neighbors of any pair i, j (edge or not)
    """
    A = nx.to_scipy_sparse_array(G, nodelist=range(G.number_of_nodes()), dtype=np.int64, format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    W = A @ A                                    # W_ij = common neighbors of i and j
    t_edge = W.multiply(A).tocsr()              # entry (i, j) = t_ij for each edge
    t = np.asarray(t_edge.sum(axis=1)).ravel() // 2
    T = t.sum() // 3
    i, j = sparse.triu(A).nonzero()             # each edge once, i < j
    t_ij = np.asarray(t_edge[i, j]).ravel()
    W_pairs = sparse.triu(W, k=1).data          # W_ij for pairs i < j (zeros not stored)

    return {
        "3-star": int((d * (d - 1) * (d - 2) // 6).sum()),     # sum_i C(d_i, 3)
        "3-path": int(((d[i] - 1) * (d[j] - 1)).sum() - 3 * T),  # sum_edges (d_i-1)(d_j-1) - 3T
        "tailed-triangle": int((t * (d - 2)).sum()),           # sum_i t_i (d_i - 2)
        "4-cycle": int((W_pairs * (W_pairs - 1) // 2).sum() // 2),  # 1/2 sum_{i<j} C(W_ij, 2)
        "diamond": int((t_ij * (t_ij - 1) // 2).sum()),        # sum_edges C(t_ij, 2)
        "4-clique": _count_4cliques(G, A),
    }


def _count_4cliques(G, A):
    """1/4 sum_i (triangles inside N(i)): each 4-clique is seen from each of its 4 nodes."""
    total = 0
    for i in G:
        B = neighbor_block(A, list(G[i]))
        total += triangle_sum(B)
    return int(round(total)) // 4
