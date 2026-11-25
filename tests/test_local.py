from itertools import combinations

import networkx as nx
import numpy as np
from scipy import sparse

from ldpsub.local import neighbor_block, pair_sum, triangle_sum, wedge_sum


def adjacency(G):
    return nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()))


def test_small_graphs():
    K3, K4, path = adjacency(nx.complete_graph(3)), adjacency(nx.complete_graph(4)), adjacency(nx.path_graph(3))

    assert pair_sum(K3) == 3 and pair_sum(K4) == 6 and pair_sum(path) == 2
    assert wedge_sum(K3) == 3      # each node is the center of 1 wedge
    assert wedge_sum(K4) == 12     # each node is the center of C(3,2) = 3 wedges
    assert wedge_sum(path) == 1    # only the middle node is a center
    assert triangle_sum(K3) == 1 and triangle_sum(K4) == 4 and triangle_sum(path) == 0


def test_against_plain_loops_on_real_values():
    # Real-valued symmetric block with a nonzero diagonal, like centered noisy bits.
    rng = np.random.default_rng(0)
    d = 9
    B = rng.normal(size=(d, d))
    B = (B + B.T) / 2

    pairs = sum(B[k, l] for k, l in combinations(range(d), 2))
    wedges = sum(
        B[j, k] * B[j, l]
        for j in range(d)
        for k, l in combinations([x for x in range(d) if x != j], 2)
    )
    triangles = sum(B[j, k] * B[j, l] * B[k, l] for j, k, l in combinations(range(d), 3))

    assert np.isclose(pair_sum(B), pairs)
    assert np.isclose(wedge_sum(B), wedges)
    assert np.isclose(triangle_sum(B), triangles)


def test_neighbor_block_dense_and_sparse():
    M = adjacency(nx.gnp_random_graph(20, 0.3, seed=0))
    N = [3, 7, 1, 15]
    expected = np.array([[M[k, l] for l in N] for k in N])
    assert np.array_equal(neighbor_block(M, N), expected)
    assert np.array_equal(neighbor_block(sparse.csr_array(M), N), expected)
