from math import comb

import networkx as nx
import numpy as np

from ldpsub.estimators import estimate_3star, local_3star, noiseless_public, run_round2
from ldpsub.exact import count_exact

RANDOM_GRAPHS = [nx.gnp_random_graph(30, p, seed=seed) for seed, p in enumerate([0.1, 0.3, 0.6])]


def true_neighbors(G):
    return [list(G[i]) for i in range(G.number_of_nodes())]


def test_3star_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        public = noiseless_public(G)
        total = sum(local_3star(N_i, public) for N_i in true_neighbors(G))
        assert total == count_exact(G)["3-star"]


def test_3star_round2_is_unbiased():
    G = RANDOM_GRAPHS[1]
    public = noiseless_public(G)
    sensitivity, eps2, runs = comb(public.D, 2), 1.0, 2000
    rng = np.random.default_rng(0)
    estimates = [run_round2(local_3star, true_neighbors(G), public, sensitivity, eps2, rng) for _ in range(runs)]

    laplace_var = 2 * (sensitivity / eps2) ** 2
    standard_error = np.sqrt(G.number_of_nodes() * laplace_var / runs)
    assert abs(np.mean(estimates) - count_exact(G)["3-star"]) < 4 * standard_error


def test_estimate_3star_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = estimate_3star(G, 1e6, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["3-star"]) < 1


def test_estimate_3star_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = estimate_3star(G, 0.5, 1.0, np.random.default_rng(5))
    b = estimate_3star(G, 0.5, 1.0, np.random.default_rng(5))
    assert a == b
