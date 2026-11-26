from math import comb

import networkx as nx
import numpy as np

from ldpsub.estimators import (
    estimate_3path,
    estimate_3star,
    estimate_4clique,
    estimate_tailed_triangle,
    local_3path,
    local_3star,
    local_4clique,
    local_tailed_triangle,
    noiseless_public,
    run_round2,
)
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


# Tailed triangle

def test_tailed_triangle_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        public = noiseless_public(G)
        total = sum(local_tailed_triangle(N_i, public) for N_i in true_neighbors(G))
        assert np.isclose(total, count_exact(G)["tailed-triangle"])


def test_tailed_triangle_is_unbiased():
    # Large eps0 so D never cuts a neighbor: this tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [estimate_tailed_triangle(G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["tailed-triangle"]) < 4 * standard_error


def test_estimate_tailed_triangle_huge_eps_is_exact():
    # eps1 = 50: flip probability e^-50, no bit is flipped.
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = estimate_tailed_triangle(G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["tailed-triangle"]) < 1


def test_estimate_tailed_triangle_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = estimate_tailed_triangle(G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = estimate_tailed_triangle(G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b


# 3-path

def test_3path_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        public = noiseless_public(G)
        total = sum(local_3path(N_i, public) for N_i in true_neighbors(G))
        assert np.isclose(total, count_exact(G)["3-path"])


def test_3path_is_unbiased():
    # Large eps0: no degree cut and no clipping of d~_j at 0. This tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [estimate_3path(G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["3-path"]) < 4 * standard_error


def test_estimate_3path_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = estimate_3path(G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["3-path"]) < 1


def test_estimate_3path_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = estimate_3path(G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = estimate_3path(G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b


# 4-clique

def test_4clique_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        public = noiseless_public(G)
        total = sum(local_4clique(N_i, public) for N_i in true_neighbors(G))
        assert np.isclose(total / 4, count_exact(G)["4-clique"])


def test_4clique_is_unbiased():
    # Large eps0 so D never cuts a neighbor: this tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [estimate_4clique(G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["4-clique"]) < 4 * standard_error


def test_estimate_4clique_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = estimate_4clique(G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["4-clique"]) < 1


def test_estimate_4clique_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = estimate_4clique(G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = estimate_4clique(G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b
