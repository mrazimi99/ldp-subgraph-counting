from functools import partial
from math import comb

import networkx as nx
import numpy as np

from ldpsub.estimators import (
    debiased_rows,
    local_3path,
    local_3star,
    local_4clique,
    local_4cycle,
    local_diamond,
    local_tailed_triangle,
    noiseless_public,
    round2_3path,
    round2_3star,
    round2_4clique,
    round2_4cycle,
    round2_diamond,
    round2_tailed_triangle,
    run_round1,
    run_round2,
)
from ldpsub.exact import count_exact
from ldpsub.privacy import bounded_neighbors

RANDOM_GRAPHS = [nx.gnp_random_graph(30, p, seed=seed) for seed, p in enumerate([0.1, 0.3, 0.6])]


def true_neighbors(G):
    return [list(G[i]) for i in range(G.number_of_nodes())]


def noiseless_local_sum(local_value, G):
    """Sum of every node's local value on the true graph (no noise)."""
    public = noiseless_public(G)
    return sum(local_value(i, N_i, public) for i, N_i in enumerate(true_neighbors(G)))


def run_protocol(round2, G, eps0, eps1, eps2, rng, **extra):
    """One full run, as in the experiments: round 1, bounded neighbors, round 2."""
    public = run_round1(G, eps0, eps1, rng)
    neighbors = bounded_neighbors(G, public.D, rng)
    return round2(public, neighbors, eps2, rng, **extra)


# 3-star (no RR: eps1 = None)

def test_3star_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        assert noiseless_local_sum(local_3star, G) == count_exact(G)["3-star"]


def test_3star_round2_is_unbiased():
    G = RANDOM_GRAPHS[1]
    public = noiseless_public(G)
    sensitivity, eps2, runs = comb(public.D, 2), 1.0, 2000
    rng = np.random.default_rng(0)
    estimates = [run_round2(local_3star, true_neighbors(G), public, sensitivity, eps2, rng) for _ in range(runs)]

    laplace_var = 2 * (sensitivity / eps2) ** 2
    standard_error = np.sqrt(G.number_of_nodes() * laplace_var / runs)
    assert abs(np.mean(estimates) - count_exact(G)["3-star"]) < 4 * standard_error


def test_3star_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = run_protocol(round2_3star, G, 1e6, None, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["3-star"]) < 1


def test_3star_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = run_protocol(round2_3star, G, 0.5, None, 1.0, np.random.default_rng(5))
    b = run_protocol(round2_3star, G, 0.5, None, 1.0, np.random.default_rng(5))
    assert a == b


# Tailed triangle

def test_tailed_triangle_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        assert np.isclose(noiseless_local_sum(local_tailed_triangle, G), count_exact(G)["tailed-triangle"])


def test_tailed_triangle_is_unbiased():
    # Large eps0 so D never cuts a neighbor: this tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [run_protocol(round2_tailed_triangle, G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["tailed-triangle"]) < 4 * standard_error


def test_tailed_triangle_huge_eps_is_exact():
    # eps1 = 50: flip probability e^-50, no bit is flipped.
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = run_protocol(round2_tailed_triangle, G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["tailed-triangle"]) < 1


def test_tailed_triangle_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = run_protocol(round2_tailed_triangle, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = run_protocol(round2_tailed_triangle, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b


# 3-path

def test_3path_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        assert np.isclose(noiseless_local_sum(local_3path, G), count_exact(G)["3-path"])


def test_3path_is_unbiased():
    # Large eps0: no degree cut and no clipping of d~_j at 0. This tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [run_protocol(round2_3path, G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["3-path"]) < 4 * standard_error


def test_3path_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = run_protocol(round2_3path, G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["3-path"]) < 1


def test_3path_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = run_protocol(round2_3path, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = run_protocol(round2_3path, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b


# 4-clique

def test_4clique_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        assert np.isclose(noiseless_local_sum(local_4clique, G) / 4, count_exact(G)["4-clique"])


def test_4clique_is_unbiased():
    # Large eps0 so D never cuts a neighbor: this tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [run_protocol(round2_4clique, G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["4-clique"]) < 4 * standard_error


def test_4clique_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = run_protocol(round2_4clique, G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["4-clique"]) < 1


def test_4clique_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = run_protocol(round2_4clique, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = run_protocol(round2_4clique, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b


# Diamond

def test_diamond_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        assert np.isclose(noiseless_local_sum(local_diamond, G) / 2, count_exact(G)["diamond"])


def test_diamond_is_unbiased():
    # Large eps0 so D never cuts a neighbor: this tests the RR and Laplace parts.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [run_protocol(round2_diamond, G, 100.0, 1.0, 1.0, rng) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["diamond"]) < 4 * standard_error


def test_diamond_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = run_protocol(round2_diamond, G, 1e6, 50.0, 1e6, np.random.default_rng(seed))
        assert abs(estimate - count_exact(G)["diamond"]) < 1


def test_diamond_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = run_protocol(round2_diamond, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    b = run_protocol(round2_diamond, G, 0.5, 1.0, 1.0, np.random.default_rng(5))
    assert a == b


# 4-cycle. tau = infinity (noiseless) or very large: clipping adds bias, so these tests do not clip.

def test_4cycle_noiseless_local_sum_is_exact():
    for G in RANDOM_GRAPHS:
        total = noiseless_local_sum(partial(local_4cycle, tau=np.inf), G)
        assert np.isclose(total / 4, count_exact(G)["4-cycle"])


def test_debiased_rows_float32_with_zeros():
    G = RANDOM_GRAPHS[1]
    public = run_round1(G, 1.0, 1.0, np.random.default_rng(0))
    i, N_i = 0, list(G[0])
    R = debiased_rows(i, N_i, public)
    assert R.dtype == np.float32
    assert R.shape == (len(N_i), G.number_of_nodes())
    assert np.all(R[:, i] == 0)                                   # m != i
    assert all(R[row, k] == 0 for row, k in enumerate(N_i))       # m != k


def test_4cycle_rr_part_is_unbiased():
    # Large eps0 (no degree cut), large tau (no clipping), huge eps2 (no Laplace): only RR is random.
    G = RANDOM_GRAPHS[1]
    rng = np.random.default_rng(0)
    estimates = [run_protocol(round2_4cycle, G, 100.0, 1.0, 1e6, rng, tau=1000.0) for _ in range(2000)]
    standard_error = np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(estimates) - count_exact(G)["4-cycle"]) < 4 * standard_error


def test_4cycle_huge_eps_is_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        estimate = run_protocol(round2_4cycle, G, 1e6, 50.0, 1e6, np.random.default_rng(seed), tau=1000.0)
        assert abs(estimate - count_exact(G)["4-cycle"]) < 1


def test_4cycle_same_seed_same_result():
    G = RANDOM_GRAPHS[1]
    a = run_protocol(round2_4cycle, G, 0.5, 1.0, 1.0, np.random.default_rng(5), tau=10.0)
    b = run_protocol(round2_4cycle, G, 0.5, 1.0, 1.0, np.random.default_rng(5), tau=10.0)
    assert a == b


# Shared round 1: one round-1 output feeds the round 2 of all six patterns.

def test_shared_round1_huge_eps_all_patterns_exact():
    for seed, G in enumerate(RANDOM_GRAPHS):
        rng = np.random.default_rng(seed)
        public = run_round1(G, 1e6, 50.0, rng)
        neighbors = bounded_neighbors(G, public.D, rng)
        eps2 = 1e12
        estimates = {
            "3-star": round2_3star(public, neighbors, eps2, rng),
            "3-path": round2_3path(public, neighbors, eps2, rng),
            "tailed-triangle": round2_tailed_triangle(public, neighbors, eps2, rng),
            "4-cycle": round2_4cycle(public, neighbors, eps2, rng, tau=1000.0),
            "diamond": round2_diamond(public, neighbors, eps2, rng),
            "4-clique": round2_4clique(public, neighbors, eps2, rng),
        }
        exact = count_exact(G)
        for pattern, estimate in estimates.items():
            assert abs(estimate - exact[pattern]) < 1, pattern
