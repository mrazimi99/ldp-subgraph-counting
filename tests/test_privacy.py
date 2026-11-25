import networkx as nx
import numpy as np

from ldpsub.privacy import (
    laplace,
    max_degree_bound,
    noisy_degrees,
    randomized_response,
    rr_probabilities,
)

SAMPLES = 200_000


def test_laplace_mean_and_variance():
    rng = np.random.default_rng(0)
    b = 2.0
    x = laplace(b, SAMPLES, rng)
    assert abs(x.mean()) < 0.05
    assert abs(x.var() - 2 * b**2) / (2 * b**2) < 0.03


def test_noisy_degrees():
    rng = np.random.default_rng(1)
    eps0 = 0.5
    d = np.full(SAMPLES, 10)
    noise = noisy_degrees(d, eps0, rng) - d
    assert abs(noise.mean()) < 0.05
    assert abs(noise.var() - 2 / eps0**2) / (2 / eps0**2) < 0.03


def test_noisy_degrees_same_seed_same_output():
    d = [1, 5, 9]
    a = noisy_degrees(d, 1.0, np.random.default_rng(7))
    b = noisy_degrees(d, 1.0, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_max_degree_bound():
    assert max_degree_bound([3.7, 10.2, 5.0]) == 10
    assert max_degree_bound([3.7, 9.99, 5.0]) == 9
    assert max_degree_bound([-2.0, 0.4]) == 1  # never below 1


def test_rr_probabilities():
    p, q = rr_probabilities(1.0)
    assert np.isclose(p + q, 1)
    assert np.isclose(p / q, np.e)


def test_randomized_response_shape():
    G = nx.gnp_random_graph(50, 0.2, seed=0)
    Y = randomized_response(G, 1.0, np.random.default_rng(0))
    assert Y.shape == (50, 50)
    assert Y.dtype == np.uint8
    assert np.array_equal(Y, Y.T)
    assert np.all(np.diag(Y) == 0)
    assert set(np.unique(Y)) <= {0, 1}


def test_randomized_response_large_eps_is_exact():
    G = nx.gnp_random_graph(50, 0.2, seed=0)
    Y = randomized_response(G, 50.0, np.random.default_rng(0))
    assert np.array_equal(Y, nx.to_numpy_array(G, nodelist=range(50), dtype=np.uint8))


def test_randomized_response_flip_rate():
    n = 400
    G = nx.gnp_random_graph(n, 0.3, seed=0)
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.uint8)
    Y = randomized_response(G, 1.0, np.random.default_rng(0))
    _, q = rr_probabilities(1.0)

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)  # each pair once
    edges = upper & (A == 1)
    non_edges = upper & (A == 0)
    assert abs((Y[edges] == 0).mean() - q) < 0.015      # edges flipped to 0
    assert abs((Y[non_edges] == 1).mean() - q) < 0.015  # non-edges flipped to 1 (noisy edges)
