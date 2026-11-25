import numpy as np

from ldpsub.privacy import laplace, max_degree_bound, noisy_degrees

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
