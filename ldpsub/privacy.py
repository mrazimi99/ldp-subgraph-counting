"""Privacy mechanisms for the two-round edge-LDP protocol (Imola, Murakami, Chaudhuri, USENIX Security 2021).

All randomness comes from a numpy Generator `rng`, so a fixed seed gives the same run.
"""

import numpy as np


def laplace(scale, size, rng):
    """Laplace noise with mean 0 and the given scale b (variance 2b^2)."""
    return rng.laplace(0.0, scale, size)


def noisy_degrees(degrees, eps0, rng):
    """Round 1: each node releases d~_i = d_i + Lap(1/eps0). Sensitivity of a degree is 1."""
    degrees = np.asarray(degrees, dtype=float)
    return degrees + laplace(1.0 / eps0, degrees.shape, rng)


def max_degree_bound(noisy_degrees):
    """Public degree bound D = floor(max d~_i), at least 1."""
    return max(1, int(np.floor(np.max(noisy_degrees))))
