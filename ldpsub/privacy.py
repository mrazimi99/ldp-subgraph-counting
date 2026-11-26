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
    """Public degree bound D = ceil(max d~_i), at least 1.

    Ceil (not floor): with very small noise, floor would give max d - 1 half the time
    and cut a true neighbor. Rounding is post-processing, so privacy is the same.
    """
    return max(1, int(np.ceil(np.max(noisy_degrees))))


def rr_probabilities(eps1):
    """Randomized response: keep a bit with p = e^eps1 / (e^eps1 + 1), flip it with q = 1 - p."""
    p = np.exp(eps1) / (np.exp(eps1) + 1)
    return p, 1 - p


def randomized_response(G, eps1, rng):
    """Round 1: the public noisy graph Y (n x n, uint8, symmetric, zero diagonal).

    Node i reports its bits a_ij for j < i through RR, so each pair is reported once.
    The server makes Y symmetric.
    """
    n = G.number_of_nodes()
    _, q = rr_probabilities(eps1)
    Y = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        true_bits = np.zeros(i, dtype=np.uint8)
        true_bits[[j for j in G[i] if j < i]] = 1
        flips = rng.random(i) < q
        Y[i, :i] = true_bits ^ flips
    return Y + Y.T


def bounded_neighbors(G, D, rng):
    """Round 2 input: each node's neighbor list, cut to D random neighbors if d_i > D.

    Returns a list where entry i is the array of neighbors node i uses.
    Each node cuts only its own list, so i may drop j while j still keeps i.
    """
    neighbors = []
    for i in range(G.number_of_nodes()):
        N_i = np.array(list(G[i]), dtype=int)
        if len(N_i) > D:
            N_i = rng.choice(N_i, size=D, replace=False)
        neighbors.append(N_i)
    return neighbors
