"""Privacy mechanisms for the two-round edge-LDP protocol (Imola, Murakami, Chaudhuri, USENIX Security 2021).

All randomness comes from a numpy Generator `rng`, so a fixed seed gives the same run.
"""

import networkx as nx
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


def rr_probabilities(eps1):
    """Randomized response: keep a bit with p = e^eps1 / (e^eps1 + 1), flip it with q = 1 - p."""
    p = np.exp(eps1) / (np.exp(eps1) + 1)
    return p, 1 - p


def randomized_response(G, eps1, rng, chunk_rows=1000):
    """Round 1: the public noisy graph Y (n x n, uint8, symmetric, zero diagonal).

    Node i reports its bits a_ij for j < i through RR, so each pair is reported once.
    The server makes Y symmetric. Rows are processed in chunks to limit memory.
    """
    n = G.number_of_nodes()
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), dtype=np.uint8, format="csr")
    _, q = rr_probabilities(eps1)

    Y = np.zeros((n, n), dtype=np.uint8)
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        true_bits = A[start:stop].toarray()
        flips = rng.random((stop - start, n)) < q
        noisy_bits = true_bits ^ flips
        reported = np.arange(n) < np.arange(start, stop)[:, None]  # row i reports columns j < i
        Y[start:stop] = noisy_bits * reported
    return Y + Y.T
