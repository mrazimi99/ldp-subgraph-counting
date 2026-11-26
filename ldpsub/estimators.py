"""Edge-LDP estimators for the six 4-node patterns, two-round protocol of Imola et al. (USENIX Security 2021).

Node i only uses its own neighbor list N_i and the public round-1 data.
Running a local value on noiseless public data (true degrees, Y = A, p = 1, q = 0)
gives the exact local value, so each estimator can be checked against the exact count.
"""

from dataclasses import dataclass
from math import comb
from typing import Optional

import networkx as nx
import numpy as np

from ldpsub.privacy import (
    laplace,
    max_degree_bound,
    noisy_degrees,
    randomized_response,
    reduce_degrees,
    rr_probabilities,
)


@dataclass
class Public:
    """What the server publishes after round 1."""

    d_noisy: np.ndarray       # noisy degrees of all nodes
    D: int                    # degree bound
    Y: Optional[np.ndarray]   # noisy graph, None if the pattern does not use RR
    p: float                  # RR keep probability
    q: float                  # RR flip probability


def degrees(G):
    return np.array([G.degree(i) for i in range(G.number_of_nodes())])


def run_round1(G, eps0, eps1, rng):
    """Nodes send noisy degrees (and RR bits if eps1 is given); the server publishes D and Y."""
    d_noisy = noisy_degrees(degrees(G), eps0, rng)
    D = max_degree_bound(d_noisy)
    if eps1 is None:
        return Public(d_noisy, D, None, 1.0, 0.0)
    p, q = rr_probabilities(eps1)
    return Public(d_noisy, D, randomized_response(G, eps1, rng), p, q)


def noiseless_public(G):
    """Round-1 data without noise: true degrees, D = max degree, Y = A, p = 1, q = 0."""
    d = degrees(G)
    A = nx.to_numpy_array(G, nodelist=range(G.number_of_nodes()), dtype=np.uint8)
    return Public(d.astype(float), int(d.max()), A, 1.0, 0.0)


def run_round2(local_value, neighbors, public, sensitivity, eps2, rng):
    """Each node sends local value + Lap(sensitivity / eps2); the server sums."""
    values = np.array([local_value(N_i, public) for N_i in neighbors], dtype=float)  # nodes
    noisy_values = values + laplace(sensitivity / eps2, len(values), rng)            # nodes
    return noisy_values.sum()                                                        # server


# 3-star: local value C(d_i, 3), sensitivity C(D, 2). Uses only eps0 (for D) and eps2.

def local_3star(N_i, public):
    return comb(len(N_i), 3)


def estimate_3star(G, eps0, eps2, rng):
    public = run_round1(G, eps0, None, rng)
    neighbors = reduce_degrees(G, public.D, rng)
    return run_round2(local_3star, neighbors, public, comb(public.D, 2), eps2, rng)
