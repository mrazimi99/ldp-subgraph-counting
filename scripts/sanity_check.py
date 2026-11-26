"""Sanity checks of the six estimators on sampled subgraphs of the real datasets (step 5).

1. Huge eps (almost no noise): the estimate equals the exact count.
2. Many runs with only RR noise (huge eps2, no degree cut, no clipping): the mean equals the
   exact count. "gap" = |mean - exact| / standard error; below about 3 is what we expect.

Usage (from the repo root): python -m scripts.sanity_check
"""

import numpy as np

from ldpsub.datasets import load_dataset
from ldpsub.estimators import (
    estimate_3path,
    estimate_3star,
    estimate_4clique,
    estimate_4cycle,
    estimate_diamond,
    estimate_tailed_triangle,
)
from ldpsub.exact import PATTERNS, count_exact

# Sample sizes chosen so every pattern has a nonzero count.
SAMPLE_SIZES = {"ca-GrQc": 1000, "ego-Facebook": 300, "ca-AstroPh": 1000, "email-Enron": 3000}
RUNS = 300
TAU = 1e4  # large: no clipping in the 4-cycle

HUGE_EPS = dict(eps0=1e6, eps1=50.0, eps2=1e12)  # eps1 = 50: flip probability e^-50;
                                                 # eps2 = 1e12 because the 4-cycle sensitivity D * TAU is large
RR_ONLY = dict(eps0=100.0, eps1=1.0, eps2=1e6)  # only RR is random


def run_estimator(pattern, G, eps0, eps1, eps2, rng):
    if pattern == "3-star":
        return estimate_3star(G, eps0, eps2, rng)
    if pattern == "3-path":
        return estimate_3path(G, eps0, eps1, eps2, rng)
    if pattern == "tailed-triangle":
        return estimate_tailed_triangle(G, eps0, eps1, eps2, rng)
    if pattern == "4-cycle":
        return estimate_4cycle(G, eps0, eps1, eps2, TAU, rng)
    if pattern == "diamond":
        return estimate_diamond(G, eps0, eps1, eps2, rng)
    if pattern == "4-clique":
        return estimate_4clique(G, eps0, eps1, eps2, rng)
    raise ValueError(pattern)


def main():
    print(f"{'dataset':<13}{'pattern':<17}{'exact':>10}{'huge eps':>12}{'mean (RR only)':>17}{'gap (SE)':>10}")
    for name, n in SAMPLE_SIZES.items():
        G = load_dataset(name, n=n, seed=0)
        exact = count_exact(G)
        for pattern in PATTERNS:
            rng = np.random.default_rng(0)
            huge_eps_estimate = run_estimator(pattern, G, **HUGE_EPS, rng=rng)
            huge_eps_ok = "OK" if abs(huge_eps_estimate - exact[pattern]) < 1 else "WRONG"

            estimates = [run_estimator(pattern, G, **RR_ONLY, rng=rng) for _ in range(RUNS)]
            mean = np.mean(estimates)
            standard_error = np.std(estimates) / np.sqrt(RUNS)
            gap = abs(mean - exact[pattern]) / standard_error if standard_error > 0 else 0.0

            print(f"{name:<13}{pattern:<17}{exact[pattern]:>10}{huge_eps_ok:>12}{mean:>17.1f}{gap:>10.2f}", flush=True)


if __name__ == "__main__":
    main()
