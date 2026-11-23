"""Download SNAP datasets into data/ and print their size.

Usage (from the repo root): python -m scripts.download_data [name ...]   (default: all)
"""

import sys

from ldpsub.datasets import DATASETS, load_dataset


def main(names):
    print(f"{'dataset':<14}{'nodes':>8}{'edges':>9}{'max deg':>9}")
    for name in names:
        G = load_dataset(name)
        max_deg = max(d for _, d in G.degree())
        print(f"{name:<14}{G.number_of_nodes():>8}{G.number_of_edges():>9}{max_deg:>9}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
