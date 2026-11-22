"""Download SNAP datasets into data/.

Usage: python scripts/download_data.py [name ...]   (default: all)
"""

import os
import sys
import urllib.request

DATASETS = {
    "ca-GrQc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "ego-Facebook": "https://snap.stanford.edu/data/facebook_combined.txt.gz",
    "ca-AstroPh": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    "email-Enron": "https://snap.stanford.edu/data/email-Enron.txt.gz",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main(names):
    for name in names:
        path = os.path.join(DATA_DIR, name + ".txt.gz")
        if os.path.exists(path):
            print(f"{name}: already downloaded")
            continue
        print(f"{name}: downloading {DATASETS[name]}")
        urllib.request.urlretrieve(DATASETS[name], path)


if __name__ == "__main__":
    main(sys.argv[1:] or list(DATASETS))
