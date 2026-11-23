"""Draw the six 4-node patterns into docs/patterns.png.

Usage (from the repo root): python -m scripts.draw_patterns
"""

import os

import matplotlib.pyplot as plt

SQUARE = [(0, 0), (1, 0), (1, 1), (0, 1)]

# name: (node positions, edges)
PATTERNS = {
    "3-star": ([(0.5, 0.45), (0, 0), (1, 0), (0.5, 1)], [(0, 1), (0, 2), (0, 3)]),
    "3-path": ([(0, 0), (0.33, 1), (0.67, 0), (1, 1)], [(0, 1), (1, 2), (2, 3)]),
    "tailed triangle": ([(0, 0), (0.6, 0), (0.3, 1), (1, 1)], [(0, 1), (1, 2), (2, 0), (1, 3)]),
    "4-cycle": (SQUARE, [(0, 1), (1, 2), (2, 3), (3, 0)]),
    "diamond": (SQUARE, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
    "4-clique": (SQUARE, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]),
}


def main():
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.4), facecolor="white")
    for ax, (name, (pos, edges)) in zip(axes, PATTERNS.items()):
        for u, v in edges:
            ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color="#444444", lw=2, zorder=1)
        ax.scatter(*zip(*pos), s=160, color="#3b6ea8", edgecolor="white", lw=1.5, zorder=2)
        ax.set_title(name, fontsize=12)
        ax.set_xlim(-0.2, 1.2)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect("equal")
        ax.axis("off")
    out = os.path.join(os.path.dirname(__file__), "..", "docs", "patterns.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
