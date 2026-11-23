import networkx as nx

from ldpsub.exact import count_brute_force


def test_brute_force_k4():
    # K4 contains: 4 three-stars (one per center), 12 three-paths (4!/2),
    # 12 tailed triangles (4 triangles x 3 ways to attach the tail),
    # 3 four-cycles, 6 diamonds (remove one edge), 1 four-clique.
    assert count_brute_force(nx.complete_graph(4)) == {
        "3-star": 4, "3-path": 12, "tailed-triangle": 12,
        "4-cycle": 3, "diamond": 6, "4-clique": 1,
    }


def test_brute_force_small_patterns():
    zero = dict.fromkeys(["3-star", "3-path", "tailed-triangle", "4-cycle", "diamond", "4-clique"], 0)
    # Star with 3 leaves: one 3-star.
    assert count_brute_force(nx.star_graph(3)) == {**zero, "3-star": 1}
    # Path on 4 nodes: one 3-path.
    assert count_brute_force(nx.path_graph(4)) == {**zero, "3-path": 1}
    # 4-cycle: one 4-cycle, and 4 three-paths (remove any one edge).
    assert count_brute_force(nx.cycle_graph(4)) == {**zero, "4-cycle": 1, "3-path": 4}
    # Triangle plus an isolated node: no connected 4-node pattern.
    G = nx.complete_graph(3)
    G.add_node(3)
    assert count_brute_force(G) == zero
