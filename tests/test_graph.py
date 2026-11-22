import networkx as nx

from ldpsub.graph import load_edge_list, sample_subgraph


def test_load_edge_list(tmp_path):
    # Comment line, a duplicate in both directions (1-2, 2-1), and a self-loop (3-3).
    path = tmp_path / "toy.txt"
    path.write_text("# comment\n1 2\n2 1\n2 3\n3 3\n10 1\n")

    G = load_edge_list(str(path))

    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3  # 1-2, 2-3, 10-1
    assert sorted(G.nodes()) == [0, 1, 2, 3]
    assert nx.number_of_selfloops(G) == 0


def test_sample_subgraph():
    G = nx.complete_graph(10)

    H = sample_subgraph(G, 4, seed=0)
    assert H.number_of_nodes() == 4
    assert H.number_of_edges() == 6  # induced subgraph of K10 on 4 nodes is K4
    assert sorted(H.nodes()) == [0, 1, 2, 3]

    # Same seed -> same subgraph.
    assert set(H.edges()) == set(sample_subgraph(G, 4, seed=0).edges())

    # n larger than the graph -> whole graph.
    assert sample_subgraph(G, 100).number_of_nodes() == 10
