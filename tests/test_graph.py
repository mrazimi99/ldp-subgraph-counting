import networkx as nx

from ldpsub.graph import load_edge_list


def test_load_edge_list(tmp_path):
    # Comment line, a duplicate in both directions (1-2, 2-1), and a self-loop (3-3).
    path = tmp_path / "toy.txt"
    path.write_text("# comment\n1 2\n2 1\n2 3\n3 3\n10 1\n")

    G = load_edge_list(str(path))

    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3  # 1-2, 2-3, 10-1
    assert sorted(G.nodes()) == [0, 1, 2, 3]
    assert nx.number_of_selfloops(G) == 0
