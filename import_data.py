import random
from itertools import combinations

# import matplotlib.pyplot as plt
import networkx as nx


def get_subgraph(G, node_num):
    while True:
        for nodes in combinations(G.nodes, node_num):
            if random.uniform(0, 1) < 0.9:  # Add more randomness
                continue
            G_sub = G.subgraph(nodes)  # Create subgraph induced by nodes
            # Check for connectivity
            if nx.is_connected(G_sub):
                return G_sub
        # If we fail to find a subgraph (which is highly unlikely)
        # change random seed and try again
        random.seed(random.randrange(100))


def import_data(node_num=None):
    """Read real-world topology data and return a graph."""
    f = open("topology_data/as20000101.txt", "r")
    lines = f.readlines()
    G = nx.Graph()
    max_node_num = 10000
    print("Reading real-world topology data...")
    count = 0
    for line in lines:
        if (max_node_num is not None) and count >= max_node_num:
            break
        if line.startswith("#"):
            continue
        a, b = line.split()
        a = int(a)
        b = int(b)
        if a == b:
            continue
        print("Add edge: ", a, b)
        G.add_edge(a, b)
        count += 1

    return get_subgraph(G, node_num)
    # Here graph `G` may not be connected, so we only use its largest connected component
    # largest_component = max(nx.connected_components(G), key=len)
    # return G.subgraph(largest_component).copy()


# # For debugging, plot the graph
# G = import_data(10)
# nx.draw(G, with_labels=True)
# plt.show()
