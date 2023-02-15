from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
from components import QuantumNetwork
from utils import set_random_seed

plt.rc('font', family='Linux Libertine')  # Use the same font as the ACM template


def plot_lantency_vs_length_vs_as_num():
    """
    X axis: the average length (in km) of quantum links between different ASes
    Y axis: average latency (sum of generation time of successful links / number of successful links)
    Line: the number of ASes
    """

    # Run in parallel
    p = Pool(8)
    node_num_list = [20, 30, 40, 50, 60, 70]
    results = []
    for node_num in node_num_list:
        results.append(p.apply_async(evaluate, args=(node_num, )))
    p.close()
    p.join()
    results = {node_num_list[i]: r.get() for i, r in enumerate(results)}

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel('Average Length of Quantum Links (km)')
    ax.set_ylabel('Average Latency (ms)')
    ax.grid(True)
    for label, (x, y) in results.items():
        ax.plot(x, y, linewidth=2.0, label=str(label))
    ax.legend(title="Number of ASes")

    plt.savefig("plot_latency_vs_length_vs_as_num.pdf")
    # plt.show()


def evaluate(node_num):
    seed = 872
    set_random_seed(seed)
    # node_num = 100
    ip_num = 10
    capacity = 10
    max_neighbors_num = 5
    arrival_rate = 2e6
    request_num = 1500

    # Make data
    seed = 87
    x = []
    y = []
    link_length = 1
    repeat = 10
    while link_length < 20:
        network = QuantumNetwork(avg_link_length=link_length)
        # network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
        network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)
        network.start()

        latency = 0
        set_random_seed(seed)
        for i in range(repeat):
            network.reset()
            latency += network.simulate_traffic("Poisson", arrival_rate, request_num, enable_load_balancing=False)[2]
        latency /= repeat
        x.append(link_length)
        y.append(latency / 1e6)  # Convert ns to ms
        link_length += 1

    return x, y
