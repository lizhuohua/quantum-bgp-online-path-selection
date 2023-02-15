import os
import pickle
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
from components import QuantumNetwork
from cycler import cycler
from utils import set_random_seed

plt.rc('font', family='Linux Libertine')  # Use the same font as the ACM template
plt.rc('font', size=20)
default_cycler = (cycler(color=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']) +
                  cycler(marker=['o', 'v', 's', 'x', '*', '+']) + cycler(linestyle=['-', '--', ':', '-.', '--', ':']))
plt.rc('axes', prop_cycle=default_cycler)


def plot_throughput_vs_request_num_vs_as_num():
    """
    X axis: the number of requests (S-D pairs)
    Y axis: throughput (the number of successful links / elapsed time)
    Line: the number of ASes
    """

    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, "plot_throughput_vs_request_num_vs_as_num.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    else:
        # Run in parallel
        p = Pool(8)
        node_num_list = [20, 30, 40, 50, 60, 70]
        results = []
        for node_num in node_num_list:
            results.append(p.apply_async(evaluate, args=(node_num, )))
        p.close()
        p.join()
        results = {node_num_list[i]: r.get() for i, r in enumerate(results)}

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Requests (S-D Pairs)')
    ax.set_ylabel('Throughput (ebits/s)')
    ax.grid(True)
    for label, (x, y) in results.items():
        ax.plot(x, y, linewidth=1.0, label=str(label))
    ax.legend(title="# of ASes", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    plt.savefig("plot_throughput_vs_request_num_vs_as_num.pdf")
    # plt.show()


def evaluate(node_num):
    # seed = 85
    seed = 811
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    # node_num = 100
    ip_num = 10
    capacity = 10
    max_neighbors_num = 5
    arrival_rate = 2e6
    # network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)
    network.start()

    # Make data
    seed = 87
    x = []
    y = []
    request_num = 10
    repeat = 20
    while request_num < 150:
        throughput = 0
        # set_random_seed(seed)
        for i in range(repeat):
            network.reset()
            throughput += network.simulate_traffic("Poisson", arrival_rate, request_num, enable_load_balancing=False)[0]
        throughput /= repeat
        x.append(request_num)
        y.append(throughput)
        request_num += 5

    return x, y
