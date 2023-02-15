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


def plot_goodput_vs_channel_noise_vs_capacity():
    """
    X axis: the channel success rate (i.e., 1 - channel_noise_rate)
    Y axis: goodput (the sum of fidelity of each successful link / elapsed time)
    Line: the capacity of speakers
    """

    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, "plot_goodput_vs_channel_noise_vs_capacity.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    else:
        # Run in parallel
        p = Pool(4)
        capacity_list = [2, 4, 6, 8, 10]
        results = []
        for capacity in capacity_list:
            results.append(p.apply_async(evaluate, args=(capacity, )))
        p.close()
        p.join()
        results = {capacity_list[i]: r.get() for i, r in enumerate(results)}

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    ax.set_xlabel('Average Channel Noise Rate')
    ax.set_ylabel('Goodput (ebits/s)')
    ax.grid(True)
    for label, (x, y) in results.items():
        ax.plot(x, y, linewidth=1.0, label=str(label))
    ax.legend(title="Capacity", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    plt.savefig("plot_goodput_vs_channel_noise_vs_capacity.pdf")
    # plt.show()


def evaluate(capacity):
    node_num = 80
    ip_num = 10
    # capacity = 10
    max_neighbors_num = 5
    arrival_rate = 2e6
    request_num = 500

    # Make data
    seed = 87
    x = []
    y = []
    channel_noise_rate = 0.0
    repeat = 20
    while channel_noise_rate <= 1.0:
        seed = 872
        set_random_seed(seed)
        network = QuantumNetwork(channel_noise_rate=channel_noise_rate)
        # network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
        network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)
        network.start()

        goodput = 0
        set_random_seed(seed)
        for i in range(repeat):
            network.reset()
            goodput += network.simulate_traffic("Poisson", arrival_rate, request_num, enable_load_balancing=False)[1]
        goodput /= repeat
        x.append(channel_noise_rate)
        y.append(goodput)
        channel_noise_rate += 0.05

    return x, y
