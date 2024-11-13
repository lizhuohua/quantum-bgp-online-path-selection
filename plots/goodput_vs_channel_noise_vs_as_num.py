import os
import pickle
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
from components import QuantumNetwork
from cycler import cycler
from utils import set_random_seed

plt.rc("font", size=20)
default_cycler = (
    cycler(color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"])
    + cycler(marker=["o", "v", "s", "x", "*", "+"])
    + cycler(linestyle=["-", "--", ":", "-.", "--", ":"])
)
plt.rc("axes", prop_cycle=default_cycler)


def plot_goodput_vs_channel_noise_vs_as_num():
    """
    X axis: the channel success rate (i.e., 1 - channel_noise_rate)
    Y axis: goodput (the sum of fidelity of each successful link / elapsed time)
    Line: the number of ASes
    """

    plt.rc("font", family="Nimbus Roman")  # Use the same font as the IEEE template
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    figure_dir = os.path.join(output_dir, "figures")
    file_path = os.path.join(output_dir, "plot_goodput_vs_channel_noise_vs_as_num.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, "rb") as f:
            results = pickle.load(f)
    else:
        # Run in parallel
        p = Pool(4)
        node_num_list = [40, 60, 80, 100]
        results = []
        for node_num in node_num_list:
            results.append(p.apply_async(evaluate, args=(node_num,)))
        p.close()
        p.join()
        results = {node_num_list[i]: r.get() for i, r in enumerate(results)}

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    # Plot
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    ax.set_xlabel("Average Channel Noise Rate")
    ax.set_ylabel("Goodput (ebits/s)")
    ax.grid(True)
    for label, (x, y) in results.items():
        ax.plot(x, y, linewidth=2.0, label=str(label))
    ax.legend(title="# of ASes", fontsize=18, title_fontsize=18)
    plt.tight_layout()
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    filename = os.path.join(figure_dir, "plot_goodput_vs_channel_noise_vs_as_num.pdf")
    plt.savefig(filename)
    os.system("pdfcrop" + " " + filename + " " + filename)


def evaluate(node_num):
    # node_num = 100
    ip_num = 10
    capacity = 50
    max_neighbors_num = 5
    arrival_rate = 2e6
    request_num = 100

    # Make data
    x = []
    y = []
    channel_noise_rate = 0.0
    repeat = 1
    while channel_noise_rate <= 1.0:
        seed = 866
        set_random_seed(seed)
        network = QuantumNetwork(channel_noise_rate=channel_noise_rate)
        # network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
        network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)
        network.start()

        goodput = 0
        # set_random_seed(seed)
        for i in range(repeat):
            network.reset()
            goodput += network.simulate_traffic("Poisson", arrival_rate, request_num, enable_load_balancing=False)[1]
        goodput /= repeat
        x.append(channel_noise_rate)
        y.append(goodput)
        channel_noise_rate += 0.05

    return x, y
