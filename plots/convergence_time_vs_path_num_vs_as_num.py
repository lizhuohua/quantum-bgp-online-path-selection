import os
import pickle
import datetime
from multiprocessing.pool import Pool
import numpy as np

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


node_num_list = [30, 40, 50, 60, 70, 80]


def plot_convergence_time_vs_path_num_vs_as_num():
    """
    X axis: the number of ASes
    Y axis: convergence time (the time it takes to converge to a stable state)
    Bar: the maximum number of paths in the routing table
    """

    plt.rc("font", family="Nimbus Roman")  # Use the same font as the IEEE template
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    figure_dir = os.path.join(output_dir, "figures")
    file_path = os.path.join(output_dir, "plot_convergence_time_vs_path_num_vs_as_num.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, "rb") as f:
            results = pickle.load(f)
    else:
        # Run in parallel
        p = Pool(1)
        # node_num_list = [20, 30, 40, 50, 60, 70]
        path_num_list = [2, 3, 4, 5, 6]
        results = []
        for path_num in path_num_list:
            results.append(p.apply_async(evaluate, args=(path_num,)))
        p.close()
        p.join()
        results = {path_num_list[i]: r.get() for i, r in enumerate(results)}
        print(results)
        # results_dict = {}
        # for node_num, neighbor_num, elapsed_time in results:
        #     results_dict.setdefault(node_num, []).append((neighbor_num, elapsed_time))
        # print(results_dict)

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    # Plot
    plt.rc("axes", prop_cycle=default_cycler)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel("Number of ASes")
    ax.set_ylabel("Convergence Time (ms)")
    ax.grid(True)

    barWidth = 1.8
    br1 = [x - 2 * barWidth for x in node_num_list]
    br2 = [x - barWidth for x in node_num_list]
    br3 = node_num_list
    br4 = [x + barWidth for x in node_num_list]
    br5 = [x + 2 * barWidth for x in node_num_list]
    colors = plt.cm.plasma(np.linspace(0, 1, 7))
    plt.bar(
        br1,
        results[2],
        # color="#EC7A08",
        width=barWidth,
        hatch="\\\\",
        edgecolor="white",
        label="2",
        zorder=3,
    )
    plt.bar(
        br2,
        results[3],
        # color="#519DE9",
        width=barWidth,
        hatch="//",
        edgecolor="white",
        label="3",
        zorder=3,
    )

    plt.bar(
        br3,
        results[4],
        # color="#519DE9",
        width=barWidth,
        hatch="--",
        edgecolor="white",
        label="4",
        zorder=3,
    )

    plt.bar(
        br4,
        results[5],
        # color="#519DE9",
        width=barWidth,
        hatch="++",
        edgecolor="white",
        label="5",
        zorder=3,
    )

    plt.bar(
        br5,
        results[6],
        # color="#519DE9",
        width=barWidth,
        hatch="xx",
        edgecolor="white",
        label="6",
        zorder=3,
    )
    # for label, (x, y) in results.items():
    #     ax.plot(x, y, linewidth=2.0, label=str(label))
    ax.legend(title="Maximum Number of Paths\nfor Each Destination", fontsize=20, title_fontsize=20, ncols=3)
    plt.tight_layout()
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    filename = os.path.join(figure_dir, "plot_convergence_time_vs_path_num_vs_as_num.pdf")
    plt.savefig(filename)
    os.system("pdfcrop" + " " + filename + " " + filename)


def evaluate(max_path_num):
    seed = 88
    # seed = 811
    set_random_seed(seed)
    # node_num = 100
    ip_num = 20
    capacity = 10
    # network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    repeat = 20
    results = []
    for node_num in node_num_list:
        elapsed_time = 0
        for i in range(repeat):
            network = QuantumNetwork(channel_noise_rate=0.1, max_path_num=max_path_num)
            network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num=5)

            # start_time = ns.sim_time(ns.MICROSECOND)
            start_time = datetime.datetime.now()
            network.start()
            # elapsed_time = ns.sim_time(ns.MICROSECOND) - start_time
            elapsed_time += (datetime.datetime.now() - start_time).microseconds / 1000
        elapsed_time /= repeat
        results.append(elapsed_time)
    print("Number of nodes:", node_num, "Convergence time:", elapsed_time)

    return results
