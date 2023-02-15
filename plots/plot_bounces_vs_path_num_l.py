import os
import pickle
import random as rd
import sys
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy
from components import QuantumNetwork
from cycler import cycler
from utils import set_random_seed

plt.rc('font', family='Linux Libertine')  # Use the same font as the ACM template
plt.rc('font', size=19)
default_cycler = (cycler(color=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']) +
                  cycler(marker=['o', 'v', 's', 'x', '+']) + cycler(linestyle=['-', '--', ':', '-.', '--']))
plt.rc('axes', prop_cycle=default_cycler)


def plot_bounces_vs_path_num_l(topo):
    """
    X axis: the number of paths
    Y axis: bounces consumed
    Line: naive Nb and online top K selection
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, f"plot_bounces_vs_path_num_l_{topo}_topo.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            results = pickle.load(f)

    # Plot
    barWidth = 0.25
    fig = plt.subplots()
    plt.rc('axes', prop_cycle=default_cycler)
    plt.xlabel('Number of Paths in Routing Table')
    plt.ylabel('Total Bounces')
    # plt.grid(True)
    len_l = list(results.keys())
    total_bounces_nb = []
    total_bounces_online = []
    total_bounces_without_info = []
    for l in len_l:
        total_bounces_nb.append(results[l]["total_bounces"][0])
        total_bounces_online.append(results[l]["total_bounces"][1])
        total_bounces_without_info.append(results[l]["total_bounces"][2])
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.xticks([r + barWidth for r in range(len(len_l))], [str(i) for i in len_l])
    plt.ylim([0, 1.2e5])
    br1 = numpy.arange(len(len_l))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.bar(br1,
            total_bounces_online,
            color='#e41a1c',
            width=barWidth,
            hatch="\\\\",
            edgecolor='black',
            label='Online Top-K')
    plt.bar(br2,
            total_bounces_without_info,
            color='#377eb8',
            width=barWidth,
            hatch='//',
            edgecolor='black',
            label='Pure Exploration')
    plt.bar(br3,
            total_bounces_nb,
            color='#4daf4a',
            width=barWidth,
            hatch="xx",
            edgecolor='black',
            label='Network Benchmarking')
    plt.tight_layout()

    plt.legend(ncol=2, fontsize=14, title_fontsize=18, loc="upper left", frameon=False)
    #plt.show()
    # print(results["fidelity"][0])
    # print(results["fidelity"][1])
    # print(results["fidelity"][2])
    plt.savefig(f"plot_bounces_vs_path_num_l_{topo}_topo.pdf")
