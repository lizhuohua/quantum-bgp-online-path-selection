import os
import pickle

# import random as rd
# import sys
# from multiprocessing.pool import Pool

from plots import bounces_vs_path_num_l
import matplotlib.pyplot as plt
import numpy

# from components import QuantumNetwork
from cycler import cycler

# from utils import set_random_seed

plt.rc("font", size=19)
default_cycler = (
    cycler(color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"])
    + cycler(marker=["o", "v", "s", "x", "+"])
    + cycler(linestyle=["-", "--", ":", "-.", "--"])
)
plt.rc("axes", prop_cycle=default_cycler)
plt.rcParams["mathtext.fontset"] = "cm"


def plot_bounces_vs_path_num_l(topo):
    """
    X axis: the number of paths
    Y axis: bounces consumed
    Line: naive Nb and online top K selection
    """
    plt.rc("font", family="Nimbus Roman")  # Use the same font as the IEEE template
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    figure_dir = os.path.join(output_dir, "figures")
    file_path = os.path.join(output_dir, f"plot_bounces_vs_path_num_l_{topo}_topo.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        l_list = [7, 8, 9, 10]

        for l in l_list:
            bounces_vs_path_num_l(l, topo)

    with open(file_path, "rb") as f:
        results = pickle.load(f)

    # Plot
    barWidth = 0.25
    # fig = plt.subplots()
    plt.rc("axes", prop_cycle=default_cycler)
    plt.xlabel("Number of Paths in Routing Table")
    plt.ylabel("Total Bounces")
    plt.grid(True)
    len_l = list(results.keys())
    total_bounces_nb = []
    total_bounces_online = []
    total_bounces_without_info = []
    for l in len_l:
        total_bounces_nb.append(results[l]["total_bounces"][0])
        total_bounces_online.append(results[l]["total_bounces"][1])
        total_bounces_without_info.append(results[l]["total_bounces"][2])
    plt.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    plt.xticks([r + barWidth for r in range(len(len_l))], [str(i) for i in len_l])
    plt.ylim([0, 1.2e5])
    br1 = numpy.arange(len(len_l))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    plt.bar(br1, total_bounces_online, color="#EC7A08", width=barWidth, hatch="\\", edgecolor="white", label="Online Top-$K$", zorder=3)
    plt.bar(
        br2, total_bounces_without_info, color="#519DE9", width=barWidth, hatch="/", edgecolor="white", label="Pure Exploration", zorder=3
    )
    plt.bar(br3, total_bounces_nb, color="#4CB140", width=barWidth, hatch="x", edgecolor="white", label="Network Benchmarking", zorder=3)
    plt.tight_layout()

    plt.legend(ncol=2, fontsize=14, title_fontsize=18, loc="upper left", frameon=False)
    # plt.show()
    # print(results["fidelity"][0])
    # print(results["fidelity"][1])
    # print(results["fidelity"][2])
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    filename = os.path.join(figure_dir, f"plot_bounces_vs_path_num_l_{topo}_topo.pdf")
    plt.savefig(filename)
    os.system("pdfcrop" + " " + filename + " " + filename)
    plt.clf()
