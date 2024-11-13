import os
import pickle
import sys
import numpy as np
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import netsquid as ns
from components import QuantumNetwork
from cycler import cycler
from event_generators import RequestGenerator
from utils import set_random_seed

plt.rc("font", size=20)
default_cycler = (
    cycler(color=["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"])
    + cycler(marker=["o", "v", "s", "x", "*", "+"])
    + cycler(linestyle=["-", "--", ":", "-.", "--", ":"])
)
plt.rc("axes", prop_cycle=default_cycler)


def plot_goodput_vs_load_balancing(topo="random"):
    """
    X axis: Number of requests
    Y axis: Goodput
    Bar: Number of paths `N` selected for load balancing
    """

    plt.rc("font", family="Nimbus Roman")  # Use the same font as the IEEE template
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    figure_dir = os.path.join(output_dir, "figures")
    file_path = os.path.join(output_dir, f"plot_goodput_vs_load_balancing_{topo}_topo.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, "rb") as f:
            results = pickle.load(f)
            print("Computed results:", results)
    else:
        # Run in parallel
        p = Pool(3)
        # False: disable load balancing, 2: share traffic via 2 paths, 4: share traffic via 4 paths
        N_list = [False, 2, 4]
        assert False in N_list  # Make sure we include baseline case, i.e., without load balancing
        results = []
        for load_balancing in N_list:
            results.append(p.apply_async(evaluate, args=(load_balancing, topo)))
        p.close()
        p.join()
        results = {N_list[i]: r.get() for i, r in enumerate(results)}

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    # Plot
    barWidth = 0.35
    # fig = plt.subplots()
    plt.xlabel("Number of Requests (S-D Pairs)")
    plt.ylabel("Goodput (ebits/s)")
    num_requests = results[False][0]
    goodput_load_balancing_disabled = results[False][1]
    goodput_load_balancing_enabled = results[2][1]
    # plt.ticklabel_format(style="sci", scilimits=(-1, 2), axis="y")
    plt.xticks([r + barWidth for r in range(len(num_requests))], [str(i) for i in num_requests])
    if topo == "random":
        plt.ylim([70, 105])
    else:
        plt.ylim([65, 85])
    br1 = np.arange(len(num_requests))
    br2 = [x + barWidth for x in br1]
    colors = plt.cm.plasma(np.linspace(0, 1, 7))
    plt.bar(
        br1,
        goodput_load_balancing_disabled,
        color="#EC7A08",
        width=barWidth,
        hatch="\\",
        edgecolor="white",
        label="Load Balancing Disabled",
        zorder=3,
    )
    plt.bar(
        br2,
        goodput_load_balancing_enabled,
        color="#519DE9",
        width=barWidth,
        hatch="/",
        edgecolor="white",
        label="Load Balancing Enabled",
        zorder=3,
    )
    plt.tight_layout()
    plt.legend(ncol=1, fontsize=14, title_fontsize=18, loc="upper left", frameon=True)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    plt.grid(True)
    filename = os.path.join(figure_dir, f"plot_goodput_vs_load_balancing_{topo}_topo.pdf")
    plt.savefig(filename)
    os.system("pdfcrop" + " " + filename + " " + filename)
    plt.clf()


def evaluate(load_balancing, topo):
    K = 4  # Number of paths we choose as good paths

    if topo == "random":
        seed = 87
    else:
        seed = 90
    set_random_seed(seed)
    node_num = 60
    ip_num = 16
    if topo == "random":
        capacity = 12
    else:
        capacity = 10
    max_neighbors_num = 20
    arrival_rate = 1e6

    request_num = 500
    # channel_success_rate = 0.98

    # network = QuantumNetwork(channel_noise_rate=0.05)
    network = QuantumNetwork(channel_noise_rate=0.05)
    if topo == "random":
        network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    elif topo == "real":
        network.initialize_network_real_topology(node_num, ip_num, capacity, max_neighbors_num)
    network.start()

    # Make data
    # Generate random AS-IP pairs, we will use these pairs to do benchmarking
    seed = 88
    # seed = 87
    set_random_seed(seed)
    request_generator = RequestGenerator(
        network.as_dict,
        network.ip_list,
        "Poisson",
        arrival_rate,
        request_num,
        with_benchmark=False,
        random_pairs=[],
        enable_load_balancing=load_balancing,
        emit_request=False,
    )
    request_generator.start()
    ns.sim_run()
    as_ip_pairs = request_generator.get_random_pairs()
    assert len(as_ip_pairs) == request_num

    # Run network benchmarking to select good paths
    results = []
    init_bounces = list(range(2, 6))
    init_sample_times = {i: 5 for i in init_bounces}
    loop_bounces = list(range(2, 11))
    delta = 0.10
    threshold1 = 0.80
    threshold2 = 0.80

    good_paths = {}  # Key: pair (AS, IP), value: good arm set of this pair

    benchmarked_pair = []
    for selected_as, selected_ip in as_ip_pairs:
        as_ip_pair = (selected_as, selected_ip)
        # Skip pairs that we have already benchmarked
        if as_ip_pair in benchmarked_pair:
            continue

        path_list = network.get_paths(selected_as, selected_ip, max_num=4)

        results = network.online_top_k_path_selection(
            selected_as, path_list, K, init_bounces, init_sample_times, loop_bounces, 3, delta, threshold1, threshold2
        )
        print(results, file=sys.stderr)
        # Record results
        benchmarked_pair.append(as_ip_pair)
        good_paths[as_ip_pair] = results["good_arm_set"]

        print("good_arm_set:", results["good_arm_set"], file=sys.stderr)

    # Sort routing table
    sorted_pairs = []
    for selected_as, selected_ip in as_ip_pairs:
        # We can only sort routing table once, because `good_path` only stores the index of the paths
        # Sorting routing table twice will screw it up
        if (selected_as, selected_ip) in sorted_pairs:
            continue
        # Some AS-IP pairs don't have `good_paths` because it doesn't have more than K paths, so we have to check this
        if (selected_as, selected_ip) in good_paths:
            sequence = good_paths[(selected_as, selected_ip)]
            network.sort_path_list_by_benchmarking(selected_as, selected_ip, sequence)
            sorted_pairs.append((selected_as, selected_ip))

    # Run simulation and record performance
    x = []
    y = []
    # request_num = 10
    repeat = 3
    network.reset()
    index = 100
    # as_ip_pairs *= 2
    # request_num *= 2
    while index <= request_num:
        goodput = 0
        # set_random_seed(seed)
        for i in range(repeat):
            network.reset()
            goodput += network.simulate_traffic(
                "Poisson", arrival_rate, index, with_benchmark=True, random_pairs=as_ip_pairs[0:index], enable_load_balancing=load_balancing
            )[1]
        goodput /= repeat
        x.append(index)
        y.append(goodput)
        index += 100

    return x, y
