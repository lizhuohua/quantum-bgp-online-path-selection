import os
import pickle
import sys
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


def plot_average_fidelity_vs_with_or_without_benchmarking_vs_l(topo="random"):
    """
    X axis: Number of paths selected from the routing table
    Y axis: Percentage average fidelity improvement
    Line: Different noise
    """

    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir,
                             f"plot_average_fidelity_vs_with_or_without_benchmarking_vs_l_{topo}_topo.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    else:
        # Run in parallel
        p = Pool(4)
        # noise_list = [0, 0.2, 0.4]
        noise_list = [0.1, 0.15, 0.2, 0.25]
        results = []
        for noise in noise_list:
            results.append(p.apply_async(evaluate, args=(noise, topo)))
        p.close()
        p.join()
        results = {noise_list[i]: r.get() for i, r in enumerate(results)}

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    for noise, (L_list, result) in results.items():
        ax.plot(L_list, result, linewidth=1.0, label=str(noise))
    ax.set_xlabel('Number of Paths in Routing Table')
    ax.set_ylabel('Average Fidelity Improvement (%)')
    ax.grid(True)
    ax.legend(title="Noise", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(f"plot_average_fidelity_vs_with_or_without_benchmarking_vs_l_{topo}_topo.pdf")
    # plt.show()


def evaluate(noise, topo):
    seed = 872
    set_random_seed(seed)
    node_num = 30
    ip_num = 10
    capacity = 60
    max_neighbors_num = 5
    arrival_rate = 2e6

    request_num = 30
    L_list = [1, 2, 3, 4, 5]
    # channel_success_rate = 0.98

    network = QuantumNetwork(channel_noise_rate=noise)
    if topo == "random":
        network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    elif topo == "real":
        network.initialize_network_real_topology(node_num, ip_num, capacity, max_neighbors_num)
    network.start()

    # Make data
    # First, evaluate average fidelity without network benchmarking
    seed = 87
    set_random_seed(seed)
    network.reset()
    throughput, goodput, _, as_ip_pairs = network.simulate_traffic("Poisson",
                                                                   arrival_rate,
                                                                   request_num,
                                                                   with_benchmark=False,
                                                                   enable_load_balancing=False)
    avg_fidelity_before = goodput / throughput

    fidelity_improvements = []
    # Second, benchmark different number of paths, update the routing table, and evaluate average fidelity again
    for l_num in L_list:
        K = 1  # Number of paths we choose as good paths
        init_bounces = list(range(2, 6))
        init_sample_times = {i: 5 for i in init_bounces}
        loop_bounces = list(range(2, 21))
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

            path_list = network.get_paths(selected_as, selected_ip, max_num=l_num)
            # If there are less or equal to K paths in the routing table, no need to benchmark this AS-IP pair
            if len(path_list) <= K:
                continue

            results = network.online_top_k_path_selection(selected_as, path_list, K, init_bounces, init_sample_times,
                                                          loop_bounces, 5, delta, threshold1, threshold2)
            print(results)
            # Record results
            benchmarked_pair.append(as_ip_pair)
            good_paths[as_ip_pair] = results["good_arm_set"]

            print(f"L={l_num}, good_arm_set:", results["good_arm_set"], file=sys.stderr)

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

        # Rerun simulation and compare performance
        network.reset()
        throughput, goodput, _, _ = network.simulate_traffic("Poisson",
                                                             arrival_rate,
                                                             request_num,
                                                             with_benchmark=True,
                                                             random_pairs=as_ip_pairs,
                                                             enable_load_balancing=False)
        average_fidelity_after = goodput / throughput
        print(f"L={l_num}, average fidelity before benchmark: {avg_fidelity_before}", file=sys.stderr)
        print(f"L={l_num}, average fidelity after benchmark: {average_fidelity_after}", file=sys.stderr)
        # Compute fidelity improvement percentage
        fidelity_improvements.append((average_fidelity_after - avg_fidelity_before) / avg_fidelity_before * 100)

    return L_list, fidelity_improvements
