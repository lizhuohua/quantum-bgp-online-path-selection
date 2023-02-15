import os
import pickle
import sys
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import netsquid as ns
from components import QuantumNetwork
from cycler import cycler
from event_generators import RequestGenerator
from utils import set_random_seed

plt.rc('font', family='Linux Libertine')  # Use the same font as the ACM template
plt.rc('font', size=20)
default_cycler = (cycler(color=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']) +
                  cycler(marker=['o', 'v', 's', 'x', '*', '+']) + cycler(linestyle=['-', '--', ':', '-.', '--', ':']))
plt.rc('axes', prop_cycle=default_cycler)

root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
output_dir = os.path.join(root_dir, "outputs")


def plot_goodput_vs_path_num_l(topo="random"):
    """
    X axis: Number of requests
    Y axis: Goodput
    Line: Number of paths selected from the routing table
    """

    file_path = os.path.join(output_dir, f"plot_goodput_vs_path_num_l_{topo}_topo.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
    else:
        # Run in parallel
        p = Pool(3)
        # L_list = [0, 1, 2, 3, 4, 5]
        L_list = [0, 2, 4]
        assert 0 in L_list  # Make sure we include baseline case, i.e., without benchmarking
        results = []
        for l_num in L_list:
            results.append(p.apply_async(evaluate, args=(l_num, topo)))
        p.close()
        p.join()
        results = {L_list[i]: r.get() for i, r in enumerate(results)}

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()
    for l_num, (request_num, goodput) in results.items():
        if l_num == 0:
            label = "Without Benchmarking"
        else:
            label = f"Benchmarking with L={l_num}"
        ax.plot(request_num, goodput, linewidth=1.0, label=label)
    ax.set_xlabel('Number of Requests (S-D Pairs)')
    ax.set_ylabel('Goodput (ebits/s)')
    ax.grid(True)
    ax.legend(fontsize=14, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(f"plot_goodput_vs_path_num_l_{topo}_topo.pdf")
    # plt.show()


def evaluate(l_num, topo):
    # good_arm_file_path = os.path.join(output_dir, f"plot_goodput_vs_path_num_l_{topo}_topo_good_arm.pickle")
    seed = 87
    set_random_seed(seed)
    node_num = 60
    ip_num = 10
    capacity = 12
    max_neighbors_num = 5
    arrival_rate = 2e6

    request_num = 100
    # L_list = [3, 4]
    # channel_success_rate = 0.98

    network = QuantumNetwork(channel_noise_rate=0.05)
    if topo == "random":
        network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    elif topo == "real":
        network.initialize_network_real_topology(node_num, ip_num, capacity, max_neighbors_num)
    network.start()

    # Make data
    # Generate random AS-IP pairs, we will use these pairs to do benchmarking
    seed = 88
    set_random_seed(seed)
    request_generator = RequestGenerator(network.as_dict,
                                         network.ip_list,
                                         "Poisson",
                                         arrival_rate,
                                         request_num,
                                         with_benchmark=False,
                                         random_pairs=[],
                                         enable_load_balancing=False,
                                         emit_request=False)
    request_generator.start()
    ns.sim_run()
    as_ip_pairs = request_generator.get_random_pairs()
    assert len(as_ip_pairs) == request_num

    results = []
    if l_num > 0:
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
                                                          loop_bounces, 3, delta, threshold1, threshold2)
            print(results, file=sys.stderr)
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

    # Run simulation and record performance
    x = []
    y = []
    # request_num = 10
    repeat = 1
    network.reset()
    index = 50
    as_ip_pairs *= 2
    request_num *= 2
    while index <= request_num:
        goodput = 0
        # set_random_seed(seed)
        for i in range(repeat):
            network.reset()
            goodput += network.simulate_traffic("Poisson",
                                                arrival_rate,
                                                index,
                                                with_benchmark=True,
                                                random_pairs=as_ip_pairs[0:index],
                                                enable_load_balancing=False)[1]
        goodput /= repeat
        x.append(index)
        y.append(goodput)
        index += 20

    return x, y
