import os
import pickle
import random as rd
import sys
from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import numpy
from components import QuantumNetwork
from utils import set_random_seed


def bounces_vs_path_num_k(num, topo):
    """
    X axis: the number of paths
    Y axis: bounces consumed
    Line: naive Nb and online top K selection
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, f"plot_bounces_vs_path_num_k_{topo}_topo.pickle")

    # if os.path.exists(file_path):
    #     print("Pickle data exists, skip simulation and plot the data directly.")
    #     print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
    #     with open(file_path, 'rb') as f:
    #         results = pickle.load(f)
    # else:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results = {}
    k_list = [num]

    print(f"naive nb is starting", file=sys.stderr)
    results_by_nb = naive_evaluate(num, topo)
    print(f"naive nb finished", file=sys.stderr)
    fidelity_by_nb = results_by_nb["fidelity"]
    print("fidelity_by_nb", fidelity_by_nb)
    total_bounces_by_nb = results_by_nb["total_bounces"]

    # bounce_by_nb = [total_bounces_by_nb * i for i in k_list]
    # results = online_evaluate_without_information_gain(10)
    # print(results["fidelity"])
    # # online_evaluate(5)
    # # # # Run in parallel
    # # p = Pool(4)
    # fidelity_by_online = []
    # total_bounces_by_online = []
    # fidelity_by_online_without_information_gain = []
    # total_bounces_by_online_without_information_gain = []
    for l in k_list:
        print(f"{l} is starting for online evaluate", file=sys.stderr)
        temp1 = online_evaluate(l, topo)
        print(temp1, file=sys.stderr)
        print(f"{l} is finished for online evaluate", file=sys.stderr)
        print(f"{l} is starting for online evaluate without information gain", file=sys.stderr)
        temp2 = online_evaluate_without_information_gain(l, topo)
        print(temp2, file=sys.stderr)
        print(f"{l} is finished for online evaluate without information gain", file=sys.stderr)
        results[l] = {
            "total_bounces": [total_bounces_by_nb, temp1["total_bounces"], temp2["total_bounces"]],
            "fidelity": [fidelity_by_nb, temp1["fidelity"], temp2["fidelity"]]
        }

        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)
        else:
            with open(file_path, 'rb') as f:
                previous = pickle.load(f)
            with open(file_path, 'wb') as f:
                previous[l] = results[l]
                pickle.dump(previous, f)


def naive_evaluate(k_num, topo):
    seed = 872
    set_random_seed(seed)
    node_num = 10
    ip_num = 5
    capacity = 50
    max_neighbors_num = 5
    arrival_rate = 2e6
    request_num = 1500

    channel_success_rate = 0.99

    network = QuantumNetwork(channel_noise_rate=1 - channel_success_rate)

    if topo == "random":
        network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    elif topo == "real":
        network.initialize_network_real_topology(node_num, ip_num, capacity, max_neighbors_num)
    # network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)
    network.start()
    print(network.as_dict)
    print(network.ip_list)
    # print(network.print_routing_table())
    results = {}
    while True:
        # selected_as = rd.choice(network.as_dict)
        selected_as = rd.choice(list(network.as_dict.values()))
        selected_ip = rd.choice(network.ip_list)
        # random use 7 real use 8
        path_list = network.get_paths(selected_as, selected_ip, max_num=10)
        if len(path_list) >= 10:
            print(selected_as, selected_ip)
            print(path_list)
            break
    path_list = path_list[0:8]
    print(path_list)
    bounces = list(range(2, 6))
    sample_times = {}
    for i in bounces:
        sample_times[i] = 5
    _bounces = list(range(2, 21))
    results["total_bounces"] = sum(_bounces) * 50 * 8
    fidelity = network.naive_network_benchmarking(selected_as, path_list, bounces, sample_times)
    print(fidelity)
    results["fidelity"] = fidelity
    return results


def online_evaluate(k_num, topo):
    seed = 872
    set_random_seed(seed)
    node_num = 10
    ip_num = 5
    capacity = 50
    max_neighbors_num = 5
    arrival_rate = 2e6
    request_num = 1500

    channel_success_rate = 0.99

    network = QuantumNetwork(channel_noise_rate=1 - channel_success_rate)
    # network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)

    if topo == "random":
        network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    elif topo == "real":
        network.initialize_network_real_topology(node_num, ip_num, capacity, max_neighbors_num)

    network.start()
    print(network.as_dict)
    print(network.ip_list)
    print(network.print_routing_table())

    while True:
        selected_as = rd.choice(list(network.as_dict.values()))
        selected_ip = rd.choice(network.ip_list)

        path_list = network.get_paths(selected_as, selected_ip, max_num=10)
        if len(path_list) >= 10:
            print(selected_as, selected_ip)
            print(path_list)
            break

    path_list = path_list[0:8]

    init_bounces = list(range(2, 6))
    init_sample_times = {}
    for i in init_bounces:
        init_sample_times[i] = 10
    loop_bounces = list(range(2, 21))
    delta = 0.10
    threshold1 = 0.80
    threshold2 = 0.75
    budget = 5000
    results = network.online_top_k_path_selection(selected_as, path_list, k_num, init_bounces, init_sample_times,
                                                  loop_bounces, budget, delta, threshold1, threshold2)
    # print(fidelity2)

    return results


def online_evaluate_without_information_gain(k_num, topo):
    seed = 872
    set_random_seed(seed)
    node_num = 10
    ip_num = 5
    capacity = 50
    max_neighbors_num = 5
    arrival_rate = 2e6
    request_num = 1500

    channel_success_rate = 0.99

    network = QuantumNetwork(channel_noise_rate=1 - channel_success_rate)
    # network.initialize_random_waxman_topology(node_num, ip_num, capacity, max_neighbors_num)

    if topo == "random":
        network.initialize_random_AS_topology(node_num, ip_num, capacity, max_neighbors_num)
    elif topo == "real":
        network.initialize_network_real_topology(node_num, ip_num, capacity, max_neighbors_num)

    network.start()
    print(network.as_dict)
    print(network.ip_list)
    print(network.print_routing_table())

    while True:
        selected_as = rd.choice(list(network.as_dict.values()))
        selected_ip = rd.choice(network.ip_list)

        path_list = network.get_paths(selected_as, selected_ip, max_num=10)
        if len(path_list) >= 10:
            print(selected_as, selected_ip)
            print(path_list)
            break

    path_list = path_list[0:8]

    init_bounces = list(range(2, 6))
    init_sample_times = {}
    for i in init_bounces:
        init_sample_times[i] = 10
    loop_bounces = list(range(2, 11))
    delta = 0.10
    threshold1 = 0.80
    threshold2 = 0.75
    budget = 5000
    results = network.online_top_k_path_selection_without_information_gain(selected_as, path_list, k_num, init_bounces,
                                                                           init_sample_times, loop_bounces, budget,
                                                                           delta, threshold1, threshold2)
    # print(results)

    return results

    # S: AS, D: IP
