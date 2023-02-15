import argparse

import plots
from components import QuantumNetwork
from utils import REGRESSION, set_random_seed


def test_simple_network():
    seed = 5
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.start()
    network.simulate_traffic("Poisson", arrival_rate=1e9, request_num=150, enable_load_balancing=False)
    network.print_successful_links()
    network.print_failed_links()


def test_naive_nb():
    seed = 5
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.start()
    network.benchmark(7, 1, "3.3.3.3")


def test_nb_via_given_path():
    seed = 5
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.start()
    # network.print_routing_table()

    # Get 4 paths from AS5 to IP "3.3.3.3"
    as5 = network.as_dict[5]
    path_list = network.get_paths(as5, "3.3.3.3", max_num=4)
    print(path_list)

    bounces = list(range(1, 11))
    sample_times = {}
    for i in bounces:
        sample_times[i] = 10

    mean_bm = network.benchmark_path(as5, path_list[0], bounces, sample_times)
    print(REGRESSION(bounces, mean_bm))


def test_real_topology_network():
    seed = 5
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.initialize_network_real_topology(max_node_num=1000)
    network.start()
    network.print_routing_table()
    network.simulate_traffic("Poisson", arrival_rate=10000, request_num=50)
    network.print_successful_links()
    network.print_failed_links()


def test_random_topology_network():
    seed = 5
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.initialize_random_AS_topology(50)
    network.start()
    network.print_routing_table()
    network.simulate_traffic("Poisson", arrival_rate=100, request_num=8000, enable_load_balancing=True)
    network.print_successful_links()
    network.print_failed_links()


def test_naive_network_benchmarking():
    seed = 55
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.start()
    # network.print_routing_table()

    # Get 4 paths from AS5 to IP "3.3.3.3"
    as5 = network.as_dict[5]
    path_list = network.get_paths(as5, "3.3.3.3", max_num=4)
    bounces = list(range(1, 5))
    sample_times = {}
    for i in bounces:
        sample_times[i] = 10

    fidelity = network.naive_network_benchmarking(as5, path_list, bounces, sample_times)
    print(fidelity)


def test_online_top_k_path_selection():
    seed = 5
    set_random_seed(seed)
    network = QuantumNetwork(channel_noise_rate=0.1)
    network.start()
    # network.print_routing_table()

    # Get 4 paths from AS5 to IP "3.3.3.3"
    as5 = network.as_dict[5]
    path_list = network.get_paths(as5, "3.3.3.3", max_num=4)
    init_bounces = list(range(2, 5))
    init_sample_times = {}
    for i in init_bounces:
        init_sample_times[i] = 10
    loop_bounces = list(range(6, 51))
    delta = 0.05
    K = 1
    fidelity = network.online_top_k_path_selection(as5, path_list, K, init_bounces, init_sample_times, loop_bounces,
                                                   delta)
    print(fidelity)


parser = argparse.ArgumentParser()
parser.add_argument("--num")
parser.add_argument("--topo")
args = parser.parse_args()

if __name__ == '__main__':
    # test_naive_nb()
    # test_multiple_links_nb()
    # test_nb_via_given_path()
    # test_simple_network()
    # test_real_topology_network()
    # test_random_topology_network()
    # test_naive_network_benchmarking()
    # test_online_top_k_path_selection()
    # plots.bounces_vs_path_num_k(1)
    # plots.plot_bounces_vs_path_num_k(args.topo)
    # plots.plot_bounces_vs_path_num_l(args.topo)
    # if args.num != "plot":
    #     plots.bounces_vs_path_num_k(int(args.num), args.topo)
    # else:
    #     plots.plot_bounces_vs_path_num_k(args.topo)
    #     plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_ratio(topo=args.topo)
    # if args.num != "plot":
    #     plots.bounces_vs_path_num_l(int(args.num))
    # else:
    #     plots.plot_bounces_vs_path_num_l()
    # plots.plot_throughput_vs_request_num_vs_as_num()
    # plots.plot_throughput_vs_request_num_vs_capacity()
    # plots.plot_throughput_vs_request_num_vs_path_nnum()
    # plots.plot_goodput_vs_channel_noise_vs_as_num()
    # plots.plot_goodput_vs_measure_noise_vs_as_num()
    # plots.plot_goodput_vs_channel_noise_vs_capacity()
    # plots.plot_goodput_vs_measure_noise_vs_capacity()
    # plots.plot_lantency_vs_length_vs_as_num()

    # plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_ratio(topo=args.topo)
    # plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_l(topo=args.topo)
    plots.plot_goodput_vs_path_num_l(topo=args.topo)
    # plots.plot_goodput_vs_ratio(topo=args.topo)
