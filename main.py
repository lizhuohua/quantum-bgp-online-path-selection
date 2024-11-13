import plots

if __name__ == "__main__":
    plots.plot_goodput_vs_load_balancing(topo="random")
    plots.plot_goodput_vs_load_balancing(topo="real")

    plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_ratio(topo="random")
    plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_ratio(topo="real")

    plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_l(topo="random")
    plots.plot_average_fidelity_vs_with_or_without_benchmarking_vs_l(topo="real")

    plots.plot_bounces_vs_path_num_k(topo="random")
    plots.plot_bounces_vs_path_num_k(topo="real")

    plots.plot_bounces_vs_path_num_l(topo="random")
    plots.plot_bounces_vs_path_num_l(topo="real")

    plots.plot_goodput_vs_channel_noise_vs_as_num()
    plots.plot_goodput_vs_channel_noise_vs_capacity()

    plots.plot_goodput_vs_path_num_l(topo="random")
    plots.plot_goodput_vs_path_num_l(topo="real")

    plots.plot_goodput_vs_ratio(topo="random")
    plots.plot_goodput_vs_ratio(topo="real")

    plots.plot_goodput_vs_measure_noise_vs_as_num()
    plots.plot_goodput_vs_measure_noise_vs_capacity()

    plots.plot_throughput_vs_request_num_vs_as_num()
    plots.plot_throughput_vs_request_num_vs_capacity()

    # Some other potential plots
    # plots.plot_throughput_vs_request_num_vs_path_num()
    # plots.plot_lantency_vs_length_vs_as_num()
