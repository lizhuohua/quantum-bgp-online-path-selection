# A Quantum Network Simulator for Quantum Border Gateway Protocols with Online Top-$k$ Path Selection Algorithms

This repository contains the source code associated with the INFOCOM'24 paper titled *Quantum BGP with Online Path Selection via Network Benchmarking*. It includes a quantum network simulator designed for experimenting with our proposed Quantum Border Gateway Protocols (BGP) equipped with online path selection algorithms. Running the code can fully reproduce all the figures and results in the paper.

## Prerequisites

To get started, ensure you have the following packages installed:

[NetSquid](https://netsquid.org/), [NetworkX](https://networkx.org/), [Matplotlib](https://matplotlib.org/)

## Repository Structure

* [network_benchmarking](./network_benchmarking): Implementation of the network benchmarking algorithm.
* [plots](./plots): Scripts to visualize evaluation results and generate figures in the paper.
* [topology_data](./topology_data): Dataset comprising real-world Autonomous System (AS) topologies.
* [components](./components.py): Defines essential network components like ASes and speakers.
* [event_generators](./event_generators.py): Protocols that generate BGP announcements and routing requests
* [import_data](./import_data.py): Utility to import and handle the real-world dataset, including subgraph sampling.
* [packets](./packets.py): Defines classical messages used for triggering various network events.
* [protocols](./protocols.py): Core implementation of our proposed Quantum BGP protocols.
* [utils](./utils.py): A collection of helper functions.

## How to Run

Execute the following command to start the simulation:

```sh
python main.py
```

## License

See [LICENSE](LICENSE)
