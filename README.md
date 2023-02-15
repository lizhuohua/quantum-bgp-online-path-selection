# A Quantum Network Simulator for Quantum Border Gateway Protocols with Online Top-$k$ Path Selection Algorithms

## Required Packages

[NetSquid](https://netsquid.org/), [NetworkX](https://networkx.org/), [Matplotlib](https://matplotlib.org/)

## Structures

* [network_benchmarking](./network_benchmarking): Network benchmarking algorithm
* [plots](./plots): Draw figures about the evaluation results
* [topology_data](./toplogy_data): Real-world AS topology
* [components](./components.py): Define network components such as ASes and speakers
* [event_generator](./event_generator.py): Protocols that generate BGP announcements and routing requests
* [import_data](./import_data.py): Import real-world dataset and sample subgraphs from it
* [packets](./packets.py): Define classical messages used for triggering events
* [protocols](./protocols.py): Our QBGP Protocols
* [utils](./utils.py): Utilities of the project

## Run

```sh
python main.py
```

## License

See [LICENSE](LICENSE)
