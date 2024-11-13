import random

import netsquid as ns
import numpy
from netsquid.protocols import Protocol

from packets import Announcement, RoutingRequest
from utils import RoutingPath


def dist(distribution, dist_rate=None, dist_max=None):
    if distribution == "Poisson":
        return numpy.random.poisson(dist_rate)
    elif distribution == "Exponential":
        return numpy.random.exponential(dist_rate)
    elif distribution == "Uniform":
        return numpy.random.uniform(dist_rate, dist_max)
    elif distribution == "Pareto":
        return numpy.random.pareto(dist_rate)
    elif distribution == "Log-Normal":
        return numpy.random.lognormal(dist_rate)


class AnnouncementGenerator(Protocol):
    '''This models websites being established inside an AS, and announcing their IP.'''

    def __init__(self, source_node, ip_list, distribution_name, arrival_rate, arrival_max):
        super().__init__()
        self.source_node = source_node
        self.source_speaker = list(
            source_node.speakers.values())[0]  # Choose one speaker from the AS, we just choose the first one
        self.ip_list = ip_list
        self.distribution_name = distribution_name
        self.arrival_rate = arrival_rate
        self.arrival_max = arrival_max

    def run(self):
        for ip in self.ip_list:
            yield self.await_timer(
                dist(self.distribution_name, dist_rate=self.arrival_rate, dist_max=self.arrival_max) + 1)

            # After some time (according to the given distribution), a message is announced
            print("{} announces an IP: {} at time {}".format(self.source_node, ip, ns.sim_time()))
            msg = Announcement(ip, [])

            self.source_speaker.ports["cconn"].tx_input(msg)


class RequestGenerator(Protocol):
    '''
    This models some speaker in an AS sending requests to access some IP.
    When `with_benchmarking = False`, AS is chosen randomly from `as_list`, the speaker is chosen randomly from this AS,
    and IP is chosen randomly from `ip_list`. Otherwise, the AS and IP are chosen from the given `random_pairs` list.
    Requests will arrive according to `distribution_name` with parameter `arrival_rate`.
    At most `arrival_max` requests will be generated.

    If `enable_load_balancing` is `n`, then the requests will be distributed among the top `n` different paths
    whose first speaker is not full. Otherwise, if `enable_load_balancing` is `False`, then the requests will always be sent to the
    first path in the routing table.
    '''

    def __init__(
        self,
        as_dict,
        ip_list,
        distribution_name,
        arrival_rate,
        arrival_max,
        with_benchmark=False,
        random_pairs=[],
        enable_load_balancing=False,
        emit_request=True,
    ):
        super().__init__()
        self.as_list = list(as_dict.values())
        self.ip_list = ip_list
        self.distribution_name = distribution_name
        self.arrival_rate = arrival_rate
        self.arrival_max = arrival_max
        self.enable_load_balancing = enable_load_balancing
        self.with_benchmark = with_benchmark
        self.random_pairs = random_pairs
        self.emit_request = emit_request

    def get_random_pairs(self):
        return self.random_pairs

    def run(self):
        # for request_id in range(self.arrival_max):
        count = 0
        while count < self.arrival_max:
            if not self.with_benchmark:
                # Randomly choose AS, speaker, and IP
                random_as = random.choice(self.as_list)
                random_ip = random.choice(self.ip_list)
                first_path = random_as.parent_network.get_paths(random_as, random_ip)[0]
                if first_path.as_list == []:
                    continue
                next_as = first_path.as_list[0]
                random_speaker = random_as.parent_network.get_speaker_to_as(random_as, next_as)
                # random_speaker = random.choice(list(random_as.speakers.values()))
                # random_speaker = list(random_as.speakers.values())[0]
                packet = RoutingRequest(random_speaker, random_ip)
            else:
                # If with benchmarking, choose AS, speaker, and IP from the given `random_pairs` list
                random_as, random_ip = self.random_pairs[count]
                next_as = random_as.parent_network.get_paths(random_as, random_ip)[0].as_list[0]
                random_speaker = random_as.parent_network.get_speaker_to_as(random_as, next_as)
                # random_speaker = list(random_as.speakers.values())[0]
                given_path = random_speaker.routing_table.get_route(random_ip)[0]
                as_path = [random_as.parent_network.as_dict[asn] for asn in given_path]
                given_path = RoutingPath(as_path)
                packet = RoutingRequest(random_speaker, random_ip, given_path)
            if random_speaker.get_next_hop_speaker(packet) is None:
                # If we are here, meaning that the `random_speaker` is requesting an IP inside the same AS
                # I.e., the source node `random_speaker` itself is the destination, so we ignore this
                RoutingRequest.request_id -= 1
                continue
            else:
                # If load balancing is enabled, randomly distribute traffic using different paths whose first speaker is not full
                if self.enable_load_balancing is not False:
                    # The number of paths to share the traffic
                    num_paths = int(self.enable_load_balancing)

                    paths = random_as.parent_network.get_paths(random_as, random_ip)

                    paths_with_qmem = []
                    for path in paths:
                        next_as = path.as_list[0]
                        speaker = random_as.parent_network.get_speaker_to_as(random_as, next_as)
                        if speaker.qmem_available():
                            paths_with_qmem.append(path)

                    paths_with_qmem = paths_with_qmem[:num_paths]

                    # paths = random_speaker.routing_table.get_route(random_ip)
                    # Find `num_paths` paths whose first speaker has quantum memory available

                    # paths_with_qmem = [p for p in paths if random_speaker.find_next_speaker_via_asn(p.as_list[0]).qmem_available()][:num_paths]
                    # Randomly choose a path to share the traffic
                    if len(paths_with_qmem) > 0:
                        selected_path = random.choice(paths_with_qmem)
                        as_path = [as_node for as_node in selected_path.as_list]
                        packet.path = RoutingPath(as_path)

                # Record the request ID in the sender speaker
                # This is used for the sender speaker's consumer protocol to consume the entanglement after it gets "EPR_READY" signal
                random_speaker.request_id_list.append(packet.request_id)

                # After some time (according to the given distribution), a message is announced
                if self.emit_request:
                    yield self.await_timer(
                        dist(self.distribution_name, dist_rate=self.arrival_rate, dist_max=self.arrival_max) + 1)
                    print("{} {} requests to access IP: {}, request_id: {}, at time {}".format(
                        random_as, random_speaker, random_ip, packet.request_id, ns.sim_time()))

                    random_speaker.ports["cconn"].tx_input(packet)

                if not self.with_benchmark:
                    self.random_pairs.append([random_as, random_ip])
                count += 1
