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
    AS is chosen randomly from `as_list`, the speaker is chosen randomly from this AS, and IP is chosen randomly from `ip_list`.
    Requests will arrive according to `distribution_name` with parameter `arrival_rate`.
    At most `arrival_max` requests will be generated.
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
            # Randomly choose AS, speaker, and IP
            if not self.with_benchmark:
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
                # If load balancing is enabled, distribute traffic using different paths
                if self.enable_load_balancing:
                    paths = random_speaker.routing_table.get_route(random_ip)
                    for path in paths:
                        next_asn = path[0]
                        next_speaker = random_speaker.find_next_speaker_via_asn(next_asn)
                        if next_speaker.qmem_available():
                            # Use path that the first speaker has quantum memory available
                            as_path = [random_as.parent_network.as_dict[asn] for asn in path]
                            packet.path = RoutingPath(as_path)
                            break

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
