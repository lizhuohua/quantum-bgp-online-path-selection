# The code mainly comes from the tutorial of NetSquid:
# https://docs.netsquid.org/latest-release/tutorial.intro.html

import itertools
import random

import netsquid as ns
import netsquid.qubits.ketstates as ks
import numpy
from netsquid.components import (ClassicalChannel, DepolarNoiseModel,
                                 QuantumChannel)
from netsquid.components.models import FibreDelayModel, FixedDelayModel
from netsquid.components.qsource import QSource, SourceStatus
from netsquid.nodes.connections import Connection, DirectConnection
from netsquid.qubits import StateSampler
from scipy.optimize import curve_fit


def REGRESSION(bounces, mean_bm):

    def exp(x, p, A):
        return A * p**(2 * x)

    popt_AB, _ = curve_fit(exp, bounces, mean_bm, p0=[0.9, 0.5], maxfev=10000)
    return popt_AB


def set_random_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    ns.set_random_state(seed=seed)


def pairwise(iterable):
    """E.g., pairwise([1, 2, 3, 4]) outputs [(1, 2), (2, 3), (3, 4)]
       If input size is less or equal to 1, output []
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def pairs(lst):
    """Iterate over pairs in a list (circular fashion).
    E.g., if `lst=[0, 1, 2, ..., 9]`, the function returns (0, 1) (1, 2) (2, 3) (3, 4) (4, 5) (5, 6) (6, 7) (7, 8) (8, 9) (9, 0).
    Reference: https://stackoverflow.com/questions/1257413/iterate-over-pairs-in-a-list-circular-fashion-in-python
    """
    n = len(lst)
    for i in range(n):
        yield lst[i], lst[(i + 1) % n]


class RoutingPath:
    '''A list of ASs that a routing request should go through.'''

    def __init__(self, as_list):
        self.as_list = as_list

    def __repr__(self):
        return f"{self.as_list}"


class Entanglement:
    """Records the information about a successfully generated entanglement link."""

    def __init__(self, source, destination, source_position, destination_position):
        self.source = source
        self.destination = destination
        self.source_position = source_position
        self.destination_position = destination_position
        self.latency = 0

    def __str__(self):
        return f"{self.source}.{self.source_position} <-> {self.destination}.{self.destination_position}"

    def fidelity(self):
        self.source.qmemory.mem_positions[self.source_position].busy = False
        self.destination.qmemory.mem_positions[self.destination_position].busy = False
        qubit_a, = self.source.qmemory.peek([self.source_position])
        qubit_b, = self.destination.qmemory.peek([self.destination_position])
        return ns.qubits.fidelity([qubit_a, qubit_b], ks.b00, squared=True)

    def consume(self):
        self.source.qmemory.mem_positions[self.source_position].busy = False
        self.destination.qmemory.mem_positions[self.destination_position].busy = False
        self.source.qmemory.pop(self.source_position)
        self.destination.qmemory.pop(self.destination_position)


class ClassicalConnection(DirectConnection):
    """A connection that transmits classical messages, from A to B or B to A."""

    def __init__(self, length, name="ClassicalConnection"):
        a2b = ClassicalChannel("Channel_A2B", length=length, models={"delay_model": FibreDelayModel()})
        b2a = ClassicalChannel("Channel_B2A", length=length, models={"delay_model": FibreDelayModel()})
        super().__init__(name=name, channel_AtoB=a2b, channel_BtoA=b2a)


class QuantumConnection(DirectConnection):
    """A connection that transmits qubits, from A to B or B to A."""

    def __init__(self):
        a2b = QuantumChannel("Channel_A2B", delay=10)
        b2a = QuantumChannel("Channel_B2A", delay=10)
        super().__init__("QuantumConnection", a2b, b2a)


class EntanglingConnection(Connection):
    """A connection that generates entanglement.

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    source_frequency : float
        Frequency with which midpoint entanglement source generates entanglement [Hz].
    name : str, optional
        Name of this connection.

    """

    def __init__(self, length, source_frequency, name="EntanglingConnection"):
        super().__init__(name=name)
        qsource = QSource(f"qsource_{name}",
                          StateSampler([ks.b00], [1.0]),
                          num_ports=2,
                          timing_model=FixedDelayModel(delay=1e9 / source_frequency),
                          status=SourceStatus.INTERNAL)
        self.add_subcomponent(qsource, name="qsource")
        qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2, models={"delay_model": FibreDelayModel()})
        qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2, models={"delay_model": FibreDelayModel()})
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(qchannel_c2b, forward_output=[("B", "recv")])
        # Connect qsource output to quantum channel input:
        qsource.ports["qout0"].connect(qchannel_c2a.ports["send"])
        qsource.ports["qout1"].connect(qchannel_c2b.ports["send"])


class EntanglingConnectionOnDemand(Connection):
    """A connection that generates an entanglement upon receiving a request in port "trigger".

    Consists of a midpoint holding a quantum source that connects to
    outgoing quantum channels.

    The request will be attached as metadata in the output qubits.

    Parameters
    ----------
    length : float
        End to end length of the connection [km].
    depolar_rate : float
        The probability that qubits will depolarize.

    """

    # Static variable used in the name of QSource. This guarantees that all the generated qubits' name are distinct.
    qsource_index = 1

    def __init__(self, length, depolar_rate):
        self.length = length
        name = "EntanglingConnection"
        name = name + str(EntanglingConnectionOnDemand.qsource_index)
        EntanglingConnectionOnDemand.qsource_index += 1
        super().__init__(name=name)
        self.add_ports("trigger")
        qsource = QSource(f"qsource_{name}", StateSampler([ks.b00], [1.0]), num_ports=2, status=SourceStatus.EXTERNAL)
        self.add_subcomponent(qsource, name="qsource")

        # Represent whether this connection is busy.
        # Users should first check whether this connection is busy, then use it.
        self.busy = False

        def store_request_and_trigger_qsource(message, connection):
            connection.busy = True
            connection.request_buffer = message
            connection.subcomponents["qsource"].ports["trigger"].tx_input(message)

        self.ports["trigger"].bind_input_handler(
            lambda message, _connection=self: store_request_and_trigger_qsource(message, _connection))

        # qchannel_c2a = QuantumChannel("qchannel_C2A", length=length / 2, models={"delay_model": FibreDelayModel()})
        # qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2, models={"delay_model": FibreDelayModel()})

        epsilon = 0.9
        rnd = random.uniform(0, 1)
        if rnd < epsilon:
            depolar_rate = random.uniform(max(0, depolar_rate - 0.005), min(1, depolar_rate + 0.005))
        else:
            depolar_rate = min(depolar_rate * random.randint(5, 30) * 0.5, 0.5)

        noise_model = DepolarNoiseModel(depolar_rate, time_independent=True)
        self.qchannel_c2a = QuantumChannel("qchannel_C2A",
                                           length=length / 2,
                                           models={"quantum_noise_model": noise_model})
        self.qchannel_c2b = QuantumChannel("qchannel_C2B", length=length / 2)
        # Add channels and forward quantum channel output to external port output:
        self.add_subcomponent(self.qchannel_c2a, forward_output=[("A", "recv")])
        self.add_subcomponent(self.qchannel_c2b, forward_output=[("B", "recv")])

        # Connect qsource output to quantum channel input:
        def attach_metadata(message, connection, channel):
            message.meta["request"] = connection.request_buffer
            channel.ports["send"].tx_input(message)
            connection.busy = False  # Qubit is sent. Reset busy bit.

        qsource.ports["qout0"].bind_output_handler(lambda message, _connection=self, _channel=self.qchannel_c2a:
                                                   attach_metadata(message, _connection, _channel))
        qsource.ports["qout1"].bind_output_handler(lambda message, _connection=self, _channel=self.qchannel_c2b:
                                                   attach_metadata(message, _connection, _channel))

    def set_depolar_rate(self, rate):
        noise_model = DepolarNoiseModel(rate, time_independent=True)
        self.qchannel_c2a.models["quantum_noise_model"] = noise_model
