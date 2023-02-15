import random as rd
import sys

import netsquid as ns
import numpy as np
from netsquid.protocols import NodeProtocol, Protocol
from netsquid.qubits import QFormalism
from netsquid.qubits import operators as ops
from netsquid.qubits import qubitapi
from netsquid.qubits.cliffords import local_cliffords
from netsquid.qubits.operators import Operator
from netsquid.qubits.qubitapi import create_qubits, measure, operate
from scipy.optimize import curve_fit

sys.path.append('..')
from packets import RoutingRequest

CLIFFORD_OPERATORS = [Operator(op.name, op.arr) for op in local_cliffords]
ns.set_qstate_formalism(QFormalism.DM)


def EXP(x, p, A):
    return A * p**x


def REGRESSION(bounces, mean_bm):
    popt_AB, pcov_AB = curve_fit(EXP, bounces, mean_bm, p0=[0.9, 0.5], maxfev=100000)
    return [np.sqrt(popt_AB[0]), popt_AB[1]]


def GET_FIDELITY(info_qubit, gates):
    [ref_qubit1, ref_qubit2] = create_qubits(2)
    operate(ref_qubit2, ops.X)
    for gate_instr in gates:
        operate(ref_qubit1, gate_instr)
        operate(ref_qubit2, gate_instr)
    fidelity = qubitapi.exp_value(
        info_qubit, ops.Operator("ref", (ns.qubits.reduced_dm(ref_qubit1) - ns.qubits.reduced_dm(ref_qubit2)) / 2))
    # fidelity = fidelity + np.random.normal(0, np.sqrt(((1+ fidelity)*(1 -fidelity))) / np.sqrt(4000))
    return fidelity


def teleport(epr_qubit, info_qubit):
    """Perform teleportation and return two classical bits."""
    operate([epr_qubit, info_qubit], ns.CNOT)
    operate(epr_qubit, ns.H)
    m1, _ = measure(epr_qubit)
    m2, _ = measure(info_qubit)
    return [m1, m2]


def correction(epr_qubit, measurement_results):
    """Perform correction to recover the information qubit."""
    if measurement_results[0]:
        operate(epr_qubit, ns.Z)
    if measurement_results[1]:
        operate(epr_qubit, ns.X)
    return epr_qubit


class MeasurementResult:
    """Classical information sent by the NB protocol.

    Parameters
    ----------
    entanglement : Entanglement
        Contains the information of the entanglement link.
    measurement_results : list
        The two classical bits used for teleportation.
    """

    def __init__(self, entanglement, measurement_results):
        self.entanglement = entanglement
        self.measurement_results = measurement_results


class NBProtocolAlice(NodeProtocol):
    # bounce: bounce number, type: list
    # num_samples: repetition times for each bounce, type: dict bounce: times

    def __init__(self, node, bounce=[], num_samples={}, path=None, target_ip=None):
        super().__init__(node)
        self._parent_protocol = node.parent_network.protocols[node.speaker_id]
        self._bounce_list = None
        self._path = path
        if isinstance(bounce, list):
            self._bounce_list = bounce
        elif isinstance(bounce, int):
            self._bounce_list = [bounce]
        self._num_samples = num_samples
        self._gates = []  # record the clifford operations we used.
        self._data_record = {}
        self._target_ip = target_ip
        self._target_protocol = None
        self.add_signal("ALICE_MEASUREMENT_READY")
        self.add_signal("BOB_MEASUREMENT_READY")
        self.add_signal("ENTANGLEMENT_READY")
        self.add_signal("EPR_READY")
        self.add_signal("EPR_FAILED")

    def set_target_protocol(self, bob_protocol):
        self._target_protocol = bob_protocol

    class RequestEPR(NodeProtocol):

        class RetryProcedure(Protocol):
            """This models retrying to request an EPR pair."""

            def __init__(self, parent_protocol):
                super().__init__()
                self.parent_protocol = parent_protocol
                self.request = RoutingRequest(self.parent_protocol.node,
                                              self.parent_protocol._target_ip,
                                              path=self.parent_protocol._path)

            def run(self):
                # Wait for a while and request a new ERP pair
                yield self.await_timer(1e9)
                self.parent_protocol.node.ports["cconn"].tx_input(self.request)
                print(f"{self.parent_protocol.node} requests an EPR pair (retry), request_id:", self.request.request_id)

        def __init__(self, node, parent_protocol):
            super().__init__(node)
            self._parent_protocol = parent_protocol._parent_protocol
            self._target_ip = parent_protocol._target_ip
            self._path = parent_protocol._path

        def run(self):
            """Request an EPR pair from the network and return the entanglement. Return value is of type `Entanglement`."""
            request = RoutingRequest(self.node, self._target_ip, path=self._path)
            request_id = request.request_id
            self.node.ports["cconn"].tx_input(request)
            print(f"{self.node} requests an EPR pair (first try), request_id:", request_id)

            # Wait for the network to generate EPR pair
            while True:
                event_epr_ready = self.await_signal(self._parent_protocol, signal_label="EPR_READY")
                event_epr_failed = self.await_signal(self._parent_protocol, signal_label="EPR_FAILED")
                expression = yield event_epr_ready | event_epr_failed

                if expression.first_term.value:  # When EPR pair is successfully generated
                    # Since we don't know whether the EPR is generated for our request, try to get it using the request ID
                    entanglement_and_fidelity = self.node.parent_as.successful_links.get(request_id)
                    if entanglement_and_fidelity is not None:
                        # If we are here, meaning that the entanglement is what we requested
                        assert entanglement_and_fidelity[0].source == self.node
                        self.result = entanglement_and_fidelity[0]  # Return value is stored here
                        break
                if expression.second_term.value:  # When a request fails
                    if request_id in self.node.parent_as.failed_links:
                        # Our request has failed, so wait for a while and request a new ERP pair
                        protocol = self.RetryProcedure(self)
                        # We have to update `request_id` here so in the next iteration we can use the new id
                        request_id = protocol.request.request_id
                        protocol.start()

    def run(self):
        print(self._bounce_list)
        for current_bounce in self._bounce_list:
            current_max_sample = self._num_samples[current_bounce]
            if current_bounce not in self._data_record:
                self._data_record[current_bounce] = []
            print("current bounce:", current_bounce, "bounce_list:", self._bounce_list)
            for current_sample in range(current_max_sample):
                # print("current sample:", current_sample)
                self._gates.clear()
                info_qubit = create_qubits(1)[0]
                for _ in range(current_bounce):
                    # clifford operation to info qubit
                    instr = rd.choice(CLIFFORD_OPERATORS)
                    self._gates.append(instr)
                    operate(info_qubit, instr)

                    # Request an ERP pair
                    request_epr_protocol = self.RequestEPR(self.node, self)
                    request_epr_protocol.start()
                    yield self.await_signal(request_epr_protocol)
                    entanglement = request_epr_protocol.result

                    # Extract EPR pair
                    # print(f"At {ns.sim_time()} {self.node} get one EPR pair and starts teleportation")
                    epr_qubit = self.node.qmemory.pop(entanglement.source_position)[0]
                    # print('At', ns.sim_time(), "alice's epr pair", epr_qubit.qstate.qrepr)

                    # Teleport info qubit to Bob using the EPR pair
                    measurement_results = teleport(epr_qubit, info_qubit)
                    msg = MeasurementResult(entanglement, measurement_results)
                    self.send_signal("ALICE_MEASUREMENT_READY", result=msg)

                    # wait for epr pair and bob's results to restore qubit sent by bob
                    request_epr_protocol = self.RequestEPR(self.node, self)
                    request_epr_protocol.start()
                    yield self.await_signal(request_epr_protocol)
                    entanglement = request_epr_protocol.result

                    epr_qubit = self.node.qmemory.pop(entanglement.source_position)[0]
                    self.send_signal("ENTANGLEMENT_READY", result=entanglement)
                    # print('At', ns.sim_time(), "alice's epr pair", epr_qubit.qstate.qrepr)
                    yield self.await_signal(self._target_protocol, "BOB_MEASUREMENT_READY")

                    measurement_results, instrfrombob = self._target_protocol.get_signal_result("BOB_MEASUREMENT_READY")
                    self._gates.append(instrfrombob)
                    info_qubit = correction(epr_qubit, measurement_results)

                fidelity = GET_FIDELITY(info_qubit, self._gates)
                self._data_record[current_bounce].append(fidelity)
            print(f"Finished bounce {current_bounce}")

    def data_processing(self):
        raw_data = self._data_record
        bounces = list(raw_data.keys())
        mean_values = [np.mean(raw_data[key]) for key in bounces]
        print('bounces:', bounces)
        print("mean_values:", mean_values)
        p, A = REGRESSION(bounces, mean_values)
        return [p, A]

    def return_data(self):
        raw_data = self._data_record
        print(raw_data)
        # print()
        bounces = list(raw_data.keys())
        mean_values = [np.mean(raw_data[key]) for key in bounces]

        assert len(mean_values) == len(self._bounce_list)
        # print('bounces:', bounces)
        # print("mean_values:",mean_values)
        return mean_values


class NBProtocolBob(NodeProtocol):

    def __init__(self, node):
        super().__init__(node)
        self.add_signal("ALICE_MEASUREMENT_READY")
        self.add_signal("BOB_MEASUREMENT_READY")
        self.add_signal("ENTANGLEMENT_READY")
        self.add_signal("EPR_READY")
        self.add_signal("EPR_FAILED")

    def set_target_protocol(self, alice_protocol):
        self._target_protocol = alice_protocol

    def run(self):
        while True:
            yield self.await_signal(self._target_protocol, signal_label="ALICE_MEASUREMENT_READY")
            measurement_result = self._target_protocol.get_signal_result("ALICE_MEASUREMENT_READY")
            entanglement = measurement_result.entanglement
            epr_qubit = self.node.qmemory.pop(entanglement.destination_position)[0]
            measurement_results_from_target = measurement_result.measurement_results
            info_qubit = correction(epr_qubit, measurement_results_from_target)

            yield self.await_signal(self._target_protocol, "ENTANGLEMENT_READY")
            entanglement = self._target_protocol.get_signal_result("ENTANGLEMENT_READY")
            epr_qubit = self.node.qmemory.pop(entanglement.destination_position)[0]
            # teleport the qubit to the next node Bob
            instr = rd.choice(CLIFFORD_OPERATORS)
            operate(info_qubit, instr)
            measurement_results = teleport(epr_qubit, info_qubit)
            self.send_signal("BOB_MEASUREMENT_READY", result=[measurement_results, instr])
