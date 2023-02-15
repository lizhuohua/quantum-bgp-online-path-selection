import netsquid as ns
from netsquid.protocols import Protocol, NodeProtocol
from netsquid.components import (PhysicalInstruction, QuantumProcessor, QuantumProgram)
from netsquid.components.instructions import (INSTR_MEASURE_BELL, INSTR_X, INSTR_Z)
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits import ketstates as ks

from packets import Announcement, RoutingRequest, RoutingRequestACK, MeasurementResult
from utils import Entanglement


class EntanglementConsumer(NodeProtocol):
    """This models the users consuming entanglements generated in the network."""

    class ConsumeProcedure(NodeProtocol):
        """This models the time delay of consuming an entanglement.
           We cannot consume it immediately, otherwise the network will always be empty.
        """
        def __init__(self, node, entanglement, request_id):
            super().__init__(node)
            self.entanglement = entanglement
            self.request_id = request_id

        def run(self):
            yield self.await_timer(1e9)  # Set a time delay
            self.entanglement.consume()
            self.node.request_id_list.remove(self.request_id)

    def __init__(self, node, parent_protocol):
        super().__init__(node)
        self.add_signal("EPR_READY")
        self.add_signal("EPR_FAILED")
        self._parent_protocol = parent_protocol

    def run(self):
        # Record the request ID that is waiting to be consumed
        # This is to prevent consuming a request multiple times
        request_id_to_remove = []
        while True:
            event_epr_ready = self.await_signal(self._parent_protocol, signal_label="EPR_READY")
            event_epr_failed = self.await_signal(self._parent_protocol, signal_label="EPR_FAILED")
            expression = yield event_epr_ready | event_epr_failed

            if expression.first_term.value:  # When EPR pair is successfully generated
                # Since we don't know whether the EPR is generated for our request, try to get it using the request ID
                count = 0  # For debugging only
                for request_id, (entanglement, fidelity) in self.node.parent_as.successful_links.items():
                    if request_id in self.node.request_id_list and request_id not in request_id_to_remove:
                        request_id_to_remove.append(request_id)
                        count += 1
                        # assert count == 1  # This makes sure that we receive every signal
                        # We must use another protocol to consume it because if we set a `await_timer` here,
                        # the execution stops here and thus some signals will be lost
                        consume_procedure = self.ConsumeProcedure(self.node, entanglement, request_id)
                        consume_procedure.start()

            if expression.second_term.value:  # When request is failed
                for request_id in self.node.parent_as.failed_links:
                    if request_id in self.node.request_id_list:
                        self.node.request_id_list.remove(request_id)


class BGPProtocol(NodeProtocol):

    class EntanglementInternal(Protocol):
        """This models the time delay of generating an entanglement within an AS."""

        def __init__(self, speaker1, speaker2, trigger_msg, diameter):
            super().__init__()
            self.speaker1 = speaker1
            self.speaker2 = speaker2
            self.trigger_msg = trigger_msg
            self.diameter = diameter

        def run(self):
            phantum_qconn = self.speaker1.speaker_to_internal_qconn[self.speaker2]
            print(f"Generating entanglement INTERNALLY between {self.speaker1} and {self.speaker2} for request {self.trigger_msg.request_id}")
            yield self.await_timer(self.diameter * 5000)  # Set a time delay

            while phantum_qconn.busy:
                print("Quantum connection is busy, try again...")
                yield self.await_timer(5)
            phantum_qconn.ports["trigger"].tx_input(self.trigger_msg)

    class EntanglementExternal(Protocol):
        """This models the time delay of generating an entanglement across an AS."""

        def __init__(self, from_speaker, to_speaker, trigger_msg):
            super().__init__()
            self.from_speaker = from_speaker
            self.to_speaker = to_speaker
            self.trigger_msg = trigger_msg

        def run(self):
            qconn = self.from_speaker.speaker_to_connection[self.to_speaker]
            print(f"Generating entanglement EXTERNALLY between {self.from_speaker} and {self.to_speaker} for request {self.trigger_msg.request_id}")
            yield self.await_timer(qconn.length * 5000)  # Set a time delay

            while qconn.busy:
                print("Quantum connection is busy, try again...")
                yield self.await_timer(5)
            qconn.ports["trigger"].tx_input(self.trigger_msg)

    def __init__(self, node):
        super().__init__(node)

        self.add_signal("EPR_READY")
        self.add_signal("EPR_FAILED")
        swap_protocol = SwapProtocol(node, "SwapProtocol")
        correct_protocol = CorrectProtocol(node, 4)
        consumer_protocol = EntanglementConsumer(node, self)
        self.add_subprotocol(swap_protocol)
        self.add_subprotocol(correct_protocol)
        self.add_subprotocol(consumer_protocol)

    def entangle_with_i_neighbor(speaker1, speaker2, trigger_msg, diameter):
        """
        Simulate the intra-domain routing protocol that generates an entanglement.
        The entangled qubits will be sent to port "internal_port" of the two speakers.
        `diameter` specifies the size of the AS. It is used to simulate the delay and decoherence.
        """
        # print(f"Generating entanglement INTERNALLY between {speaker1} and {speaker2} for request {trigger_msg.request_id}")

        # phantum_qconn = speaker1.speaker_to_internal_qconn[speaker2]
        # phantum_qconn.ports["trigger"].tx_input(trigger_msg)
        protocol = BGPProtocol.EntanglementInternal(speaker1, speaker2, trigger_msg, diameter)
        protocol.start()

    def entangle_with_e_neighbor(from_speaker, to_speaker, trigger_msg):
        """
        Trigger the entangling connection between two speakers so that an entanglement pair is distributed.
        """
        # print(f"Generating entanglement EXTERNALLY between {from_speaker} and {to_speaker} for request {trigger_msg.request_id}")
        # qconn = from_speaker.speaker_to_connection[to_speaker]
        # qconn.ports["trigger"].tx_input(trigger_msg)
        protocol = BGPProtocol.EntanglementExternal(from_speaker, to_speaker, trigger_msg)
        protocol.start()

    def get_next_speaker(self, msg):
        if msg.path is None:
            # If the request doesn't enforce the path it should use, find a route according to the routing algorithm
            next_speaker = self.node.get_next_hop_speaker(msg)
        else:
            # Else, find the next speaker according to the path that the request specifies
            as_list = msg.path.as_list
            if self.node.parent_as in as_list:
                index = as_list.index(self.node.parent_as)
                if index + 1 < len(as_list):
                    next_as = as_list[index + 1]
                else:
                    next_as = None
            else:
                next_as = as_list[0]

            if next_as is not None:
                next_speaker = self.node.find_next_speaker_via_asn(next_as.asn)
            else:
                next_speaker = None
        return next_speaker

    def handle_routing_request(self, msg):
        # Store the request
        self.node.current_requests[msg.request_id] = msg
        next_speaker = self.get_next_speaker(msg)
        print(f"{self.node} stores request {msg.request_id}, next hop speaker: {next_speaker}")
        if next_speaker is not None:
            if self.node.qmem_available():
                # Check whether we have buffered measurement results for this request waiting to be forwarded
                measurement_msg_list = self.node.measurement_buffer.get(msg.request_id)
                if measurement_msg_list is not None:
                    # Forward the buffered messages
                    for measure_msg in measurement_msg_list:
                        print(f"{self.node} forwards measurement result to {next_speaker}")
                        next_speaker.ports["cconn"].tx_input(measure_msg)
                    # Cleanup the buffer
                    del self.node.measurement_buffer[msg.request_id]

                # Record the next hop speaker used for the request
                # Later when we need to forward measurement results we can know which direction we need to forward
                self.node.request_to_next_hop[msg.request_id] = next_speaker
                # Add the current speaker in the request message
                msg.visited_speakers.append(self.node)

                if next_speaker.asn == self.node.asn:
                    BGPProtocol.entangle_with_i_neighbor(self.node, next_speaker, msg, 50)
                else:
                    BGPProtocol.entangle_with_e_neighbor(self.node, next_speaker, msg)
            else:
                # If there isn't enough capacity, report failure
                ack_msg = RoutingRequestACK(msg.request_id, self.node, None, False)
                self.node.ports["cconn"].tx_input(ack_msg)
        else:
            # If cannot get the next hop speaker, meaning that the current speaker is the destination
            # Check whether we have buffered measurement results for this request waiting to be corrected
            measurement_msg_list = self.node.measurement_buffer.get(msg.request_id)
            if measurement_msg_list is not None:
                for measurement_msg in measurement_msg_list:
                    # self.node.ports["correction_trigger"].tx_input(measurement_msg)
                    protocol = self.TriggerCorrectionProcedure(self.node, measurement_msg)
                    protocol.start()

    def handle_announcement(self, msg):
        print("{} in AS{} receives msg: IP: {}, path: {} at time {}".format(
            self.node, self.node.asn, msg.ip, msg.path, ns.sim_time()))

        # To guarantee fast convergence of BGP announcement propagation, we only store at most `max_path_num` paths for each IP
        if len(self.node.routing_table.table.get(msg.ip, [])) >= self.node.parent_network.max_path_num:
            return

        # If the routing table doesn't have this route, meaning that this is an update, so propagate it to internal neighbors
        if not self.node.routing_table.has_route(msg.ip, msg.path):
            for speaker in self.node.i_neighbors:
                new_msg = Announcement(msg.ip, msg.path)
                speaker.ports["cconn"].tx_input(new_msg)
                print("{} in AS{} forwards msg: IP: {}, path: {} to Speaker{} in AS{} via iBGP at time {}".format(
                    self.node, self.node.asn, new_msg.ip, new_msg.path, speaker.speaker_id, speaker.asn, ns.sim_time()))
            self.node.routing_table.add_route(msg.ip, msg.path)

        for speaker in self.node.e_neighbors:
            if speaker.asn not in msg.path:
                new_msg = Announcement(msg.ip, [self.node.asn] + msg.path)
                speaker.ports["cconn"].tx_input(new_msg)
                print("{} in AS{} forwards msg: IP: {}, path: {} to Speaker{} in AS{} via eBGP at time {}".format(
                    self.node, self.node.asn, new_msg.ip, new_msg.path, speaker.speaker_id, speaker.asn, ns.sim_time()))

    def handle_measurement_result(self, msg):
        request_msg = self.node.current_requests.get(msg.request_id)
        if request_msg is None:
            print(f"Cannot get request in {self.node} using ID {msg.request_id}, buffer it")
            # If the request is not found, meaning that the measurement results arrive earlier than the request.
            # This is possible because measurement results are send via classical channel while requests are attached with qubits,
            # but qubit generation and transmission is slower than classical channels.
            # In this case we buffer the measurement results.
            if msg.request_id not in self.node.measurement_buffer:
                self.node.measurement_buffer[msg.request_id] = [msg]
            else:
                self.node.measurement_buffer[msg.request_id].append(msg)
        else:
            print(f"Get request in {self.node} using ID {msg.request_id}")
            # If the request is found, forward the measurement results to the next hop speaker
            next_speaker = self.node.request_to_next_hop.get(msg.request_id)
            if next_speaker is None:
                # The current speaker is the destination
                print(f"{self.node} is the destination, measurement result: ", msg.measurement_result)

                protocol = self.TriggerCorrectionProcedure(self.node, msg)
                protocol.start()
                # self.node.ports["correction_trigger"].tx_input(msg)
            else:
                # The current speaker is a repeater, forward the measurement results
                next_speaker = self.node.request_to_next_hop[msg.request_id]
                next_speaker.ports["cconn"].tx_input(msg)

    class TriggerCorrectionProcedure(Protocol):
        """This models the time delay of triggering correction procedure.
        We need a small delay here, otherwise, correction procedure may start before quantum memory receiving an input.
        """
        def __init__(self, node, msg):
            super().__init__()
            self.node = node
            self.msg = msg

        def run(self):
            yield self.await_timer(5)  # Set a time delay
            self.node.ports["correction_trigger"].tx_input(self.msg)

    def handle_routing_request_ack(self, msg):
        if msg.success:
            request_msg = self.node.current_requests.get(msg.request_id)
            source = request_msg.source_speaker
            destination = msg.destination_speaker

            # Store the entanglement information
            if source.qmemory.request_to_position.get(msg.request_id) is not None:
                assert len(source.qmemory.request_to_position[msg.request_id]) == 1
                source_position = source.qmemory.request_to_position[msg.request_id][0]
                destination_position = msg.destination_position
                entanglement = Entanglement(source, destination, source_position, destination_position)

                entanglement.latency = ns.sim_time() - request_msg.start_time
                print("Successful entanglement: ", entanglement, "with fidelity", entanglement.fidelity(), "for request", msg.request_id, "Latency:", entanglement.latency)
                self.node.parent_as.successful_links[msg.request_id] = (entanglement, entanglement.fidelity())

                # Send signal to other protocols on the same speaker (e.g., NB protocols) indicating an EPR pair is ready
                self.send_signal("EPR_READY")
        else:
            request_msg = self.node.current_requests.get(msg.request_id)
            if self.node == request_msg.source_speaker:
                self.node.parent_as.failed_links[msg.request_id] = f"{msg.destination_speaker} is full."
                # Send signal to other protocols on the same speaker (e.g., NB protocols) indicating that the request has failed
                self.send_signal("EPR_FAILED")
            if len(request_msg.visited_speakers) > 0:
                last_speaker = self.get_last_speaker(msg.request_id)
                if last_speaker is not None:
                    print("Current speaker:", self.node, "Last speaker:", last_speaker)
                    last_speaker.ports["cconn"].tx_input(msg)

    def get_last_speaker(self, request_id):
        request_msg = self.node.current_requests.get(request_id)
        visited_speakers = request_msg.visited_speakers
        if self.node in visited_speakers:
            index = visited_speakers.index(self.node) - 1
            if index >= 0:
                return visited_speakers[index]
            else:
                return None
        else:
            return visited_speakers[-1]

    def cleanup_request(self, request_id):
        print(f"{self.node} cleans up request {request_id}")
        if request_id in self.node.request_to_next_hop:
            del self.node.request_to_next_hop[request_id]

        if request_id in self.node.qmemory.request_to_position:
            positions = self.node.qmemory.request_to_position[request_id]
            # self.node.qmemory.discard(positions, check_positions=False)
            for p in positions:
                if p in self.node.qmemory.used_positions:
                    print(f"Discarding qubit from {self.node} position {p}")
                    self.node.qmemory.mem_positions[p].busy = False
                    self.node.qmemory.discard(p)
                    print("Finish discard")
            del self.node.qmemory.request_to_position[request_id]

    def run(self):
        classical_port = self.node.ports["cconn"]
        while True:
            qevexpr = None
            for index in range(self.node.internal_qport_num):
                qport = self.node.ports[f"internal_qport{index+1}"]
                if qevexpr is None:
                    qevexpr = self.await_port_input(qport)
                else:
                    qevexpr |= self.await_port_input(qport)

            for index in range(self.node.qconn_num):
                qport = self.node.ports[f"qconn{index+1}"]
                if qevexpr is None:
                    qevexpr = self.await_port_input(qport)
                else:
                    qevexpr |= self.await_port_input(qport)
            cevexpr = self.await_port_input(classical_port)
            expression = yield cevexpr | qevexpr
            if expression.first_term.value:  # Upon receiving a classical message
                msgs = classical_port.rx_input().items
                assert len(msgs) == 1
                msg = msgs[0]
                match msg:
                    case Announcement():  # If the message is a BGP announcement
                        self.handle_announcement(msg)

                    case RoutingRequest():  # If the message is a routing request
                        self.handle_routing_request(msg)

                    case MeasurementResult():  # If the message is a measurement result
                        self.handle_measurement_result(msg)

                    case RoutingRequestACK():  # If the message is an acknowledgment for a routing request
                        self.handle_routing_request_ack(msg)

                    case _:
                        print("Error: cannot determine classical message type")
                        exit(1)

            if expression.second_term.value:  # Upon receiving a quantum message
                qports = [self.node.ports[f"qconn{index+1}"] for index in range(self.node.qconn_num)]
                qports += [self.node.ports[f"internal_qport{index+1}"] for index in range(self.node.internal_qport_num)]
                for qport in qports:
                    msg = qport.rx_input()
                    if msg is not None:
                        assert len(msg.meta["request"].items) == 1
                        assert len(msg.items) == 1
                        routing_request = msg.meta["request"].items[0]

                        print(f"{self.node} receives a qubit from port {qport.name} for request {routing_request.request_id} at time {ns.sim_time()}")
                        # Since the quantum source will send qubits back with the request attached,
                        # we need to check whether the request is already received.
                        if routing_request.request_id not in self.node.current_requests:
                            # If the request is not already being handled, handle it
                            self.handle_routing_request(routing_request)
                            # self.node.ports["cconn"].tx_input(routing_request)
                        self.node.qmemory.ports["input"].tx_input(msg)

    def start(self):
        super().start()
        self.start_subprotocols()


class SwapProtocol(NodeProtocol):
    """Perform Swap on a repeater node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    name : str
        Name of this protocol.

    """

    def __init__(self, node, name):
        super().__init__(node, name)

    def run(self):
        """
        When receiving a qubit (and its attached request metadata), store it in the quantum memory.
        If there are two qubits stored for a request, meaning that we are ready to do swapping.
        """
        while True:
            yield self.await_port_input(self.node.qmemory.ports["input"])
            message = self.node.qmemory.ports["input"].rx_input()

            # Extract qubit and request from memory, and store the qubit in qmemory
            qmemory = self.node.qmemory
            qubit = message.items[0]
            routing_request = message.meta["request"].items[0]

            if len(qmemory.unused_positions) == 0:
                # If the quantum memory is full, send a acknowledgment message to the speaker itself
                print(f"{self.node} is FULL! {routing_request.visited_speakers}")
                ack_msg = RoutingRequestACK(routing_request.request_id, self.node, None, False)
                self.node.ports["cconn"].tx_input(ack_msg)
            else:
                request_id = routing_request.request_id
                # If the quantum memory is not full, store it
                position = qmemory.unused_positions[0]
                print(f"Before {self.node} stores qubit for request {request_id}, unused positions: {qmemory.unused_positions}")

                # `QuantumMemory.put()` may raise exceptions, so we use a loop to solve this
                while True:
                    try:
                        qmemory.put(qubit, positions=position)
                        break
                    except ns.components.qmemory.MemPositionBusyError:
                        print("MemPositionBusyError happens, try again...")
                        qmemory.mem_positions[position].busy = False

                # Record which qubit is stored for which request, and do swapping if ready
                print(f"{self.node} stores qubit at position {position} for request {request_id}, unused positions: {qmemory.unused_positions}")
                if request_id not in qmemory.request_to_position:
                    qmemory.request_to_position[request_id] = [position]

                    # It is also possible that the repeater chain only has two nodes, so we don't need to do any swapping
                    if self.node.request_to_next_hop.get(request_id) is None:
                        if len(routing_request.visited_speakers) == 1:
                            request_ack = RoutingRequestACK(request_id, self.node, position, True)
                            routing_request.source_speaker.ports["cconn"].tx_input(request_ack)
                else:
                    qmemory.request_to_position[request_id].append(position)
                    assert len(qmemory.request_to_position[request_id]) == 2
                    positions = qmemory.request_to_position[request_id]

                    # Do swapping for request `request_id`
                    protocol = self.BellMeasureProtocol(self, request_id, positions)
                    protocol.start()

    class BellMeasureProtocol(Protocol):
        """This models the execution of the Bell measurement.
           We cannot directly `yield qmemory.execute_program(...)` because it takes some time and will lead to missing messages.
        """

        def __init__(self, parent_protocol, request_id, positions):
            super().__init__()
            self.parent_protocol = parent_protocol
            self.request_id = request_id
            self.positions = positions

            # Prepare Bell measurement instruction
            self._program = QuantumProgram(num_qubits=2)
            q1, q2 = self._program.get_qubit_indices(num_qubits=2)
            self._program.apply(INSTR_MEASURE_BELL, [q1, q2], output_key="m", physical=True, inplace=False)

        def run(self):
            while self.parent_protocol.node.qmemory.busy:
                yield self.await_timer(1)

            print(f"{self.parent_protocol.node} starts to measure for request {self.request_id} on positions {self.positions}")
            yield self.parent_protocol.node.qmemory.execute_program(self._program, qubit_mapping=self.positions, error_on_fail=True)
            yield self.await_timer(50)
            print(f"{self.parent_protocol.node} finished measurement for request {self.request_id}")
            m, = self._program.output["m"]
            msg = MeasurementResult(m, self.request_id)

            # Send result to the next hop speaker via classical channel
            # If cannot find the next hop, meaning that the request is already failed
            if self.parent_protocol.node.request_to_next_hop.get(self.request_id) is not None:
                next_speaker = self.parent_protocol.node.request_to_next_hop[self.request_id]
                next_speaker.ports["cconn"].tx_input(msg)


class SwapCorrectProgram(QuantumProgram):
    """Quantum processor program that applies all swap corrections."""
    default_num_qubits = 1

    def set_corrections(self, x_corr, z_corr):
        self.x_corr = x_corr % 2
        self.z_corr = z_corr % 2

    def program(self):
        q1, = self.get_qubit_indices(1)
        if self.x_corr == 1:
            self.apply(INSTR_X, q1)
        if self.z_corr == 1:
            self.apply(INSTR_Z, q1)
        yield self.run()


class CorrectProtocol(NodeProtocol):
    """Perform corrections for a swap on an end-node.

    Parameters
    ----------
    node : :class:`~netsquid.nodes.node.Node` or None, optional
        Node this protocol runs on.
    num_nodes : int
        Number of nodes in the repeater chain network.

    """

    class ExecuteCorrectionProgram(Protocol):
        """This models the execution of the correction procedure.
           We cannot directly `yield qmemory.execute_program(...)` because it takes some time and will lead to missing messages.
        """

        def __init__(self, parent_protocol, request_id, positions):
            super().__init__()
            self.parent_protocol = parent_protocol
            self.request_id = request_id
            self.positions = positions

        def run(self):
            request_id = self.request_id
            positions = self.positions
            if self.parent_protocol._x_corr[request_id] or self.parent_protocol._z_corr[request_id]:
                self.parent_protocol._program.set_corrections(self.parent_protocol._x_corr[request_id], self.parent_protocol._z_corr[request_id])
                while self.parent_protocol.node.qmemory.busy:
                    yield self.await_timer(1)
                yield self.parent_protocol.node.qmemory.execute_program(self.parent_protocol._program, qubit_mapping=positions)

            # Routing request has been successfully handled
            # Notify the source node about the end node and qubit position
            assert len(positions) == 1
            request_ack = RoutingRequestACK(request_id, self.parent_protocol.node, positions[0], True)
            self.parent_protocol.node.current_requests[request_id].source_speaker.ports["cconn"].tx_input(request_ack)

            del self.parent_protocol._x_corr[request_id]
            del self.parent_protocol._z_corr[request_id]
            del self.parent_protocol._counter[request_id]

    def __init__(self, node, num_nodes):
        super().__init__(node, "CorrectProtocol")
        self._x_corr = {}
        self._z_corr = {}
        self._program = SwapCorrectProgram()
        self._counter = {}

    def reset(self):
        self._x_corr.clear()
        self._z_corr.clear()
        self._counter.clear()
        super().reset()

    def run(self):
        while True:
            yield self.await_port_input(self.node.ports["correction_trigger"])
            message = self.node.ports["correction_trigger"].rx_input()
            print(f"{self.node} receives correction msg for request {message.items[0].request_id}")
            if message is None or len(message.items) != 1:
                continue
            request_id = message.items[0].request_id
            if self.node.qmemory.request_to_position.get(request_id) is None:
                # The request is already failed, so ignore this correction message
                continue
            print(f"{self.node} gets correction trigger for request {request_id}")
            m = message.items[0].measurement_result

            if request_id not in self._x_corr:
                self._x_corr[request_id] = 0
            if request_id not in self._z_corr:
                self._z_corr[request_id] = 0

            if m == ks.BellIndex.B01 or m == ks.BellIndex.B11:
                self._x_corr[request_id] += 1
            if m == ks.BellIndex.B10 or m == ks.BellIndex.B11:
                self._z_corr[request_id] += 1

            self._counter[request_id] = self._counter.get(request_id, 0) + 1

            visited_speakers = self.node.current_requests[request_id].visited_speakers
            print("Visited speakers: ", visited_speakers)
            # Note that `visited_speakers` doesn't include the current speaker, so
            # the total number of nodes in the chain is num_repeaters + 1.
            num_repeaters = len(visited_speakers)
            if self._counter[request_id] == num_repeaters - 1:
                print(f"Execute correction at {self.node} for request {request_id}")
                print("self._x_corr[request_id]", self._x_corr[request_id])
                print("self._z_corr[request_id]", self._z_corr[request_id])

                positions = self.node.qmemory.request_to_position[request_id]
                print("Position: ", positions)

                protocol = self.ExecuteCorrectionProgram(self, request_id, positions)
                protocol.start()


class QMemory(QuantumProcessor):
    """
    Wrapper for `QuantumProcessor` which maintains a map from request to qubits.
    """
    def __init__(self, name, num_positions=1, mem_noise_models=None, phys_instructions=None,
                 mem_pos_types=None, fallback_to_nonphysical=False, properties=None):
        super().__init__(name, num_positions, mem_noise_models, phys_instructions, mem_pos_types, fallback_to_nonphysical, properties)
        self.request_to_position = {}
        self.add_ports(["input"])

    def reset(self):
        self.request_to_position.clear()
        self.discard(self.used_positions)
        # super().reset()


def create_qprocessor(num_positions, gate_noise_rate=0, mem_noise_rate=0):
    """Factory to create a quantum processor for each node in the repeater chain network.

    Has memory positions and the physical instructions necessary for teleportation.

    Parameters
    ----------
    num_positions : int
        The number of qubits that the quantum memory can maintain.

    gate_noise_rate : float
        The probability that quantum operation results will depolarize.

    mem_noise_rate : float
        The probability that qubits stored in quantum memory will depolarize.

    Returns
    -------
    :class:`~netsquid.components.qprocessor.QuantumProcessor`
        A quantum processor to specification.

    """
    gate_noise_model = DepolarNoiseModel(gate_noise_rate, time_independent=True)
    mem_noise_model = DepolarNoiseModel(mem_noise_rate, time_independent=True)
    physical_instructions = [
        PhysicalInstruction(INSTR_X, duration=1, quantum_noise_model=None),
        PhysicalInstruction(INSTR_Z, duration=1, quantum_noise_model=None),
        # We have to set `apply_q_noise_after=False` to make sure the noise is added before measurement
        # Otherwise the measurement results will be precise
        PhysicalInstruction(INSTR_MEASURE_BELL, duration=7, quantum_noise_model=gate_noise_model, apply_q_noise_after=False),
    ]
    qproc = QMemory("QuantumProcessor",
                    num_positions=num_positions,
                    fallback_to_nonphysical=False,
                    mem_noise_models=[mem_noise_model] * num_positions,
                    phys_instructions=physical_instructions)
    return qproc
