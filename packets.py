import netsquid as ns


class Announcement:
    '''A BGP announcement message'''

    def __init__(self, ip, path):
        self.ip = ip
        self.path = path

    def __str__(self):
        return f"IP: {self.ip}, path: {self.path}"


class RoutingRequest:
    '''A message that requests to access an IP.'''

    # Static variable used as the ID of routing requests, this guarantees that each request has a unique ID.
    request_id = 0

    def __init__(self, source_speaker, dest_ip, path=None):
        self.request_id = RoutingRequest.request_id
        RoutingRequest.request_id += 1
        self.source_speaker = source_speaker
        self.dest_ip = dest_ip
        # If `path` is not None, the request will be routed along the path
        # Otherwise, the routing algorithm determines the path
        self.path = path
        # Store all the speakers that have tried to serve this request
        # This is useful because we should not try paths that have already tried
        self.visited_speakers = []
        self.start_time = ns.sim_time()  # The time stamp that the request is generated, used to compute latency

    def __repr__(self):
        return f"Source speaker: {self.source_speaker}, destination: {self.dest_ip}"


class RoutingRequestACK:
    """
    A message that relies an routing request.
    When the routing is successful, the message is sent to the source speaker,
    so that it knows the entanglement has been established and can be used.
    When the routing is failed, the message is sent to all the visited speakers,
    so that they can release resources preserved for this request.
    """

    def __init__(self, request_id, destination_speaker, destination_position, success):
        """
        request_id: int
            The unique ID for this request
        destination_speaker: BGPSpeaker
            The end node of the entanglement
        destination_position: int
            The qubit position stored in the end node
        success: bool
            Determine whether the request is successful or not
        """
        self.request_id = request_id
        self.destination_speaker = destination_speaker
        self.destination_position = destination_position
        self.success = success


class MeasurementResult:
    """
    Measurement results sent by repeaters who make Bell state measurement while swapping.
    We also attach the request ID, so the receiver knows which qubit it should correct.
    """

    def __init__(self, measurement_result, request_id):
        self.measurement_result = measurement_result
        self.request_id = request_id
