import math

import numpy as np
from waymo_open_dataset.protos.map_pb2 import MapPoint
from waymo_open_dataset.protos.scenario_pb2 import Track


def intersection_poly_lines(p1: MapPoint, p2: MapPoint, q1: MapPoint, q2: MapPoint):
    """

    Args:
        p1: Poly segment 1 start point
        p2: Poly segment 1 end point
        q1: Poly segment 2 start point
        q2: Poly segment 2 end point

    Returns: Whether two polyline segments intersect

    """
    p1 = [p1.x, p1.y]
    p2 = [p2.x, p2.y]
    q1 = [q1.x, q1.y]
    q2 = [q2.x, q2.y]
    d1 = cross(np.subtract(p1, q1), np.subtract(q2, q1))
    d2 = cross(np.subtract(p2, q1), np.subtract(q2, q1))
    d3 = cross(np.subtract(q1, p1), np.subtract(p2, p1))
    d4 = cross(np.subtract(q2, p1), np.subtract(p2, p1))
    if ((d1 * d2) < 0) and ((d3 * d4) < 0):
        return True
    elif (((d1 * d2) == 0) and ((d3 * d4) < 0)) or (((d1 * d2) < 0) and ((d3 * d4) == 0)):
        return True
    else:
        return False


def get_agent_speed(agent_track: Track, step: int):
    speed = round(math.sqrt(
        agent_track.states[step].velocity_x ** 2 + agent_track.states[step].velocity_y ** 2
    ) * 3.6, 2)

    return speed


# checks whether the lines p1p2 and p3p4 intersect
# returns:
#   0 for no intersection
#   1 for a line starting on the other one
#   2 for full intersection
def intersection(p1, p2, p3, p4):
    d1 = cross(np.subtract(p1, p3), np.subtract(p4, p3))
    d2 = cross(np.subtract(p2, p3), np.subtract(p4, p3))
    d3 = cross(np.subtract(p3, p1), np.subtract(p2, p1))
    d4 = cross(np.subtract(p4, p1), np.subtract(p2, p1))
    if ((d1 * d2) < 0) and ((d3 * d4) < 0):
        return 2
    elif (((d1 * d2) == 0) and ((d3 * d4) < 0)) or (((d1 * d2) < 0) and ((d3 * d4) == 0)):
        return 1
    else:
        return 0


def cross(p1, p2):
    return p1[0] * p2[1] - p1[1] * p2[0]
