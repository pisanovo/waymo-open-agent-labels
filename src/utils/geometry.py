import math
import numpy as np
from waymo_open_dataset.protos.map_pb2 import LaneCenter
from waymo_open_dataset.protos.scenario_pb2 import Scenario

from shapely.geometry import Polygon


def ccw_sort(p):
    """
    https://stackoverflow.com/questions/44025403/how-to-use-matplotlib-path-to-draw-polygon
    """
    p = np.array(p)
    mean = np.mean(p, axis=0)
    d = p - mean
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]


# creates a shapely polygon for an agent at a certain step
def get_agent_box_at_tick(scenario: Scenario, tick: int, agent_id: int):
    polygon = "missing"

    for agent_track in scenario.tracks:
        if agent_track.id == agent_id:

            # Add agent state position if state is valid
            if agent_track.states[tick].valid:
                state = agent_track.states[tick]
                angle = math.radians((math.degrees(state.heading) -90) % 360)

                x1 = state.width / 2
                y1 = state.length / 2
                x2 = -state.width / 2
                y2 = state.length / 2
                x3 = -state.width / 2
                y3 = -state.length / 2
                x4 = state.width / 2
                y4 = -state.length / 2

                # rotate points around (0, 0)
                x1_ = x1 * math.cos(angle) - y1 * math.sin(angle)
                y1_ = x1 * math.sin(angle) + y1 * math.cos(angle)
                x2_ = x2 * math.cos(angle) - y2 * math.sin(angle)
                y2_ = x2 * math.sin(angle) + y2 * math.cos(angle)
                x3_ = x3 * math.cos(angle) - y3 * math.sin(angle)
                y3_ = x3 * math.sin(angle) + y3 * math.cos(angle)
                x4_ = x4 * math.cos(angle) - y4 * math.sin(angle)
                y4_ = x4 * math.sin(angle) + y4 * math.cos(angle)

                # move points from (0, 0) to agent center
                x1_ = x1_ + state.center_x
                y1_ = y1_ + state.center_y
                x2_ = x2_ + state.center_x
                y2_ = y2_ + state.center_y
                x3_ = x3_ + state.center_x
                y3_ = y3_ + state.center_y
                x4_ = x4_ + state.center_x
                y4_ = y4_ + state.center_y

                polygon = Polygon([[x1_, y1_], [x2_, y2_], [x3_, y3_], [x4_, y4_]])

            else:
                polygon == "invalid"

    # raise error if agent is invalid at the specified step
    if polygon == "invalid":
        raise ValueError(f"Agent ID {agent_id} is invalid at tick {tick}")

    # raise error if agent id does not exist
    if polygon == "missing":
        raise ValueError(f"Agent ID {agent_id} not in scenario")

    return polygon


def get_lane_change_intersections_polygons(scenario: Scenario, lane_changes: list[list[int]]) -> list[Polygon]:
    """

    Args:
        scenario: The scenario to be used
        intersections: A nested list of intersection boundary points

    Returns: The intersection as a polygon

    """
    lane_changes_intersections_polygons: list[Polygon] = []

    for lane_change in lane_changes:
        road_centers: list[LaneCenter] = [getattr(map_feature, "lane") for map_feature in scenario.map_features
                                          if map_feature.id in lane_change]
        unsorted_poly_points: list[(int, int)] = []

        num_points = min([len(road_center.polyline) for road_center in road_centers])

        for road_center in road_centers:
            unsorted_poly_points.append((road_center.polyline[0].x, road_center.polyline[0].y))
            unsorted_poly_points.append((road_center.polyline[num_points - 1].x, road_center.polyline[num_points - 1].y))

        # Sort polygon points to avoid wrong intersection polygons
        points = [tuple(point_arr) for point_arr in ccw_sort(unsorted_poly_points)]
        lane_changes_intersections_polygons.append(Polygon(points))

    return lane_changes_intersections_polygons


def get_intersections_polygons(scenario: Scenario, intersections: list[list[int]]) -> list[Polygon]:
    """

    Args:
        scenario: The scenario to be used
        intersections: A nested list of intersection boundary points

    Returns: The intersection as a polygon

    """
    intersections_polygons: list[Polygon] = []

    for intersection in intersections:
        road_centers: list[LaneCenter] = [getattr(map_feature, "lane") for map_feature in scenario.map_features
                                          if map_feature.id in intersection]
        unsorted_poly_points: list[(int, int)] = []

        for road_center in road_centers:
            unsorted_poly_points.append((road_center.polyline[0].x, road_center.polyline[0].y))
            unsorted_poly_points.append((road_center.polyline[len(road_center.polyline) - 1].x,
                                         road_center.polyline[len(road_center.polyline) - 1].y))

        # Sort polygon points to avoid wrong intersection polygons
        points = [tuple(point_arr) for point_arr in ccw_sort(unsorted_poly_points)]
        intersections_polygons.append(Polygon(points))

    return intersections_polygons
