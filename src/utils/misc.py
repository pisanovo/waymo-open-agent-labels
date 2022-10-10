import json
import math
import os
from pathlib import Path
from matplotlib import patches
from scipy import spatial
from scipy.spatial import KDTree
from shapely.geometry import Polygon, Point
from waymo_open_dataset.protos.map_pb2 import LaneCenter
from waymo_open_dataset.protos.scenario_pb2 import Scenario, ObjectState, Track
from config import PROJECT_DIR, AREA_LABELS_DIR
from src.labeling.areas import get_intersections_and_lane_changes, get_element_array_index_or_none, \
    get_lane_center_from_ids
from src.utils.geometry import get_intersections_polygons, get_agent_box_at_tick, get_lane_change_intersections_polygons


def get_intersection_lane_changes_cache(scenario: Scenario):
    """

    Args:
        scenario: The scenario to be used

    Returns: Retrieves the scenario areas from disk if available

    """
    json_path = f"{PROJECT_DIR}/{AREA_LABELS_DIR}/{scenario.scenario_id}"
    # Check if cache exists
    if os.path.isfile(json_path):
        f = open(json_path)
        data = json.load(f)

        intersections = data[scenario.scenario_id]["intersections"]
        lane_changes = data[scenario.scenario_id]["lane_changes"]
    else:
        intersections, lane_changes = get_intersections_and_lane_changes(scenario)

        area_json = {
            scenario.scenario_id: {
                "intersections": intersections,
                "lane_changes": lane_changes
            }
        }

        Path(f"{PROJECT_DIR}/{AREA_LABELS_DIR}").mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(area_json, f, ensure_ascii=False, indent=4)

    return intersections, lane_changes


def is_agent_in_lane_change_intersection_at_tick(scenario: Scenario, lane_changes: list[list[int]], tick: int,
                                                 agent_id: int, kd_tree) -> bool:
    """

    Args:
        scenario: The scenario to be used
        lane_changes: A nested list of lane changes boundary points
        tick: The scenario time step
        agent_id: Id for the agent to check

    Returns: Whether the agent is in any intersection at a given tick

    """
    lane_changes_intersections_polygons = get_lane_change_intersections_polygons(scenario, lane_changes)

    agent_state: ObjectState = next(track.states[tick] for track in scenario.tracks if track.id == agent_id)

    angle = math.degrees(agent_state.heading) + 90

    rect = patches.Rectangle(
        (agent_state.center_x, agent_state.center_y),
        agent_state.width,
        agent_state.length,
        angle=angle,
    )

    agent_poly_points = [tuple(point_arr) for point_arr in rect.get_verts()]
    agent_polygon = Polygon(agent_poly_points)

    for lane_changes_intersections_polygon in lane_changes_intersections_polygons:
        distance = lane_changes_intersections_polygon.distance(agent_polygon)
        if agent_polygon.intersects(lane_changes_intersections_polygon) or distance < 0.5:
            return True

    return False


def is_agent_in_intersection_at_tick(scenario: Scenario, intersections: list[list[int]], tick: int,
                                     agent_id: int, use_agent_center_point=False) -> bool:
    """

    Args:
        scenario: The scenario to be used
        intersections: A nested list of intersection boundary points
        tick: The scenario time step
        agent_id: Id for the agent to check

    Returns: Whether the agent is in any intersection at a given tick

    """

    if not next(track.states[tick].valid for track in scenario.tracks if track.id == agent_id):
        return False

    intersections_polygons = get_intersections_polygons(scenario, intersections)

    agent_state: ObjectState = next(track.states[tick] for track in scenario.tracks if track.id == agent_id)

    # angle = math.degrees(agent_state.heading) + 90
    angle = (math.degrees(agent_state.heading) - 90) % 360

    rect = patches.Rectangle(
        (agent_state.center_x, agent_state.center_y),
        agent_state.width,
        agent_state.length,
        angle=angle,
    )

    agent_poly_points = [tuple(point_arr) for point_arr in rect.get_verts()]

    if use_agent_center_point:
        agent_point = Point(agent_state.center_x, agent_state.center_y)
        for intersections_polygon in intersections_polygons:
            if agent_point.intersects(intersections_polygon):
                return True
    else:
        agent_polygon = Polygon(agent_poly_points)
        for intersection_polygon in intersections_polygons:
            if agent_polygon.intersects(intersection_polygon):
                return True

    return False


def get_agents_in_intersections_at_tick(scenario: Scenario, intersections: list[list[int]], tick: int) -> list[int]:
    """

    Args:
        scenario: The scenario to be used
        intersections: A nested list of intersection boundary points
        tick: The scenario time step

    Returns: A list of all agent ids that are in any intersection at a given tick

    """
    agent_ids: list[int] = [track.id for track in scenario.tracks
                            if track.object_type != Track.ObjectType.TYPE_CYCLIST
                            and track.object_type != Track.ObjectType.TYPE_PEDESTRIAN]

    agents_in_intersections = [idx for idx in agent_ids
                               if is_agent_in_intersection_at_tick(scenario, intersections, tick, idx)]

    return agents_in_intersections


def get_scenario_lane_center_kd_tree(scenario: Scenario):
    """

    Args:
        scenario: The scenario to be used

    Returns: The KD Tree for lane centers in a scenario

    """
    # Get all scene lane centers that are not bike lanes
    lane_centers: list[LaneCenter] = [getattr(map_feature, "lane")
                                      for map_feature in scenario.map_features
                                      if map_feature.WhichOneof("feature_data") == "lane"
                                      and getattr(map_feature,
                                                  "lane").type != LaneCenter.LaneType.TYPE_BIKE_LANE]

    lane_points = []

    for lane_center in lane_centers:
        for point in lane_center.polyline:
            lane_points.append((point.x, point.y))

    return spatial.KDTree(lane_points)


def get_agent_lane_center(scenario: Scenario, tick: int, agent_id: int, kd: KDTree, k_lane: int = 1) -> (
        int, int, float):
    """

    Args:
        scenario: The scenario to be used
        tick: The scenario time step
        agent_id: The id of the agent to be used
        kd: The KD Tree for lane centers

    Returns: agent id, closest center lane id and distance

    """

    # Check if agent and tracks step together are valid
    if not next(track.states[tick].valid for track in scenario.tracks if track.id == agent_id):
        raise ValueError(f"Received invalid agent (id:{agent_id}) at tick {tick}")

    lane_centers = [(map_feature.id, getattr(map_feature, "lane"))
                    for map_feature in scenario.map_features
                    if map_feature.WhichOneof("feature_data") == "lane"
                    and getattr(map_feature,
                                "lane").type != LaneCenter.LaneType.TYPE_BIKE_LANE]

    lane_point_ids = []

    for i_lane_center in lane_centers:
        lane_id, lane_center = i_lane_center
        lane_point_ids.append([lane_id, len(lane_center.polyline)])

    agent_pos = next(
        (track.states[tick].center_x, track.states[tick].center_y) for track in scenario.tracks if track.id == agent_id)

    if k_lane > 1:
        k = 2

        while True:
            distance_list, point_index_list = kd.query(agent_pos, k)
            lane_ids = []
            for i, point_index in enumerate(point_index_list):
                lane_id = get_point_index_lane_id(point_index, lane_point_ids)
                lane_ids.append(lane_id)
                lane_ids = list(set(lane_ids))

                if len(lane_ids) == k_lane:
                    return agent_id, lane_id, distance_list[i]

            k += 1
    else:
        distance, point_index = kd.query(agent_pos)
        lane_id = get_point_index_lane_id(point_index, lane_point_ids)
        return agent_id, lane_id, distance


def get_point_index_lane_id(point_index: int, lane_point_ids):
    lane_point_id_index = 0
    i = 0
    while lane_point_id_index <= point_index:
        lane_point_id_index += lane_point_ids[i][1]
        i += 1

    return lane_point_ids[i - 1][0]


def get_agents_lane_centers(scenario: Scenario, tick: int, kd_tree: KDTree) -> list[(int, int, float)]:
    """

    Args:
        scenario: The scenario to be used
        tick: The scenario time step

    Returns: A list containing for each agent the agent id, closest center lane id and distance

    """
    agents_id_closest_lane_id_dist_lane = [
        get_agent_lane_center(scenario, tick, track.id, kd_tree)
        for track in scenario.tracks
        if track.states[tick].valid
    ]

    return agents_id_closest_lane_id_dist_lane


def is_agent_valid_at_tick(scenario: Scenario, agent_id: int, tick: int):
    """

    Args:
        scenario: The scenario to be used
        agent_id: Agent to check
        tick: Tick to check

    Returns: Whether the agent is valid at the specified tick

    """
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    if tick < 0 or tick >= num_steps:
        return False

    return next(track.states[tick].valid for track in scenario.tracks if track.id == agent_id)


def touches_lane_at_tick(scenario: Scenario, tick: int, agent_id: int, kd_tree: KDTree):
    """

        Args:
            kd_tree:
            scenario: The scenario to be used
            tick: The time step
            agent_id: Id for the agent

        Returns: True if the agent touches a lane at the specified tick,
            False if the agent touches no lane at the specified tick or if the agent is invalid at the specified tick

        """

    agent_id_, lane_point_ids, distance = get_agent_lane_center(scenario, tick, agent_id, kd_tree)

    for agent_track in scenario.tracks:
        if agent_track.id == agent_id:
            if agent_track.states[tick].valid:
                state = agent_track.states[tick]

                # if closest lane is further than the size of the agent, the agent cannot touch a lane
                max_agent_size = max(state.width / 2, state.height / 2)
                if (distance > max_agent_size):
                    return False

                lane_centers: list[(int, LaneCenter)] = [(map_feature.id, getattr(map_feature, "lane"))
                                                         for map_feature in scenario.map_features
                                                         if map_feature.WhichOneof("feature_data") == "lane"]
                lane: LaneCenter = next(lane for id, lane in lane_centers
                                        if id == lane_point_ids)
                lane_poly = lane.polyline

                polygon = get_agent_box_at_tick(scenario, tick, agent_id)

                point0 = Point(lane_poly[0].x, lane_poly[0].y)
                point1 = Point(lane_poly[1].x, lane_poly[1].y)

                distanceBetweenPoints = math.sqrt(
                    (point0.x - point1.x) * (point0.x - point1.x) + (point0.y - point1.y) * (point0.y - point1.y))

                i = 0
                while i < len(lane_poly):
                    point = Point(lane_poly[i].x, lane_poly[i].y)
                    if polygon.contains(point):
                        return True
                    else:
                        distanceToCurrentPoint = math.sqrt(
                            (state.center_x - point.x) * (state.center_x - point.x) + (state.center_y - point.y) * (
                                    state.center_y - point.y))
                        skip = math.floor((distanceToCurrentPoint - max_agent_size) / distanceBetweenPoints)
                        i += max(1, skip)
                        # print("skip:", skip)

            # return False if the agent is invalid
            else:
                return False

    return False


def get_lc_intersection_lanes(scenario, lane_changes, lane_id: int):
    """
    Given a lane id, if that lane is part of a lci return all lanes within that lane change intersection
    Args:
        scenario: The scenario to be used
        lane_changes: Set of all scenario lane change intersections
        lane_id: The given lane

    Returns: All lanes from the matching lane change intersection or empty array

    """
    index_in_lane_changes = get_element_array_index_or_none(lane_changes, lane_id)
    if index_in_lane_changes is None:
        return []
    lane_ids_in_lc_intersection = lane_changes[index_in_lane_changes]
    lanes = [get_lane_center_from_ids(scenario, idx) for idx in lane_ids_in_lc_intersection]

    return lanes