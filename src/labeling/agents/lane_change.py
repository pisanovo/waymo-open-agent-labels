import math
from enum import Enum
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
from waymo_open_dataset.protos.map_pb2 import LaneCenter
from waymo_open_dataset.protos.scenario_pb2 import Scenario
from src.utils.misc import get_intersection_lane_changes_cache, is_agent_in_intersection_at_tick, \
    is_agent_in_lane_change_intersection_at_tick, get_agent_lane_center, get_lc_intersection_lanes


class LaneChangeTypes(Enum):
    LEFT_LANE_CHANGE = 1
    RIGHT_LANE_CHANGE = 2


def generate_agent_lane_change_timeline(scenario: Scenario, agent_id: int, kd_tree: KDTree, lc_type: LaneChangeTypes) \
        -> list[int]:
    intersections, lane_changes = get_intersection_lane_changes_cache(scenario)
    lane_centers: list[(int, LaneCenter)] = [(map_feature.id, getattr(map_feature, "lane"))
                                             for map_feature in scenario.map_features
                                             if map_feature.WhichOneof("feature_data") == "lane"
                                             and getattr(map_feature,
                                                         "lane").type != LaneCenter.LaneType.TYPE_BIKE_LANE]

    # Get the lane change step timelines from each of the three categories:
    #   1. Lane changes occurring in intersections
    #   2. Lane changes occurring in lane change intersections
    #   3. Lane changes that are not part of category (1) and (2)
    steps_intersection = get_in_intersection(scenario, agent_id, kd_tree, intersections, lane_centers, lc_type)
    steps_lc_intersection = get_in_lane_change_intersection(scenario, agent_id, kd_tree, lane_changes, lane_centers, lc_type)
    steps_other = get_other(scenario, agent_id, kd_tree, intersections, lane_changes, lane_centers, lc_type)

    merged_steps = list(set(steps_lc_intersection + steps_intersection + steps_other))

    return merged_steps


def get_in_intersection(scenario: Scenario, agent_id: int, kd_tree: KDTree, intersections, lane_centers, lc_type):
    """
    Retrieves steps during which the specified agent performs a lc within an intersection
    Args:
        scenario: The scenario to be used
        agent_id: Integer id of the agent
        kd_tree: KDTree used to fetch closest lane given a position
        intersections: The set of intersections in the scene
        lane_centers: Lane centers in the scene
        lc_type: Lane change type

    Returns: List of steps where a lc occurs
    """
    steps = []

    lane_before_id = None
    lane_set_at_step = None
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    lane_id_after_intersection = get_lane_id_after_intersection(scenario, agent_id, kd_tree, intersections)

    for step in range(num_steps):
        if not next(track.states[step].valid for track in scenario.tracks if track.id == agent_id):
            continue

        is_in_intersection_now = is_agent_in_intersection_at_tick(scenario, intersections, step, agent_id)
        closest_lane_id = get_agent_lane_center(scenario, step, agent_id, kd_tree)[1]

        if is_in_intersection_now:
            lane_now_id = closest_lane_id

            if lane_before_id is not None and lane_id_after_intersection is not None and lane_before_id != lane_now_id:
                lane_before: LaneCenter = next(lane for idx, lane in lane_centers
                                               if idx == lane_before_id)

                lane_now: LaneCenter = next(lane for idx, lane in lane_centers
                                            if idx == lane_now_id)

                has_common_exit_lane = any(lane in lane_now.exit_lanes for lane in lane_before.exit_lanes)
                has_common_entry_lane = any(lane in lane_now.entry_lanes for lane in lane_before.entry_lanes)

                # Check if continuation exists
                if lane_id_after_intersection is None:
                    continuation_exists = True
                else:
                    continuation_exists = search_following_lane(
                        lane_now_id, lane_id_after_intersection, 0, lane_centers, intersections
                    )

                # Get lane neighbours
                neighbours = lane_before.left_neighbors if lc_type == LaneChangeTypes.LEFT_LANE_CHANGE \
                    else lane_before.right_neighbors

                # Check if lane change is valid
                if lane_now_id in [neighbour.feature_id for neighbour in neighbours] \
                        and not has_common_exit_lane and not has_common_entry_lane and continuation_exists:
                    steps_found = get_lane_change_steps(scenario, agent_id, step, lane_set_at_step, kd_tree)

                    if steps_found:
                        steps.extend(steps_found)
                        lane_before_id = lane_now_id
                        lane_set_at_step = step
                elif lane_now_id in lane_before.exit_lanes:
                    lane_before_id = lane_now_id
                    lane_set_at_step = step

        else:
            if lane_before_id is None or lane_before_id != closest_lane_id:
                lane_before_id = closest_lane_id
                lane_set_at_step = step

    return steps


def get_lane_id_after_intersection(scenario: Scenario, agent_id: int, kd_tree: KDTree, intersections):
    """
    Get the lane center after an intersection for a given agent id
    Args:
        scenario: The scenario to be used
        agent_id: The identifier of the agent
        kd_tree: KDTree used to retrieve closest lanes
        intersections: Set of all intersection within the given scenario

    Returns: Lane id of the first lane after an intersection

    """
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    inside_intersection = False

    for step in range(num_steps):
        if not next(track.states[step].valid for track in scenario.tracks if track.id == agent_id):
            continue

        is_in_intersection_now = is_agent_in_intersection_at_tick(scenario, intersections, step, agent_id)

        if not inside_intersection and is_in_intersection_now:
            inside_intersection = True
        elif inside_intersection and not is_in_intersection_now:
            return get_agent_lane_center(scenario, step, agent_id, kd_tree)[1]


def get_in_lane_change_intersection(scenario: Scenario, agent_id: int, kd_tree: KDTree, lane_changes, lane_centers, lc_type):
    """
    Retrieves steps during which the specified agent performs a lc within a lane change intersection
    Args:
        scenario: The scenario to be used
        agent_id: Integer id of the agent
        kd_tree: KDTree used to fetch closest lane given a position
        lane_changes: The set of lane change intersections in the scene
        lane_centers: Lane centers in the scene
        lc_type: Lane change type

    Returns: List of steps where a lc occurs
    """
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    distance_ext_line_threshold = 2.5  # meters
    steps = []

    is_in_lane_change_before = False
    step_enter_lane_change: int
    step_exit_lane_change: int

    for step in range(num_steps):
        if not next(track.states[step].valid for track in scenario.tracks if track.id == agent_id):
            continue

        is_in_lane_change_now = is_agent_in_lane_change_intersection_at_tick(scenario, lane_changes, step, agent_id, kd_tree)

        if not is_in_lane_change_before and is_in_lane_change_now:
            is_in_lane_change_before = True
            step_enter_lane_change = step
        elif is_in_lane_change_before and not is_in_lane_change_now:
            is_in_lane_change_before = False
            step_exit_lane_change = step

            lane_id_after_intersection = get_agent_lane_center(scenario, step, agent_id, kd_tree)[1]

            lane_after_intersection: LaneCenter = next(lane for id, lane in lane_centers
                                                         if id == lane_id_after_intersection)

            if not lane_after_intersection.entry_lanes:
                lane_id_when_in_intersection = lane_id_after_intersection
            else:
                lane_id_when_in_intersection = lane_after_intersection.entry_lanes[0]

            lc_lanes: list[LaneCenter] = get_lc_intersection_lanes(scenario, lane_changes, lane_id_when_in_intersection)

            if not lc_lanes:
                lane_id_when_in_intersection = lane_id_after_intersection
                lc_lanes = get_lc_intersection_lanes(scenario, lane_changes, lane_id_when_in_intersection)

            # sometimes the closest lane doesn't refer to the lane within the lc, thus get the n-th closest lane
            k = 2
            while not lc_lanes:
                lane_id_when_in_intersection = get_agent_lane_center(scenario, step_enter_lane_change, agent_id, kd_tree, k_lane=k)[1]

                lc_lanes = get_lc_intersection_lanes(scenario, lane_changes, lane_id_when_in_intersection)
                k += 1

            lane_when_in_intersection: LaneCenter = next(lane for id, lane in lane_centers
                                                         if id == lane_id_when_in_intersection)

            lane_id_before_intersection = lane_when_in_intersection.entry_lanes[0]
            lane_before_intersection: LaneCenter = next(lane for id, lane in lane_centers
                                                        if id == lane_id_before_intersection)

            last_point_before_intersection = lane_before_intersection.polyline[
                len(lane_before_intersection.polyline) - 1]
            second_last_point_before_intersection = lane_before_intersection.polyline[
                len(lane_before_intersection.polyline) - 2]

            num_points = min([len(road_center.polyline) for road_center in lc_lanes])

            first_point_after_intersection = lane_when_in_intersection.polyline[num_points - 1]

            is_of_line = is_left_of_line if lc_type == LaneChangeTypes.LEFT_LANE_CHANGE else is_right_of_line

            # Check if a valid lane change occured
            if distance(second_last_point_before_intersection, last_point_before_intersection, first_point_after_intersection) > distance_ext_line_threshold:
                if is_of_line(second_last_point_before_intersection, last_point_before_intersection, first_point_after_intersection):
                    steps.extend([step for step in range(step_enter_lane_change, step_exit_lane_change+1)])

    return steps


def get_other(scenario: Scenario, agent_id: int, kd_tree: KDTree, intersections, lane_changes, lane_centers, lc_type):
    """
    Retrieves steps during which the specified agent performs a lc that is not in an intersection or
    lane change intersection
    Args:
        scenario: The scenario to be used
        agent_id: Integer id of the agent
        kd_tree: KDTree used to fetch closest lane given a position
        intersections: The set of intersections in the scene
        lane_changes: The set of lane change intersections in the scene
        lane_centers: Lane centers in the scene
        lc_type: Lane change type

    Returns: List of steps where a lc occurs
    """
    steps = []

    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    lane_before_id = None
    lane_set_at_step = None

    for step in range(num_steps):
        if not next(track.states[step].valid for track in scenario.tracks if track.id == agent_id):
            continue

        closest_lane_id = get_agent_lane_center(scenario, step, agent_id, kd_tree)[1]

        is_in_lane_change_now = is_agent_in_lane_change_intersection_at_tick(scenario, lane_changes, step, agent_id, kd_tree)
        is_in_intersection_now = is_agent_in_intersection_at_tick(scenario, intersections, step, agent_id)

        if not is_in_lane_change_now and not is_in_intersection_now:
            lane_now_id = closest_lane_id

            if lane_before_id is None:
                lane_before_id = lane_now_id
                lane_set_at_step = step

            elif lane_before_id != lane_now_id:
                lane_before: LaneCenter = next(lane for idx, lane in lane_centers
                                               if idx == lane_before_id)

                lane_now: LaneCenter = next(lane for idx, lane in lane_centers
                                            if idx == lane_now_id)

                has_common_exit_lane = any(lane in lane_now.exit_lanes for lane in lane_before.exit_lanes)

                neighbours = lane_before.left_neighbors if lc_type == LaneChangeTypes.LEFT_LANE_CHANGE \
                    else lane_before.right_neighbors

                if lane_now_id in [neighbour.feature_id for neighbour in neighbours] and not has_common_exit_lane:
                    steps_found = get_lane_change_steps(scenario, agent_id, step, lane_set_at_step, kd_tree)
                    steps.extend(steps_found)

                lane_before_id = lane_now_id
                lane_set_at_step = step

    return steps


def get_lane_change_steps(scenario: Scenario, agent_id: int, detection_step: int, lane_before_step: int,
                          kd_tree: KDTree, sigma: int = 4, threshold_num_steps: int = 2):
    """
    When a lane change is detected, identify the starting and end time step for the lane change maneuver
    Args:
        scenario: The scenario to be used
        agent_id: Integer id of the agent
        kd_tree: KDTree used to fetch closest lane given a position
        detection_step: The step at which a lane change was detected
        lane_before_step: The step at which the previous lane was set
        sigma: Smoothing factor for distances to lanes
        threshold_num_steps: Only consider lc where the maneuver occurred for at least threshold_num_steps steps

    Returns: List of steps where the lc occurred
    """
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    lane_now_id = get_agent_lane_center(scenario, detection_step, agent_id, kd_tree)[1]

    distance_steps = []
    distance_list = []

    for i in range(lane_before_step, detection_step + 1):
        if next(track.states[i].valid for track in scenario.tracks if track.id == agent_id):
            distance_steps.append(i)
            distance_list.append(get_agent_lane_center(scenario, i, agent_id, kd_tree)[2])

    # Apply distance smoothing
    distance_list_smooth = gaussian_filter1d(distance_list, sigma=sigma)

    steps = []

    for i in range(len(distance_list_smooth) - 1, -1, -1):
        if distance_list_smooth[i - 1] < distance_list_smooth[i]:
            steps.append(distance_steps[i])
        else:
            break

    distance_steps = []
    distance_list = []

    for i in range(detection_step + 1, num_steps):
        if next(track.states[i].valid for track in scenario.tracks if track.id == agent_id):
            _, lane_id, distance = get_agent_lane_center(scenario, i, agent_id, kd_tree)
            if lane_id == lane_now_id:
                distance_steps.append(i)
                distance_list.append(distance)
            else:
                break

    # Apply distance smoothing
    distance_list_smooth = gaussian_filter1d(distance_list, sigma=sigma)

    for i in range(0, len(distance_list_smooth) - 1):
        if distance_list_smooth[i] > distance_list_smooth[i + 1]:
            steps.append(distance_steps[i])
        else:
            break

    if len(steps) <= threshold_num_steps:
        steps = []

    return steps


def search_following_lane(from_lane_id: int, outside_intersection_lane_id: int, depth: int, lane_centers: list[LaneCenter], intersections):
    """
    Given an origin lane and a destination lane, calculate whether it is possible to reach the destination in at most
    x (depth) lane switches using valid maneuvers (keep lane, left lane change, right lane change)
    This method is used to identify if a lane change inside of an intersection should be valid
    Args:
        from_lane_id: Origin lane
        outside_intersection_lane_id: Destination lane outside of the intersection
        depth: current depth counter
        lane_centers: Scenario lane centers
        intersections: The set of intersection in a scenario

    Returns: Whether a valid continuation exists
    """
    if depth > 3:
        return False

    from_lane: LaneCenter = next(lane for id, lane in lane_centers
                                   if id == from_lane_id)
    outside_lane: LaneCenter = next(lane for id, lane in lane_centers
                                   if id == outside_intersection_lane_id)

    exit_lane_outside = True
    try:
        exit_lane = from_lane.exit_lanes[0]
    except:
        return False

    for intersection in intersections:
        if exit_lane in intersection:
            exit_lane_outside = False

    left_neighbours = [neighbour.feature_id for neighbour in outside_lane.left_neighbors]
    right_neighbours = [neighbour.feature_id for neighbour in outside_lane.right_neighbors]

    if exit_lane_outside and (exit_lane == outside_intersection_lane_id or exit_lane in left_neighbours + right_neighbours):
        return True

    if exit_lane_outside and (exit_lane != outside_intersection_lane_id or exit_lane not in outside_lane.left_neighbors + outside_lane.right_neighbors):
        return False
    else:
        # Continue search with neighbours (representing lane changes)
        for lane in left_neighbours + right_neighbours:
            res = search_following_lane(lane, outside_intersection_lane_id, depth + 1, lane_centers, intersections)
            if res:
                return True


def distance(line_a, line_b, point):
    """
    Calculates distance between point and line
    Args:
        line_a: Point on line
        line_b: Other point on line
        point: Point to calculate the distance for

    Returns: The distance from the point to the line
    """
    a = line_a.y - line_b.y
    b = line_b.x - line_a.x
    c = line_a.x * line_b.y - line_b.x * line_a.y

    return abs(a*point.x + b*point.y + c) / math.sqrt(a**2 + b**2)


def is_left_of_line(line_a, line_b, point):
    """
    Calculate whether a point is on the left side of a line
    Args:
        line_a: Point on line
        line_b: Other point on line
        point: Point to calculate the distance for

    Returns: Whether the point is on the left side of the line
    """
    return ((line_b.x - line_a.x) * (point.y - line_a.y) - (line_b.y - line_a.y) * (point.x - line_a.x)) > 0


def is_right_of_line(line_a, line_b, point):
    """
    Calculate whether a point is on the right side of a line
    Args:
        line_a: Point on line
        line_b: Other point on line
        point: Point to calculate the distance for

    Returns: Whether the point is on the right side of the line
    """
    return ((line_b.x - line_a.x) * (point.y - line_a.y) - (line_b.y - line_a.y) * (point.x - line_a.x)) < 0

