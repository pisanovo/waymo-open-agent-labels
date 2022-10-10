import math
from enum import Enum

import numpy as np
from scipy.spatial import KDTree
from waymo_open_dataset.protos.map_pb2 import LaneCenter
from waymo_open_dataset.protos.scenario_pb2 import Scenario

from config import LEFT_TURN_VEC_THRESHOLD, RIGHT_TURN_VEC_THRESHOLD, U_TURN_VEC_THRESHOLD
from src.utils.misc import get_intersection_lane_changes_cache, is_agent_in_intersection_at_tick, get_agent_lane_center, \
    is_agent_valid_at_tick


class TurnTypes(Enum):
    LEFT_TURN = 1
    RIGHT_TURN = 2


def generate_turn_timeline(scenario: Scenario, agent_id: int, kd_tree: KDTree, turn_type: TurnTypes):
    """
    Args:
        scenario: The scenario to be used for agent labeling
        agent_id: The id for the agent for which the turn labels are generated
        kd_tree: KDTree for all scenario map lane centers excluding bike lanes
        turn_type: The turn type (left/right turn)

    Returns: a list of steps where a turn for the specified turn type occurs
    """

    # Scenario lanes used to generate the timeline, current implementation excludes bike lanes
    lane_centers: list[(int, LaneCenter)] = [(map_feature.id, getattr(map_feature, "lane"))
                                             for map_feature in scenario.map_features
                                             if map_feature.WhichOneof("feature_data") == "lane"
                                             and getattr(map_feature,
                                                         "lane").type != LaneCenter.LaneType.TYPE_BIKE_LANE]

    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    intersections, _ = get_intersection_lane_changes_cache(scenario)
    steps_turn: list[int] = []

    in_intersection_before: bool = False
    step_enter_intersection: int
    lane_id_before_intersection: int
    lane_id_after_intersection: int

    # The entry vector for an agent in an intersection can be the heading of the object or a vector from the last
    # two points of the lane center that the agent was assigned to before the intersection
    entry_vec: [float, float]

    for step in range(num_steps):
        if not is_agent_valid_at_tick(scenario, agent_id, step):
            continue

        in_intersection_now = is_agent_in_intersection_at_tick(scenario, intersections, step, agent_id)

        # Use object heading as entry vector if no prior step outside of the intersection exists
        if in_intersection_now and not in_intersection_before and not is_agent_valid_at_tick(scenario, agent_id, step - 1):
            in_intersection_before = True
            step_enter_intersection = step

            # Get agent heading in degrees
            agent_heading_angle = ((90 - math.degrees(agent_track.states[step].heading)) + 360) % 360

            # Entry vector with origin agent bounding box center
            entry_vec = [math.sin(math.radians(agent_heading_angle)), math.cos(math.radians(agent_heading_angle))]

        # Cases where we assign the last two points of the lane before the intersection as entry vector
        elif in_intersection_now and not in_intersection_before:
            # To find the lane before the intersection we consider the not just one step before entering the
            # intersection but rather a few steps further back to avoid mapping the wrong lane
            lane_assignment_range = range(step - 5, step)

            prior_valid_step = next((i for i, state in enumerate(agent_track.states)
                                     if state.valid and i in lane_assignment_range), None)

            if prior_valid_step is None:
                continue

            in_intersection_before = True
            step_enter_intersection = step

            lane_id_before_intersection = get_agent_lane_center(scenario, prior_valid_step, agent_id, kd_tree)[1]
            lane_before_intersection: LaneCenter = next(lane for idx, lane in lane_centers
                                                        if idx == lane_id_before_intersection)

            lane_before_poly = lane_before_intersection.polyline

            lane_before_second_last_point = (
                lane_before_poly[len(lane_before_poly) - 2].x, lane_before_poly[len(lane_before_poly) - 2].y
            )
            lane_before_last_point = (
                lane_before_poly[len(lane_before_poly) - 1].x, lane_before_poly[len(lane_before_poly) - 1].y
            )

            entry_vec = [
                lane_before_last_point[0] - lane_before_second_last_point[0],
                lane_before_last_point[1] - lane_before_second_last_point[1]
            ]

        # Compare entry vector with agent heading angle as exit vector if no exit lane exists
        elif in_intersection_now and in_intersection_before and not has_future_valid_steps(scenario, agent_id, step):
            in_intersection_before = False

            agent_heading_angle = ((90 - math.degrees(agent_track.states[step].heading)) + 360) % 360

            exit_vec = [math.sin(math.radians(agent_heading_angle)), math.cos(math.radians(agent_heading_angle))]

            if is_turn(entry_vec, exit_vec, turn_type, agent_id, step, turn_type):
                for i in range(step_enter_intersection, step+1):
                    steps_turn.append(i)

        elif not in_intersection_now and in_intersection_before:
            in_intersection_before = False

            lane_id_after_intersection = get_agent_lane_center(scenario, step, agent_id, kd_tree)[1]

            lane_after_intersection: LaneCenter = next(lane for idx, lane in lane_centers
                                                       if idx == lane_id_after_intersection)

            lane_after_poly = lane_after_intersection.polyline

            lane_after_first_point = (lane_after_poly[0].x, lane_after_poly[0].y)
            lane_after_second_point = (lane_after_poly[1].x, lane_after_poly[1].y)

            lane_after_vec = [
                lane_after_second_point[0] - lane_after_first_point[0],
                lane_after_second_point[1] - lane_after_first_point[1]
            ]

            if is_turn(entry_vec, lane_after_vec, turn_type, agent_id, step, turn_type):
                for i in range(step_enter_intersection, step + 1):
                    steps_turn.append(i)

    return steps_turn


def has_prior_valid_steps(scenario: Scenario, agent_id: int, step: int) -> bool:
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)

    return next((True for i, state in enumerate(agent_track.states)
                 if state.valid and i in range(0, step)), False)


def has_future_valid_steps(scenario: Scenario, agent_id: int, step: int) -> bool:
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    return next((True for i, state in enumerate(agent_track.states)
                 if state.valid and i in range(step + 1, num_steps)), False)


def is_turn(entry_vec, exit_vec, turn_type: TurnTypes, agent_id, step, type) -> bool:
    # Angle (deg) between the entry and exit vector
    vec_angle_deg_entry_exit = np.degrees(np.math.atan2(np.linalg.det([exit_vec, entry_vec]),
                                                        np.dot(exit_vec, entry_vec)))
    # Angle (deg) between the up vector (0, 1) and the entry vector
    vec_angle_up_entry = np.degrees(np.math.atan2(np.linalg.det([entry_vec, (0, 1)]),
                                                  np.dot(entry_vec, (0, 1))))
    # Angle (deg) between the up vector (0, 1) and the exit vector
    vec_angle_up_exit = np.degrees(np.math.atan2(np.linalg.det([exit_vec, (0, 1)]),
                                                 np.dot(exit_vec, (0, 1))))

    # We want to compare two non negative angle values where the exit angle is greater
    if vec_angle_up_entry < 0:
        vec_angle_up_entry += 360
    if vec_angle_up_exit < 0:
        vec_angle_up_exit += 360
    if vec_angle_up_exit < vec_angle_up_entry:
        vec_angle_up_exit += 360

    angle_dif_comparison = vec_angle_up_exit - vec_angle_up_entry > 180 if turn_type == TurnTypes.LEFT_TURN \
        else vec_angle_up_exit - vec_angle_up_entry < 180

    exceeds_angle_threshold = vec_angle_deg_entry_exit <= LEFT_TURN_VEC_THRESHOLD if turn_type == TurnTypes.LEFT_TURN \
        else vec_angle_deg_entry_exit >= RIGHT_TURN_VEC_THRESHOLD

    if angle_dif_comparison and exceeds_angle_threshold \
            and vec_angle_deg_entry_exit <= 180 - U_TURN_VEC_THRESHOLD:
        return True

    return False
