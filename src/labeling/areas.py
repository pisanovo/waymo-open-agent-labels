from typing import Union

import numpy as np
from waymo_open_dataset.protos.map_pb2 import LaneCenter, MapPoint, MapFeature
from waymo_open_dataset.protos.scenario_pb2 import Scenario

from config import LIN_INTERPOLATE_POINTS_NUM, LANES_MAX_ALTITUDE_DIF
from src.utils import maths


def pairwise(lst):
    return zip(lst[1:], lst)


def merge_sub_lists(lst: list, index_a, index_b):
    """

    Args:
        lst: original list
        index_a: index of first sub list
        index_b: index of second sub list

    Returns: A list containing the merged two sublists

    """
    lst[index_a] = lst[index_a] + lst[index_b]
    del lst[index_b]
    return lst


def get_element_array_index_or_none(data, search):
    """

    Args:
        data: list input
        search: element to be searched

    Returns: The index where the element occurs in the list containing sublists (max. depth 1 sublist) or none if not
    found

    """
    for i, e in enumerate(data):
        try:
            return (i, e.index(search))[0]
        except ValueError:
            pass
    return None


def get_lane_center_from_ids(scenario: Scenario, idx: Union[int, list[int]]):
    """

    Args:
        scenario: Scenario to be used
        idx: lane center ID(s)

    Returns: List of lane centers (type) from list of lane center ids

    """
    if type(idx) == int:
        return [getattr(map_feature, "lane") for map_feature in scenario.map_features if map_feature.id == idx][0]
    elif type(idx) == list[int]:
        return [getattr(map_feature, "lane") for map_feature in scenario.map_features if map_feature.id == idx]


# If changes are made to the implementation cached files under /output/labels/areas must be deleted
def get_intersections_and_lane_changes(scenario: Scenario) -> (list[list[int]], list[list[int]]):
    feature_lane_centers: list[MapFeature] = [map_feature for map_feature in scenario.map_features
                                              if map_feature.WhichOneof("feature_data") == "lane"]

    # Get all scene lane centers that are not bike lanes
    lane_centers: list[(int, LaneCenter)] = [(map_feature.id, getattr(map_feature, "lane"))
                                             for map_feature in scenario.map_features
                                             if map_feature.WhichOneof("feature_data") == "lane"
                                             and getattr(map_feature,"lane").type != LaneCenter.LaneType.TYPE_BIKE_LANE]

    # Initial state: Each lane center is an intersection of size one
    intersections = [[lane_center_feature.id] for lane_center_feature in feature_lane_centers]

    # Finding intersections: Phase I
    for i_lane in lane_centers:
        lane_id, lane = i_lane
        for i_other_lane in lane_centers:
            other_lane_id, other_lane = i_other_lane

            # Intersection algorithm takes two lines with LIN_INTERPOLATE_POINTS_NUM - 1 segments
            indices_lane = np.linspace(0, len(lane.polyline) - 1, LIN_INTERPOLATE_POINTS_NUM, dtype="int")
            indices_other_lane = np.linspace(0, len(other_lane.polyline) - 1, LIN_INTERPOLATE_POINTS_NUM, dtype="int")

            # The points of the line segments
            points_seg_lane: list[MapPoint] = [lane.polyline[index] for index in indices_lane]
            points_seg_other_lane: list[MapPoint] = [other_lane.polyline[index] for index in indices_other_lane]

            # Iterate over each segment and compare with every segment of other line
            for lane_seg_start, lane_seg_end in pairwise(points_seg_lane):
                for other_lane_seg_start, other_lane_seg_end in pairwise(points_seg_other_lane):
                    intersect = maths.intersection_poly_lines(lane_seg_start, lane_seg_end,
                                                              other_lane_seg_start, other_lane_seg_end)

                    alt_lane_seg = 1/2 * (lane_seg_start.z + lane_seg_end.z)
                    alt_other_lane_seg = 1/2 * (other_lane_seg_start.z + other_lane_seg_end.z)

                    if intersect and abs(alt_lane_seg - alt_other_lane_seg) <= LANES_MAX_ALTITUDE_DIF:
                        index_lane_id = get_element_array_index_or_none(intersections, lane_id)
                        index_other_lane_id = get_element_array_index_or_none(intersections, other_lane_id)

                        if index_lane_id != index_other_lane_id:
                            # Merge two intersections to a larger intersection
                            merge_sub_lists(intersections, index_lane_id, index_other_lane_id)

                        break

    lane_centers: list[list[int]] = [intersection for intersection in intersections if len(intersection) == 1]

    # Finding lane change areas
    lane_changes = []

    for lane_arr in lane_centers:
        lane_id = lane_arr[0]
        lane_center: LaneCenter = get_lane_center_from_ids(scenario, lane_id)

        exit_lanes = [x for x in lane_center.exit_lanes if x != []]

        if len(exit_lanes) > 1:
            contains_intersections = False

            for exit_lane in exit_lanes:
                index = get_element_array_index_or_none(intersections, exit_lane)
                if index is not None and len(intersections[index]) > 1:
                    contains_intersections = True

            if not contains_intersections:
                lane_changes.extend([exit_lanes])

    # Only consider sets with at least two lane centers as an intersection
    intersections: list[list[int]] = [intersection for intersection in intersections if len(intersection) > 1]

    # Finding intersections: Phase II. Intersections from phase I may be still missing lane centers
    for intersection in intersections:
        intersection_lane_centers: list[LaneCenter] = [getattr(map_feature, "lane")
                                                       for map_feature in scenario.map_features
                                                       if map_feature.id in intersection]

        intersection_exit_lanes = [road_center.exit_lanes for road_center in intersection_lane_centers]
        intersection_entry_lanes = [road_center.entry_lanes for road_center in intersection_lane_centers]

        # Remove [] entries, e.g., [[302], [], [133]]
        intersection_exit_lanes = [x for x in intersection_exit_lanes if x != []]
        intersection_entry_lanes = [x for x in intersection_entry_lanes if x != []]

        for single_lane_arr in lane_centers:
            single_lane_id = single_lane_arr[0]
            single_lane: LaneCenter = get_lane_center_from_ids(scenario, single_lane_id)

            # Check if the one center lane shares entry or exit lanes with any of the center lanes in the intersection
            if single_lane.entry_lanes in intersection_entry_lanes or single_lane.exit_lanes in intersection_exit_lanes:
                index_intersection_id = get_element_array_index_or_none(intersections, intersection[0])

                if index_intersection_id is not None:
                    # merge_sub_lists(intersections, index_intersection_id, index_lane_id)
                    intersections[index_intersection_id].append(single_lane_id)
                    lane_centers.remove(single_lane_arr)

    return intersections, lane_changes

