from scipy.spatial import KDTree
from waymo_open_dataset.protos.scenario_pb2 import Scenario

from src.datatypes.driving_states import DrivingState
from src.labeling.agents.lane_change import generate_agent_lane_change_timeline, LaneChangeTypes
from src.labeling.agents.slowdown import generate_agent_slowdown_timeline
from src.labeling.agents.accelerate import generate_agent_accelerate_timeline
from src.labeling.agents.stopped import generate_agent_stopped_timeline
from src.labeling.agents.parked import generate_agent_parked_timeline
from src.labeling.agents.turn import generate_turn_timeline, TurnTypes


def generate_agent_merged_timeline(scenario: Scenario, agent_id: int, kd_tree: KDTree) -> \
        list[(int, DrivingState, DrivingState)]:
    """

    Args:
        scenario: The scenario to be used
        agent_id: The agent for which the timeline is created
        kd_tree: KDTree that can be used to calculated the closest lane given a point

    Returns: Returns a list containing for each step the direction and speed label tuple

    """
    agent_track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    # Get agent labels
    label_left_turn = generate_turn_timeline(scenario, agent_id, kd_tree, TurnTypes.LEFT_TURN)
    label_right_turn = generate_turn_timeline(scenario, agent_id, kd_tree, TurnTypes.RIGHT_TURN)
    label_accelerate = generate_agent_accelerate_timeline(scenario, agent_id)
    label_slowdown = generate_agent_slowdown_timeline(scenario, agent_id)
    label_stopped = generate_agent_stopped_timeline(scenario, agent_id, kd_tree)
    label_parked = generate_agent_parked_timeline(scenario, agent_id, kd_tree)
    label_left_lane_change = generate_agent_lane_change_timeline(scenario, agent_id, kd_tree,
                                                                 LaneChangeTypes.LEFT_LANE_CHANGE)
    label_right_lane_change = generate_agent_lane_change_timeline(scenario, agent_id, kd_tree,
                                                                  LaneChangeTypes.RIGHT_LANE_CHANGE)

    label_merged = []
    for step in range(num_steps):
        all_labels = []

        # Generate a list that contains for each step all labels that where found for that step
        if step in label_left_turn:
            all_labels.append(DrivingState.LEFT_TURN)
        if step in label_right_turn:
            all_labels.append(DrivingState.RIGHT_TURN)
        if step in label_accelerate:
            all_labels.append(DrivingState.ACCELERATE)
        if step in label_slowdown:
            all_labels.append(DrivingState.SLOW_DOWN)
        if step in label_stopped:
            all_labels.append(DrivingState.STOPPED)
        if step in label_parked:
            all_labels.append(DrivingState.PARKED)
        if step in label_left_lane_change:
            all_labels.append(DrivingState.LANE_CHANGE_LEFT)
        if step in label_right_lane_change:
            all_labels.append(DrivingState.LANE_CHANGE_RIGHT)

        # Identify the final labels to be used, a hierarchy is used
        if step in label_parked:
            priority_labels = (DrivingState.PARKED, DrivingState.NO_CHANGE)
        else:
            if step in label_left_turn and step in label_left_lane_change:
                direction_label = DrivingState.LEFT_TURN_LANE_CHANGE_LEFT
            elif step in label_left_turn and step in label_right_lane_change:
                direction_label = DrivingState.LEFT_TURN_LANE_CHANGE_RIGHT
            elif step in label_right_turn and step in label_left_lane_change:
                direction_label = DrivingState.RIGHT_TURN_LANE_CHANGE_LEFT
            elif step in label_right_turn and step in label_right_lane_change:
                direction_label = DrivingState.RIGHT_TURN_LANE_CHANGE_RIGHT
            elif step in label_left_turn:
                direction_label = DrivingState.LEFT_TURN
            elif step in label_right_turn:
                direction_label = DrivingState.RIGHT_TURN
            elif step in label_left_lane_change:
                direction_label = DrivingState.LANE_CHANGE_LEFT
            elif step in label_right_lane_change:
                direction_label = DrivingState.LANE_CHANGE_RIGHT
            else:
                direction_label = DrivingState.NO_CHANGE

            if step in label_stopped:
                speed_label = DrivingState.STOPPED
            elif step in label_accelerate:
                speed_label = DrivingState.ACCELERATE
            elif step in label_slowdown:
                speed_label = DrivingState.SLOW_DOWN
            else:
                speed_label = DrivingState.NO_CHANGE

            priority_labels = (direction_label, speed_label)

        item = (step, priority_labels, all_labels)

        if item:
            label_merged.append(item)

    return label_merged
