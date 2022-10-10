from scipy.ndimage import gaussian_filter1d
from scipy.spatial import KDTree
from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track

from src.utils.maths import get_agent_speed
from src.utils.misc import touches_lane_at_tick


def generate_agent_stopped_timeline(scenario: Scenario, agent_id: int, kd_tree: KDTree) -> list[int]:
    """

    Args:
        scenario: The scenario to be used
        agent_id: Id for the agent

    Returns: An array containing the ticks for which the agent received the label 'stopped'

    """

    # algorithm idea:
    # test if agent is moving at all in the 91 time steps
    # if no, stopped iff touching lane
    # if yes, agent is stopped while not moving for longer than 3 ticks

    agent_track: Track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    steps_stopped = []

    step_list = []
    speed_list = []
    for step in range(num_steps):
        if agent_track.states[step].valid:
            step_list.append(step)
            speed_list.append(get_agent_speed(agent_track, step))
    speed_list_smooth = gaussian_filter1d(speed_list, sigma=7)

    for_steps_stopped = 0
    potential_stopped_steps = []

    for i, (step, speed) in enumerate(zip(step_list, speed_list)):
        if i == 0:
            potential_stopped_steps.append(step)

        speed_now = speed_list_smooth[step_list.index(step)]

        if speed_now < 0.01 and touches_lane_at_tick(scenario, step, agent_id, kd_tree):
            for_steps_stopped += 1
            potential_stopped_steps.append(step)
            # if scenario ends, add all steps where the agents has been stopped
            if i == len(step_list) - 1:
                steps_stopped.extend(potential_stopped_steps)
        else:
            # if agent has been stopped for at least 3 timesteps and starts driving now, add these steps to the list
            if for_steps_stopped >= 3:
                steps_stopped.extend(potential_stopped_steps)
                for_steps_stopped = 0
                potential_stopped_steps = []

    steps_stopped = list(set(steps_stopped))

    return steps_stopped