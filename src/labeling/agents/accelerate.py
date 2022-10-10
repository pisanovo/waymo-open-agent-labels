from scipy.ndimage import gaussian_filter1d
from waymo_open_dataset.protos.scenario_pb2 import Scenario, Track

from src.utils.maths import get_agent_speed


def generate_agent_accelerate_timeline(scenario: Scenario, agent_id: int) -> list[int]:
    """

    Args:
        scenario: The scenario to be used
        agent_id: Id for the agent

    Returns: An array containing the ticks for which the agent received the label 'accelerate'

    """
    agent_track: Track = next(track for track in scenario.tracks if track.id == agent_id)
    num_steps = len(agent_track.states)

    steps_accelerate = []

    step_list = []
    speed_list = []
    for step in range(num_steps):
        if agent_track.states[step].valid:
            step_list.append(step)
            speed_list.append(get_agent_speed(agent_track, step))
    # Apply smoothing to the speed values since raw data contains inaccuracies
    speed_list_smooth = gaussian_filter1d(speed_list, sigma=7)

    for_steps_accelerate = 0
    speed_step_before = -1
    potential_accelerate_steps = []

    for i, (step, speed) in enumerate(zip(step_list, speed_list)):
        if i == 0:
            speed_step_before = step
            speed_now = speed_list_smooth[step_list.index(step)]
            if speed_now > 2.0:
                potential_accelerate_steps.append(step)

        speed_before = speed_list_smooth[step_list.index(speed_step_before)]
        speed_now = speed_list_smooth[step_list.index(step)]

        # Speed threshold
        if speed_before < speed_now and speed_now > 2.0:
            for_steps_accelerate += 1
            potential_accelerate_steps.append(step)
            if i == len(step_list) - 1:
                steps_accelerate.extend(potential_accelerate_steps)
        else:
            # Number of steps threshold
            if for_steps_accelerate >= 5:
                steps_accelerate.extend(potential_accelerate_steps)
                for_steps_accelerate = 0
                speed_step_before = step
                potential_accelerate_steps = []

    steps_accelerate = list(set(steps_accelerate))

    return steps_accelerate
