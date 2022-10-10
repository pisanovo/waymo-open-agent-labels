import json
import logging
import sys
from pathlib import Path
import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import os
from config import PROJECT_DIR, AGENT_LABELS_DIR
from src.labeling.agents.timelines import generate_agent_merged_timeline
from src.utils import usage
from src.utils.misc import get_scenario_lane_center_kd_tree

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] - %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
logging.addLevelName(logging.ERROR, "\033[1;41m%s\033[1;0m" % logging.getLevelName(logging.ERROR))

logging.getLogger("moviepy").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

FILENAME = f"{PROJECT_DIR}/data_records/u_scenario_training_00000.tfrecord"
AGENT_LABELS_DIR = f"{PROJECT_DIR}/{AGENT_LABELS_DIR}"

# Print memory usage before start
usage.print_mem_usage()

logger.info(f"Start loading data from {FILENAME}")

raw_dataset = tf.data.TFRecordDataset([FILENAME])
scenario_data = []

start = 1
end = 446

# Load scenarios
for i, data in enumerate(raw_dataset):
    if i+1 not in range(start, end+1):
        continue

    proto_string = data.numpy()
    proto = scenario_pb2.Scenario()
    proto.ParseFromString(proto_string)
    scenario_data.append(proto)

logger.info(f"Loaded {len(scenario_data)} scenarios")

# Print memory usage after loading scenarios
usage.print_mem_usage()

for i, scenario in enumerate(scenario_data):
    num = start + i
    kd_tree = get_scenario_lane_center_kd_tree(scenario)

    logger.info(f"[{num}#446] Creating agent labels for scenario {scenario.scenario_id}")

    try:
        agents_labels = []
        agents_labels_for_export = []

        for track in scenario.tracks:
            if track.object_type != 3:
                merged_timeline = generate_agent_merged_timeline(scenario, track.id, kd_tree)
                agents_labels.append((track.id, merged_timeline))

                merged_timeline_for_export = []

                current_direction_label, current_speed_label = merged_timeline[0][1]

                priority_labels_values = {
                    "timestep": merged_timeline[0][0],
                    "direction_label": current_direction_label.name,
                    "speed_label": current_speed_label.name
                }
                merged_timeline_for_export.append(priority_labels_values)

                for j in range(1, len(merged_timeline)):
                    timestep = merged_timeline[j]
                    st = timestep[0]
                    priority_labels = timestep[1]
                    direction_label, speed_label = priority_labels

                    if current_direction_label.name != direction_label.name or current_speed_label.name != speed_label.name:
                        priority_labels_values = {
                            "timestep": st,
                            "direction_label": direction_label.name,
                            "speed_label": speed_label.name
                        }
                        merged_timeline_for_export.append(priority_labels_values)
                        current_direction_label = direction_label
                        current_speed_label = speed_label

                agents_labels_for_export.append({
                    "agent_id": track.id,
                    "labels": merged_timeline_for_export
                })

        json_dump = json.dumps(
            {"scenario_id": scenario.scenario_id,
             "data": agents_labels_for_export},
            indent=2
        )

        folder_path = AGENT_LABELS_DIR + f"/submission_v1/"
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        data_path = f"{folder_path}/446_{num}_{scenario.scenario_id}.json"
        with open(data_path, 'w') as file:
            file.write(json_dump)
    except Exception:
        logger.error(f"[{num}#446] Could not create agent labels for scenario {scenario.scenario_id}")

logger.info(f"Done.")
