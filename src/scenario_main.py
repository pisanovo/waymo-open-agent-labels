import logging
import sys
import warnings
import tensorflow as tf
import os
from numpy import VisibleDeprecationWarning
from waymo_open_dataset.protos import scenario_pb2
from config import PROJECT_DIR, CANVAS_SIZE_PIXELS
from src.generate import generate_video
from src.generate.generate_image import AnnotateType, visualize_all_agents_smooth, visualize_all_agents_smooth_one_step
from src.labeling.agents.timelines import generate_agent_merged_timeline
from src.utils import usage
from src.utils.misc import get_scenario_lane_center_kd_tree

# Fix memory usage (out-of-memory errors)
# matplotlib.use('Agg')
# matplotlib.interactive(False)

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

# Waymo template contains deprecated numpy calls
warnings.filterwarnings("error", category=VisibleDeprecationWarning)
# MoviePy memory info not necessary
logging.getLogger("moviepy").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.CRITICAL)

logger = logging.getLogger(__name__)

# .tfrecord data file path
FILENAME = f"{PROJECT_DIR}/data_records/u_scenario_training_00000.tfrecord"

# Print memory usage before start
usage.print_mem_usage()

logger.info(f"Start loading data from {FILENAME}")

raw_dataset = tf.data.TFRecordDataset([FILENAME])
scenario_data = []

# Scenario number start
start = 41
# Scenario number end
end = 100

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
    # Setup KDTree used to, e.g., calculate the closest lane for an agent at step t
    kd_tree = get_scenario_lane_center_kd_tree(scenario)

    logger.info(f"[{start + i}#446] Visualizing agents for scenario {scenario.scenario_id}")

    # try:
        # Generate labels and video
    agents_labels = [(track.id, generate_agent_merged_timeline(scenario, track.id, kd_tree)) for track in scenario.tracks if track.object_type != 3 and track.object_type != 2]
    images = visualize_all_agents_smooth(scenario, AnnotateType.WITH_ID_AND_SPEED, agents_labels, kd_tree, size_pixels=CANVAS_SIZE_PIXELS, white=True)
    generate_video.fast_create_video(
        images,
        folder=f"docs/scene_videos/label_rewrite_3_white",
        video_name=f"[{start + i}#446] {scenario.scenario_id} (agent_labels)"
    )

        # Use below code to only generate one frame image
        # agents_labels = []
        # visualize_all_agents_smooth_one_step(scenario, AnnotateType.WITH_ID_AND_SPEED, [], kd_tree, size_pixels=3000, step=22,
        #                                      white=True)
    # except Exception as e:
    #     logger.error(f"[{start + i}#446] Could not create agent labels for scenario {scenario.scenario_id}")
    #     logger.error(e.)

logger.info(f"Done.")
