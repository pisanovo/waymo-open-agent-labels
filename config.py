import os

"""
System
"""
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = "output"
AREA_LABELS_DIR = "output/labels/areas"
AGENT_LABELS_DIR = "output/labels/agents"


"""
Video generation
"""
# When generating the scenario video, a batch of n smaller videos is created where n is the specified value.
# This improves runtime
NUM_THREADS = 16
# Frames per second in the scenario video
VIDEO_FPS = 4
# Video pixel size
CANVAS_SIZE_PIXELS = 2000
# Density
DPI = 200


"""
Image generation
"""
# Controls text size for agent boxes in images and videos
ANNOTATE_TEXT_SIZE = 5
# Colors for different agent types
COLOR_AUTONOMOUS_VEHICLE = "palegreen"
COLOR_PEDESTRIAN_CYCLIST = "wheat"
# Colors for different traffic signal states
TRAFFIC_SIGNAL_COLOR_STATE_UNKNOWN = "grey"
TRAFFIC_SIGNAL_COLOR_STATE_GO = "lime"
TRAFFIC_SIGNAL_COLOR_STATE_CAUTION = "gold"
TRAFFIC_SIGNAL_COLOR_STATE_STOP = "red"
# The seed used for lane and agent colors
COLOR_MAP_SEED = 8617739


"""
Lane labels
"""
# Used during detection of intersection areas. Reduces the number of segments each center lane has for the intersection
# comparison to the value specified
LIN_INTERPOLATE_POINTS_NUM = 5
# Used during detection of intersection areas. For two segments to intersect the height difference must not exceed
# the value specified
LANES_MAX_ALTITUDE_DIF = 2


"""
Agent labels
"""
# Threshold in degrees for a left turn between the entry and the exit vector
LEFT_TURN_VEC_THRESHOLD = -30
# Threshold in degrees for a right turn between the entry and the exit vector
RIGHT_TURN_VEC_THRESHOLD = 30
# A U-Turn occurs if the agent performed a maneuver that resulted in a heading change of 180 ± x degrees where x is the
# specified value
U_TURN_VEC_THRESHOLD = 15
