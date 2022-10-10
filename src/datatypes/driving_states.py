import dataclasses
from enum import Enum


@dataclasses.dataclass
class DrivingState(Enum):
    UNDEFINED = "?"
    KEEP_LANE = "kl"
    SLOW_DOWN = "✘"
    ACCELERATE = "+"
    STOPPED = "st"
    PARKED = "pa"
    LEFT_TURN = "←"
    RIGHT_TURN = "→"
    LANE_CHANGE_LEFT = "LCL"
    LANE_CHANGE_RIGHT = "LCR"
    LEFT_TURN_LANE_CHANGE_LEFT = "←//lcl"
    LEFT_TURN_LANE_CHANGE_RIGHT = "←//lcr"
    RIGHT_TURN_LANE_CHANGE_LEFT = "→//lcl"
    RIGHT_TURN_LANE_CHANGE_RIGHT = "→//lcr"
    NO_CHANGE = "~"



