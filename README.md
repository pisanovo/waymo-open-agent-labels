# Rule Based Labeling of Traffic Agents
This repository provides an extension to the Waymo Open Motion Dataset by including
agent labels.

## Scope
Following base label types are currently supported:
- Left/Right Turn
- Stopped
- Parked
- Left/Right Lane Change
- Accelerate
- Slowdown

Following composite label types are currently supported:
- Turn + Lane Change (e.g. Left Turn Right Lane Change)

If none of the above labels are assigned the following label type is 
assigned instead:
- No Change

## Installation
- Please follow the quick-start guide on 
https://github.com/waymo-research/waymo-open-dataset for the Motion dataset.
- Install remaining missing packages
- Download the dataset from https://www.waymo.com/open into the `data_records` 
folder (This project uses data provided as .tfrecord files)

## Usage
- `create_agent_labels.py` is used to generate and save agent labels
- `scenario_main.py` is used to generate scenario videos showing agent labels

## Reference
TODO: Add paper ref