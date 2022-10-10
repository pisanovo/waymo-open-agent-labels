import json
import os

from config import PROJECT_DIR, AGENT_LABELS_DIR

AGENT_LABELS_DIR = f"{PROJECT_DIR}/{AGENT_LABELS_DIR}"

dirA = "testing"
dirB = "testing2"

# Can be used to compare the outputs of two different agent labeling versions in order to detect changes
for filename in os.listdir(f"{AGENT_LABELS_DIR}/{dirA}"):
    pathA = os.path.join(f"{AGENT_LABELS_DIR}/{dirA}", filename)
    fA = open(pathA)
    dataA = json.load(fA)
    scenarioIdA = dataA["scenario_id"]

    for other_filename in os.listdir(f"{AGENT_LABELS_DIR}/{dirB}"):
        pathB = os.path.join(f"{AGENT_LABELS_DIR}/{dirB}", other_filename)
        fB = open(pathB)
        dataB = json.load(fB)
        scenarioIdB = dataB["scenario_id"]

        if scenarioIdA == scenarioIdB:
            agentIdsA = set([agentA["agent_id"] for agentA in dataA["data"]])
            agentIdsB = set([agentB["agent_id"] for agentB in dataB["data"]])

            if agentIdsA != agentIdsB:
                distinctIds = agentIdsA.difference(agentIdsB).union(agentIdsB.difference(agentIdsA))
                print(f"!!! Agent IDs do not match. Differences: {distinctIds}. Continuing with common elements...")

            commonIds = list(agentIdsA.intersection(agentIdsB))

            for agentId in commonIds:
                agentDictA = next(agent for agent in dataA["data"] if agent["agent_id"] == agentId)
                agentDictB = next(agent for agent in dataB["data"] if agent["agent_id"] == agentId)
                agentLabelsA = agentDictA["labels"]
                agentLabelsB = agentDictB["labels"]

                if json.dumps(agentLabelsA) != json.dumps(agentLabelsB):
                    print(f"Found mismatch in {scenarioIdA} for agent ID: {agentId}")


