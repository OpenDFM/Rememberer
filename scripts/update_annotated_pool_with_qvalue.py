#!/usr/bin/python3

import yaml
import history
from typing import Dict

input_yaml = "history-pools/annotated_pool-auto.yaml"
output_yaml = "history-pools/annotated_pool-auto.q.yaml"

with open(input_yaml) as f:
    pool: Dict[ history.HistoryReplay.Key
              , history.HistoryReplay.Record
              ] = yaml.load(f, Loader=yaml.Loader)

for k, rcd in pool.items():
    task: str = k[1]
    total_reward: int = len(task.splitlines())

    for act in rcd["action_dict"].values():
        act["qvalue"] = total_reward-rcd["other_info"]["total_reward"]


with open(output_yaml, "w") as f:
    yaml.dump(pool, f, Dumper=yaml.Dumper)
