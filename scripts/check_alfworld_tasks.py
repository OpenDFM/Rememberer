import alfworld.agents.environment as environment
#import alfworld.agents.modules.generic as generic
import yaml

from typing import Dict
from typing import Any

task_prefixes = { 'pick_and_place': 'put'
                , 'pick_clean_then_place': 'clean'
                , 'pick_heat_then_place': 'heat'
                , 'pick_cool_then_place': 'cool'
                , 'look_at_obj': 'examine'
                , 'pick_two_obj': 'puttwo'
                }

with open("alfworld_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.Loader)

split = "train"
env: environment.AlfredTWEnv = getattr(environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

for i in range(50):
    info: Dict[str, Any] = env.reset()[1]
    task_name: str = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

    for pr in task_prefixes:
        if task_name.startswith(pr):
            break

    print(i, task_prefixes[pr], task_name)

split = "eval_out_of_distribution"
env: environment.AlfredTWEnv = getattr(environment, config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

for i in range(134):
    info: Dict[str, Any] = env.reset()[1]
    task_name: str = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

    for pr in task_prefixes:
        if task_name.startswith(pr):
            break

    print(i, task_prefixes[pr], task_name)
