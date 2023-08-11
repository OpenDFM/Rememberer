#!/usr/bin/python3

import sys
sys.path.append("..")

#import history
import yaml

from typing import TypeVar, Union
from typing import List, Dict

Key = TypeVar("Key")
Action = TypeVar("Action")
InfoDict = Dict[ str
               , Union[ float
                      , int
                      , List[Action]
                      ]
               ]
ActionDict = Dict[ Action
                 , Dict[ str
                       , Union[int, float]
                       ]
                 ]
Record = Dict[str, Union[InfoDict, ActionDict, int]]

input_pools: List[str] = sys.argv[1:]

if len(input_pools)>=2:
    #  Merge Two History Pools {{{ # 
    with open(input_pools[0]) as f:
        pool1: Dict[Key, Record] = yaml.load(f, Loader=yaml.Loader)
    with open(input_pools[1]) as f:
        pool2: Dict[Key, Record] = yaml.load(f, Loader=yaml.Loader)

    for k in pool2:
        if k in pool1:
            # merge other_info
            other_info1: InfoDict = pool1[k]["other_info"]
            other_info2: InfoDict = pool2[k]["other_info"]
            other_info1["number"] = (other_info1["number"]+other_info2["number"]) // 2
            other_info1["last_reward"] = (other_info1["last_reward"]+other_info2["last_reward"]) / 2.
            other_info1["total_reward"] = (other_info1["total_reward"]+other_info2["total_reward"]) / 2.

            # merge action_dict
            action_dict1: ActionDict = pool1[k]["action_dict"]
            action_dict2: ActionDict = pool2[k]["action_dict"]
            for act in action_dict2:
                if act in action_dict1:
                    action_dict1[act]["reward"] =\
                            (action_dict1[act]["reward"] + action_dict2[act]["reward"]) / 2.
                    action_dict1[act]["qvalue"] =\
                            (action_dict1[act]["qvalue"] + action_dict2[act]["qvalue"]) / 2.
                    action_dict1[act]["number"] =\
                            (action_dict1[act]["number"] + action_dict2[act]["number"]) // 2
                else:
                    action_dict1[act] = action_dict2[act]
        else:
            pool1[k] = pool2[k]

    pool: Dict[Key, Record] = pool1
    #  }}} Merge Two History Pools # 
else:
    with open(input_pools[0]) as f:
        pool: Dict[Key, Record] = yaml.load(f, Loader=yaml.Loader)

sum_reward = 0.
nb_rewards = 0
for rcd in pool.values():
    total_reward: float = rcd["other_info"]["total_reward"]
    q_value: float = max( map( lambda act_d: act_d["qvalue"]
                             , rcd["action_dict"].values()
                             )
                        )
    sum_reward += total_reward+q_value
    nb_rewards += 1

print(sum_reward/nb_rewards)
