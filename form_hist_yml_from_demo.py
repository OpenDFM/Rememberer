#!/usr/bin/python3

import vh_to_html
import lxml.etree
import lxml.html
import yaml

import pickle as pkl
import os
import os.path
import functools
import numpy as np

from typing import Dict, List, Tuple
from typing import Union, Any

from android_env.proto.task_pb2 import Task
from google.protobuf import text_format
from android_env.components.action_type import ActionType

demo_directory = "llmdemo-data/"
demo_list: List[str] = list( map( lambda f: os.path.join( demo_directory
                                                        , f
                                                        )
                                , filter( lambda f: f.endswith(".pkl.prs")
                                        , os.listdir(demo_directory)
                                        )
                                )
                           )
demo_directory = "llmdemo-data-extra.out/"
demo_list += map( lambda f: os.path.join( demo_directory
                                        , f
                                        )
                , filter( lambda f: f.endswith(".pkl")
                        , os.listdir(demo_directory)
                        )
                )

#vh_directory = "llmdemo-vh/"

task_definition_directory = "../android_env/apps/wikihow/templates.miniout"

Key = Tuple[str, str, str]
ActionDict = Dict[str, Dict[str, Union[int, float]]]
history_record: Dict[Key, Dict[str, Any]] = {}

with open("llmdemo-vh/step_0.html") as f:
    init_html: str = f.read()
init_html = init_html.split("\n\n")[0]

#for dem in ["llmdemo-data-extra.out/clean_utensils-7.pkl"]:
for dem in demo_list:
    with open(dem, "rb") as f:
        record_dict: Dict[str, Dict[str, Any]] = pkl.load(f)

    if "command" in record_dict["meta"]:
        task: List[str] = record_dict["meta"]["command"]
    else:
        task_definition_id: str = record_dict["meta"]["task_definition_id"]
        with open( os.path.join( task_definition_directory
                               , task_definition_id + ".textproto"
                               )
                 ) as f:
            textproto: str = f.read()
        task_definition = Task()
        text_format.Parse(textproto, task_definition)

        task: List[str] = list(task_definition.command)
    step0: str = task[0]
    task: str = "\n".join(task)

    # first step
    action: str = "INPUT(2, {:})".format(step0[34:-1])
    history_record[ ( init_html
                    , task
                    , ""
                    )
                  ] = { "other_info": { "action_history": []
                                      , "last_reward": 0.
                                      , "total_reward": 0.
                                      , "number": 1
                                      }
                      , "action_dict": {
                          action: {
                              "reward": 1.0
                            , "number": 1
                            }
                        }
                      }

    action_history: List[str] = [action]
    last_reward = 1.
    total_reward = 1.

    # succeeding step
    trajectory: List[Dict[str, Any]] = record_dict["trajectories"][0]
    for i, st in enumerate(trajectory):
        if "instruction" not in st\
                or st["instruction"] is None\
                or len(st["instruction"])==0:
            continue

        error_information = "{:} No.{:d}".format(dem, i)

        instruction: str = "\n".join(st["instruction"])

        if "view_hierarchy" in st\
                and st["view_hierarchy"] is not None:
            v: int = i
        elif "view_hierarchy" in trajectory[i-1]\
                and trajectory[i-1]["view_hierarchy"] is not None:
            v: int = i-1
        view_hierarchy: lxml.etree.Element = lxml.etree.fromstring(trajectory[v]["view_hierarchy"])
        html_list: List[lxml.html.Element]
        bbox_list: List[List[int]]
        html_list, bbox_list = vh_to_html.convert_tree(view_hierarchy)

        screen: str = "".join( map( functools.partial( lxml.html.tostring
                                                     , pretty_print=True
                                                     , encoding="unicode"
                                                     )
                                  , html_list
                                  )
                             ).strip()

        # find TOUCH
        found_touch = False
        for j in range(i, i+5):
            if trajectory[j]["action_type"]==ActionType.TOUCH:
                found_touch = True
                break

        assert found_touch, error_information
        touch_position: np.ndarray = trajectory[j]["touch_position"]
        height: int
        width: int
        height, width, _ = trajectory[j]["observation"].shape
        touch_position[0] *= width
        touch_position[1] *= height
        #print(height, width, touch_position)

        # find bbox
        found_bbox = False
        found_index = -1
        min_area = 2073600 # 1920*1080
        clickable_distances: List[np.float32] = []
        clickable_indices: List[int] = []
        for elm_id, (bb, html) in enumerate(zip(bbox_list, html_list)):
            if html.get("clickable")=="true":
                    #and touch_position[0]>=bb[0]\
                    #and touch_position[0]<=bb[2]\
                    #and touch_position[1]>=bb[1]\
                    #and touch_position[1]<=bb[3]:
                #found_bbox = True
                #area: int = (bb[3]-bb[1]) * (bb[2]-bb[0])
                #print(elm_id, bb, area)
                #if area<min_area:
                    #found_index = elm_id
                    #min_area = area
                distance: np.float32 = min( touch_position[0]-bb[0]
                                          , bb[2]-touch_position[0]
                                          , touch_position[1]-bb[1]
                                          , bb[3]-touch_position[1]
                                          )

                clickable_indices.append(elm_id)
                clickable_distances.append(distance)

        error_information += "-{:d}".format(j)
        #assert found_bbox and found_index!=-1, error_information

        clickable_distances = np.asarray(clickable_distances)
        found_index: int = clickable_indices[np.argmax(clickable_distances)]
        #print(found_index, bbox_list[found_index], lxml.html.tostring(html_list[found_index], encoding="unicode"))
        action = "CLICK({:d})".format(found_index)

        # find reward
        found_reward = False
        #assert i<j, error_information
        for k in range(j, i+5):
            if "reward" in trajectory[k]\
                    and trajectory[k]["reward"] > 0.:
                found_reward = True
                break
        if not found_reward:
            reward = 0.
            action = "SCROLL(DOWN)"
        else:
            reward = 1.

        history_record[ ( screen
                        , task
                        , instruction
                        )
                      ] = { "other_info": { "action_history": action_history.copy()
                                          , "last_reward": last_reward
                                          , "total_reward": total_reward
                                          }
                          , "action_dict": {
                              action: { "reward": reward
                                      , "number": 1
                                      }
                            }
                          }

        action_history.append(action)
        last_reward += 1.
        total_reward += 1.

#print(len(history_record))
with open("history-pools/annotated_pool-auto.yaml", "w") as f:
    yaml.dump(history_record, f, Dumper=yaml.Dumper, allow_unicode=True)
