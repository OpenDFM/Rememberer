#!/usr/bin/python3
# Copyright 2023 SJTU X-Lance Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by Danyang Zhang @X-Lance.

import logging
import argparse

import os
import os.path
import sys
import yaml
import datetime
import string

import agent_protos
import wikihow_agent
import android_env
from android_env.wrappers import VhIoWrapper
from android_env.environment import AndroidEnv
from transformers import AutoTokenizer
import dm_env
import history
import itertools

from typing import Dict, List, Set
import numpy as np

import lxml.etree
from PIL import Image
import vh_to_html

def dump( path: str
        , step: int
        , command: str
        , screen: np.ndarray
        , view_hierarchy: lxml.etree.Element
        , instruction: str
        ):
    #  function dump {{{ # 
    if not os.path.exists(os.path.join(path, "command")):
        with open(os.path.join(path, "command"), "w") as f:
            f.write(command + "\n")

    image = Image.fromarray(screen)
    image.save(os.path.join(path, "screen.{:d}.jpg".format(step)))

    html_elements: List[lxml.html.Element] =\
            vh_to_html.convert_tree(view_hierarchy)[0]

    screen_representation: List[str] = []
    for html in html_elements:
        screen_representation.append( lxml.html.tostring( html
                                                        , pretty_print=True
                                                        , encoding="unicode"
                                                        )
                                    )
    screen_representation: str = "".join(screen_representation)

    with open(os.path.join(path, "view_hierarchy.{:d}".format(step)), "w") as f:
        f.write(screen_representation)

    with open(os.path.join(path, "instruction.{:d}".format(step)), "w") as f:
        f.write(instruction + "\n")
    #  }}} function dump # 

def traverse_environment( env: AndroidEnv
                        , model: wikihow_agent.Agent
                        , logger: logging.Logger
                        , except_list: Set[int] = set()
                        , max_nb_steps: int = 15
                        ) -> Set[int]:
    #  function traverse_environment {{{ # 
    """
    Args:
        env (AndroidEnv): the traversed environment
        model (wikihow_agent.Agent): the agent
        logger (logging.Logger): the logger
        except_list (Set[int]): tasks in this set won't be tested

        max_nb_steps (int): if the number of steps exceeds `max_nb_steps`, the
          episode will be killed and considered as failed.

    Returns:
        Set[int]: set of the succeeded tasks
    """

    success_list: Set[int] = set()

    nb_stepss: List[int] = []
    rewards: List[float] = []
    succeedss: List[int] = []
    for i in range(env.nb_tasks):
        if i in except_list:
            continue
        #os.makedirs(args.dump_path[_i], exist_ok=True)

        model.reset()
        step: dm_env.TimeStep = env.switch_task(i)
        command: str = "\n".join(env.command())
        instruction: str = env.task_instructions(latest_only=True)

        nb_steps = 0
        nb_nothing_steps = 0
        #dump( args.dump_path[_i], nb_steps, command
            #, step.observation["pixels"]
            #, step.observation["view_hierarchy"]
            #, instruction
            #)

        reward: float = step.reward
        succeeds: bool = True
        while not step.last():
            action: Dict[str, np.ndarray]\
                    = model( command
                           , step.observation["view_hierarchy"]
                           , instruction
                           , step.reward
                           , reward
                           )
            step = env.step(action)
            if len(env.task_instructions())>0:
                instruction = env.task_instructions(latest_only=True)
            reward += step.reward

            if action["action_type"]==VhIoWrapper.ActionType.NOTHING\
                    and "records" in action\
                    and not action["records"]:
                nb_nothing_steps += 1
            else:
                nb_steps += 1
                #dump( args.dump_path[_i], nb_steps, command
                    #, step.observation["pixels"]
                    #, step.observation["view_hierarchy"]
                    #, instruction
                    #)

            if nb_steps>=max_nb_steps:
                succeeds = False
                break

        model.end( command
                 , step.observation["view_hierarchy"]
                 , instruction
                 , step.reward
                 , reward
                 )

        if succeeds:
            success_list.add(i)

        nb_stepss.append(nb_steps)
        rewards.append(reward)
        succeedss.append(int(succeeds))
        logger.info( "\x1b[42mEND!\x1b[0m TaskId: %d, TaskName: %s, #Steps: %d(%d), Reward: %.1f, Succeds: %s"
                   , i, env.task_id, nb_steps, nb_nothing_steps, reward, str(succeeds)
                   )
    logger.info( "──────────%.2f──────────%.3f──────────%.3f──────────"
               , np.mean(np.asarray(nb_stepss))
               , np.mean(np.asarray(rewards))
               , np.mean(np.asarray(succeedss))
               )
    return success_list
    #  }}} function traverse_environment # 

def main():
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument("--train-path", type=str)
    parser.add_argument("--task-path", type=str)
    parser.add_argument("--avd-name", type=str)
    parser.add_argument("--tokenizer-path", type=str)

    parser.add_argument("--load-replay", action="append", type=str)
    parser.add_argument("--save-replay", action="append", type=str)
    parser.add_argument("--item-capacity", type=int)
    parser.add_argument("--action-capacity", type=int)
    parser.add_argument("--matcher", default="lcs", type=str, choices=["lcs", "lcs+inspat", "inspat"])
    parser.add_argument("--gamma", default=1., type=float)
    parser.add_argument("--step-penalty", default=0., type=float)
    parser.add_argument("--update-mode", default="mean", type=str, choices=["mean", "const"])
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--n-step-flatten", type=int)
    parser.add_argument("--double-q-learning", action="store_true")
    parser.add_argument("--iteration-mode", default="turn", type=str, choices=["turn", "random"])

    parser.add_argument("--prompt-template", type=str)
    parser.add_argument("--max-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--stop", type=str)
    parser.add_argument("--request-timeout", default=3., type=float)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--norandom", action="store_true")

    parser.add_argument("--starts-from", default=0, type=int)
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--except", nargs="+", type=int)

    parser.add_argument("--replay-file", nargs="+", type=str)
    parser.add_argument("--dump-path", nargs="+", type=str)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options # 

    #  Config Logger {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    file_handler = logging.FileHandler( os.path.join( args.log_dir
                                                    , "normal-{:}.log".format(datetime_str)
                                                    )
                                      )
    debug_handler = logging.FileHandler( os.path.join( args.log_dir
                                                     , "debug-{:}.log".format(datetime_str)
                                                     )
                                       )
    stdout_handler = logging.StreamHandler(sys.stdout)
    sdebug_handler = logging.FileHandler( os.path.join( args.log_dir
                                                      , "sdebug-{:}.log".format(datetime_str)
                                                      )
                                        )
    openai_error_handler = logging.FileHandler( os.path.join( args.log_dir
                                                            , "openai-{:}.log".format(datetime_str)
                                                            )
                                              )
    hdebug_handler = logging.FileHandler( os.path.join( args.log_dir
                                                      , "hdebug-{:}.log".format(datetime_str)
                                                      )
                                        )

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    sdebug_handler.setLevel(logging.DEBUG)
    openai_error_handler.setLevel(logging.DEBUG)
    hdebug_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    sdebug_handler.setFormatter(formatter)
    openai_error_handler.setFormatter(formatter)
    hdebug_handler.setFormatter(formatter)

    #stdout_handler.addFilter(logging.Filter("main"))
    stdout_handler.addFilter(logging.Filter("wikihow"))
    sdebug_handler.addFilter(logging.Filter("wikihow"))
    openai_error_handler.addFilter(logging.Filter("openaiE"))
    hdebug_handler.addFilter(logging.Filter("history"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)
    logger.addHandler(openai_error_handler)
    logger.addHandler(hdebug_handler)

    logger = logging.getLogger("wikihow")
    #  }}} Config Logger # 

    #  Build Agent and Environment {{{ # 
    matcher_functions: Dict[str, history.MatcherConstructor[wikihow_agent.Key]]\
            = { "lcs": history.LCSNodeMatcher
              , "inspat": history.InsPatMatcher
              , "lcs+inspat": history.LambdaMatcherConstructor( [ history.LCSNodeMatcher
                                                                , history.InsPatMatcher
                                                                ]
                                                              , [0.5, 0.5]
                                                              ).get_lambda_matcher
              }
    if args.double_q_learning:
        history_replay: history.AbstractHistoryReplay[wikihow_agent.Key, wikihow_agent.Action]\
                = history.DoubleHistoryReplay( args.item_capacity
                                             , args.action_capacity
                                             , matcher=matcher_functions[args.matcher]
                                             , gamma=args.gamma
                                             , step_penalty=args.step_penalty
                                             , update_mode=args.update_mode
                                             , learning_rate=args.learning_rate
                                             , n_step_flatten=args.n_step_flatten
                                             , iteration_mode=args.iteration_mode
                                             )
        history_replay.load_yaml(args.load_replay)
    else:
        history_replay: history.AbstractHistoryReplay[wikihow_agent.Key, wikihow_agent.Action]\
                = history.HistoryReplay( args.item_capacity
                                       , args.action_capacity
                                       , matcher=matcher_functions[args.matcher]
                                       , gamma=args.gamma
                                       , step_penalty=args.step_penalty
                                       , update_mode=args.update_mode
                                       , learning_rate=args.learning_rate
                                       , n_step_flatten=args.n_step_flatten
                                       )
        history_replay.load_yaml(args.load_replay[0])

    with open(os.path.join(args.prompt_template, "prompt_pth.txt")) as f:
        prompt_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "input_template.txt")) as f:
        input_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "advice_template.txt")) as f:
        advice_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "canonical_examplar_E0.1.txt")) as f:
        canonical1: str = f.read()
    with open(os.path.join(args.prompt_template, "canonical_examplar_E0.2.txt")) as f:
        canonical2: str = f.read()
    template_group = agent_protos.TemplateGroup( whole_template=prompt_template
                                               , input_template=input_template
                                               , advice_template=advice_template
                                               , canonical1=canonical1
                                               , canonical2=canonical2
                                               )
    with open(args.config) as f:
        openaiconfig: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
    api_key: str = openaiconfig["api_key"]
    model = wikihow_agent.AutoAgent( history_replay=history_replay
                           , prompt_templates=template_group
                           , api_key=api_key
                           , max_tokens=args.max_tokens
                           , temperature=args.temperature
                           , stop=args.stop
                           , request_timeout=args.request_timeout
                           , static=args.static
                           , manual=args.manual
                           , train=args.train
                           , norandom=args.norandom
                           )
    #model = wikihow_agent.ManualAgent()
    #model = wikihow_agent.ReplayAgent(args.replay_file)

    env = android_env.load( args.task_path
                          , args.avd_name
                          , os.path.expanduser("~/.android/avd")
                          , os.path.expanduser("~/Android/Sdk")
                          , os.path.expanduser("~/Android/Sdk/emulator/emulator")
                          , os.path.expanduser("~/Android/Sdk/platform-tools/adb")
                          , run_headless=True
                          , mitm_config={"method": "syscert"}
                          , unify_vocabulary=os.path.join( args.tokenizer_path
                                                         , "vocab.txt"
                                                         )
                          , with_view_hierarchy=True
                          )
    env = VhIoWrapper( env
                     , AutoTokenizer.from_pretrained(args.tokenizer_path)
                     , nb_click_frames=3
                     , nb_scroll_frmaes=10
                     )
    if args.train:
        train_env = android_env.load( args.train_path
                                    , args.avd_name
                                    , os.path.expanduser("~/.android/avd")
                                    , os.path.expanduser("~/Android/Sdk")
                                    , os.path.expanduser("~/Android/Sdk/emulator/emulator")
                                    , os.path.expanduser("~/Android/Sdk/platform-tools/adb")
                                    , run_headless=True
                                    , mitm_config={"method": "syscert"}
                                    , unify_vocabulary=os.path.join( args.tokenizer_path
                                                                   , "vocab.txt"
                                                                   )
                                    , with_view_hierarchy=True
                                    )
        train_env = VhIoWrapper( train_env
                               , AutoTokenizer.from_pretrained(args.tokenizer_path)
                               , nb_click_frames=3
                               , nb_scroll_frmaes=10
                               )

    logger.info("The environment is ready.")
    #  }}} Build Agent and Environment # 

    #  Work Flow {{{ # 
    except_list: Set[int] = set() if getattr(args, "except") is None else set(getattr(args, "except"))

    if not args.train:
        starts_from = 0
        nb_epochs = 1
    else:
        starts_from: int = args.starts_from
        nb_epochs: int = args.epochs
    max_nb_steps = 15
    for epch in range(starts_from, nb_epochs):
        if args.train:
            model.train(True)
            success_list: Set[int] = traverse_environment( train_env, model
                                                         , logger, except_list
                                                         , max_nb_steps=max_nb_steps
                                                         )
            if epch==0:
                except_list |= success_list
        model.train(False)
        # [ 3,  5,  7,  8,  9, 10, 16, 17, 25, 30, 38, 39, 43, 44, 46, 47, 48, 55, 57, 58]
        traverse_environment( env, model
                            , logger
                             #, except_list={ 64, 23, 52,  1, 56
                             #, 65, 27, 40, 20, 63
                             #, 60, 24, 54, 31, 42
                             #, 51, 59, 33,  4,  6
                             #, 67, 29, 37, 19, 61
                             #, 34, 28, 50,  0, 18
                             #, 32, 13, 21,  2, 53
                             #, 26, 15, 66, 14, 22
                             #, 36, 69, 35, 41, 68
                             #, 11, 62, 45, 49, 12
                             #}
                            , max_nb_steps=max_nb_steps
                            )

        if args.train:
            if args.double_q_learning:
                history_replay.save_yaml( [ args.save_replay[0] % epch
                                          , args.save_replay[1] % epch
                                          ]
                                        )
            else:
                history_replay.save_yaml(args.save_replay[0] % epch)

        epoch_str = "Epoch {:}".format(epch)
        logger.info("\x1b[31m━━━━━━━━━━━━━━━━━━━%s━━━━━━━━━━━━━━━━━━━━\x1b[0m", epoch_str)
        logger.info( "Size: %d, Avg AD Size: %d"
                   , len(history_replay)
                   , sum( map( lambda rcd: len(rcd["action_dict"])
                             , itertools.chain( history_replay._history_replays[0]._record.values()
                                              , history_replay._history_replays[1]._record.values()
                                              ) if args.double_q_learning\
                          else history_replay._record.values()
                             )
                        )\
                   / float(len(history_replay))
                   )
        logger.info("\x1b[31m━━━━━━━━━━━━━━━━━━━%s━━━━━━━━━━━━━━━━━━━━\x1b[0m", "━" * len(epoch_str))
    #  }}} Work Flow # 

if __name__ == "__main__":
    main()
