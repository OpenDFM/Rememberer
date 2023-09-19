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

import sys
sys.path.append("../WebShop")

import gym
import importlib
importlib.import_module("web_agent_site.envs")
#import web_agent_site.envs
from web_agent_site.utils import DEFAULT_FILE_PATH

import functools
from sentence_transformers import SentenceTransformer
import history
import yaml

import string
import webshop_agent
import agent_protos
import itertools

import logging
import argparse
import datetime
import os

from typing import List, Dict, Set
import numpy as np

#  Interfaces of WebAgentTextEnv {{{ # 
# def init( observation_mode: str = "html" # "html": raw html
#                                          # "text": " [SEP] " joined element text contents
#                                          # "text_rich": "\n" joined discription with bounded text contents
#                                          #              bounding labels are
#                                          #              [button][button_] and [clicked
#                                          #              button][clicked button_];
#                                          #              non-product-link as [clicked
#                                          #              button] will be prefixed with a
#                                          #              discription as "You have clicked {t}.\n"
#                                          # "url": url
#     , file_path: str = utils.DEFAULT_FILE_PATH # path to a json as the data file
#     , num_products: Optional[int]
#     , human_goals: Optional[bool] # if the human-annotated goals should be used
#     , num_prev_actions: int = 0 # the number of history actions to append
#                                 # after the observation; actions are appended
#                                 # in a reverse order
#     , num_prev_obs: int = 0 # the number of history observations to append
#                             # after the current observation; observations are
#                             # appended in a reverse order; observations are
#                             # suffixed interleavingly with the actions like:
#                             # 
#                             # <current_obs> [SEP] act_{n-1} [SEP] obs_{n-1} [SEP] ... [SEP] obs_0
#     )
# 
# def step( action: str # search[keywords]
#                       # click[element], element should in
#                       # self.text_to_clickable and shouldn't be search
#         ) -> Tuple[ str # observation
#                   , float # reward
#                   , bool # done or not
#                   , None
#                   ]
# 
# def get_available_actions()\
#         -> Dict[str, bool | List[str]]
#         # {
#         #   "has_search_bar": bool
#         #   "clickables": List[str]
#         # }
# def get_instruction_text() -> str
# def observation -> str
# def state -> Dict[str, str]
#           # {
#           #     "url": str
#           #     "html": str
#           #     "instruction_text": str
#           # }
# text_to_clickable: Dict[str, Any] # {element_text: bs4 element}
# instruction_text: str
# 
# def reset( session: Optional[Union[int, str]] # int for the goal index
#          ) -> Tuple[ str # observation
#                    , None
#                    ]
#  }}} Interfaces of WebAgentTextEnv # 

def traverse_environment( env: gym.Env
                        , task_set: List[int]
                        , model: webshop_agent.Agent
                        , logger: logging.Logger
                        , except_list: Set[int] = set()
                        , max_nb_steps: int = 15
                        , max_nb_consective_nothings: int = 15
                        ) -> Set[int]:
    #  function traverse_environment {{{ # 
    """
    Args:
        env (gym.Env): the environment
        task_set (List[int]): the traversed task set
        model (agent.Agent): the agent
        logger (logging.Logger): the logger
        except_list (Set[int]): tasks in this set won't be tested

        max_nb_steps (int): if the number of steps exceeds `max_nb_steps`, the
          episode will be killed and considered failed.
        max_nb_consective_nothings (int): if the number of consecutive NOTHINGG
          steps exceeds `max_nb_consective_nothings`, the episode will be
          killed and considered failed.

    Returns:
        Set[int]: set of the succeeded tasks
    """

    success_list: Set[int] = set()

    nb_stepss: List[int] = []
    rewards: List[float] = []
    succeedss: List[int] = []
    for idx, i in enumerate(task_set):
        if i in except_list:
            continue

        model.reset()
        observation: str = env.reset(session=i)[0]
        task: str = env.get_instruction_text()
        available_actions: List[str] = env.get_available_actions()["clickables"]

        nb_steps = 0
        nb_nothing_steps = 0
        nb_consecutive_nothings = 0

        reward = 0.
        total_reward = 0.
        succeeds = False
        while nb_steps<max_nb_steps and nb_consecutive_nothings<max_nb_consective_nothings:
            action: str = model( task
                               , observation
                               , reward
                               , total_reward
                               , available_actions
                               )
            if action!="NOTHINGG":
                observation, reward, done, _ = env.step(action)
                total_reward += reward
                available_actions = env.get_available_actions()["clickables"]

                nb_steps += 1
                nb_consecutive_nothings = 0
                if done:
                    succeeds = reward==1.
                    break
            else:
                nb_nothing_steps += 1
                nb_consecutive_nothings += 1

        model.end( task
                 , observation
                 , reward
                 , total_reward
                 , available_actions
                 )

        if succeeds:
            success_list.add(i)

        nb_stepss.append(nb_steps)
        rewards.append(total_reward)
        succeedss.append(int(succeeds))
        logger.info("\x1b[43mEND!\x1b[0m %s", task)
        logger.info( "\x1b[42mEND!\x1b[0m TaskIdx: %d, TaskId: %d, #Steps: %d(%d), Reward: %.2f, Succeds: %s"
                   , idx, i, nb_steps, nb_nothing_steps, total_reward, str(succeeds)
                   )
    logger.info( "──────────{:.2f}──────────{:.3f}──────────{:.3f}──────────"\
                 .format( np.mean(np.asarray(nb_stepss))
                        , np.mean(np.asarray(rewards))
                        , np.mean(np.asarray(succeedss))
                        )
               )
    return success_list
    #  }}} function traverse_environment # 

def main():
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument( "--observation-mode"
                       , default="text", type=str
                       , choices=[ "html"
                                 , "text"
                                 , "text_rich"
                                 , "url"
                                 ]
                       )
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--prev-actions", default=0, type=int)
    parser.add_argument("--prev-observations", default=0, type=int)

    # Matcher Options
    parser.add_argument( "--sentence-transformer"
                       , default="all-MiniLM-L12-v2", type=str
                       , choices=[ "all-MiniLM-L12-v2"
                                 , "all-mpnet-base-v2"
                                 ]
                       )

    # Replay Options
    parser.add_argument("--load-replay", action="append", type=str)
    parser.add_argument("--save-replay", action="append", type=str)
    parser.add_argument("--item-capacity", type=int)
    parser.add_argument("--action-capacity", type=int)
    parser.add_argument( "--matcher"
                       , default="pgpat+iprel", type=str
                       , choices=[ "pgpat+iprel"
                                 , "pgpat+iprel+insrel"
                                 , "pgpat+insrel"
                                 , "9pgpat1insrel"
                                 , "pgpat"
                                 , "insrel"
                                 ]
                       )
    parser.add_argument("--gamma", default=1., type=float)
    parser.add_argument("--step-penalty", default=0., type=float)
    parser.add_argument("--update-mode", default="mean", type=str, choices=["mean", "const"])
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--n-step-flatten", type=int)
    parser.add_argument("--double-q-learning", action="store_true")
    parser.add_argument("--iteration-mode", default="turn", type=str, choices=["turn", "random"])

    # Agent Options
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
    parser.add_argument("--pub-to-local-mapping", type=str)
    parser.add_argument("--trainseta", default=0, type=int)
    parser.add_argument("--trainsetb", default=20, type=int)
    parser.add_argument("--testseta", default=0, type=int)
    parser.add_argument("--testsetb", default=10, type=int)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options # 

    #  Config Logger {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

    file_handler = logging.FileHandler(os.path.join(args.log_dir, "normal-{:}.log".format(datetime_str)))
    debug_handler = logging.FileHandler(os.path.join(args.log_dir, "debug-{:}.log".format(datetime_str)))
    stdout_handler = logging.StreamHandler(sys.stdout)
    sdebug_handler = logging.FileHandler(os.path.join(args.log_dir, "sdebug-{:}.log".format(datetime_str)))
    odebug_handler = logging.FileHandler(os.path.join(args.log_dir, "openai-{:}.log".format(datetime_str)))

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    sdebug_handler.setLevel(logging.DEBUG)
    odebug_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    sdebug_handler.setFormatter(formatter)
    odebug_handler.setFormatter(formatter)

    stdout_handler.addFilter(logging.Filter("webshop"))
    sdebug_handler.addFilter(logging.Filter("webshop"))
    odebug_handler.addFilter(logging.Filter("openaiE"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)
    logger.addHandler(odebug_handler)

    logger = logging.getLogger("webshop")
    #  }}} Config Logger # 

    #  Build Agent and Environment {{{ # 
    sentence_transformer = SentenceTransformer(args.sentence_transformer)
    matcher_functions: Dict[str, history.LambdaMatcherConstructor[webshop_agent.Key]]\
            = { "pgpat+iprel": history.LambdaMatcherConstructor( [ history.PagePatMatcher
                                                                 , functools.partial( history.InsPageRelMatcher
                                                                                    , transformer=sentence_transformer
                                                                                    )
                                                                 ]
                                                               , [0.5, 0.5]
                                                               ).get_lambda_matcher
              , "pgpat+iprel+insrel": history.LambdaMatcherConstructor( [ history.PagePatMatcher
                                                                        , functools.partial( history.InsPageRelMatcher
                                                                                           , transformer=sentence_transformer
                                                                                           )
                                                                        , functools.partial( history.DenseInsMatcher
                                                                                           , transformer=sentence_transformer
                                                                                           )
                                                                        ]
                                                                      , [1/3., 1/3., 1/3.]
                                                                      ).get_lambda_matcher
              , "pgpat+insrel": history.LambdaMatcherConstructor( [ history.PagePatMatcher
                                                                  , functools.partial( history.DenseInsMatcher
                                                                                     , transformer=sentence_transformer
                                                                                     )
                                                                  ]
                                                                , [0.5, 0.5]
                                                                ).get_lambda_matcher
              , "9pgpat1insrel": history.LambdaMatcherConstructor( [ history.PagePatMatcher
                                                                   , functools.partial( history.DenseInsMatcher
                                                                                      , transformer=sentence_transformer
                                                                                      )
                                                                   ]
                                                                 , [0.9, 0.1]
                                                                 ).get_lambda_matcher
              , "pgpat": history.PagePatMatcher
              , "insrel": functools.partial( history.DenseInsMatcher
                                           , transformer=sentence_transformer
                                           )
              }
    if args.double_q_learning:
        history_replay: history.AbstractHistoryReplay[webshop_agent.Key, webshop_agent.Action]\
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
        history_replay: history.AbstractHistoryReplay[webshop_agent.Key, webshop_agent.Action]\
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

    with open(os.path.join(args.prompt_template, "prompt_pthw.txt")) as f:
        prompt_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "input_template_w.txt")) as f:
        input_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "advice_template.txt")) as f:
        advice_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "canonical_examplar_wE0.1.txt")) as f:
        canonical1: str = f.read()
    with open(os.path.join(args.prompt_template, "canonical_examplar_wE0.2.txt")) as f:
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
    model = webshop_agent.AutoAgent( history_replay=history_replay
                                   , prompt_templates=template_group
                                   , api_key=api_key
                                   , max_tokens=args.max_tokens
                                   , temperature=args.temperature
                                   , stop=args.stop
                                   , request_timeout=args.request_timeout
                                   , static=args.static
                                   , manual=args.manual
                                   , train=args.train
                                   , env_mode=args.observation_mode
                                   , norandom=args.norandom
                                   )
    #model = webshop_agent.ManualAgent(args.observation_mode)

    env = gym.make( "WebAgentTextEnv-v0"
                  , observation_mode=args.observation_mode
                  , file_path=(args.file_path if args.file_path is not None and args.file_path != ""
                                            else DEFAULT_FILE_PATH)
                  , num_products=None
                  , human_goals=True
                  , num_prev_actions=args.prev_actions
                  , num_prev_obs=args.prev_observations
                  )
    #if args.train:
        #train_env = gym.make( "WebAgentTextEnv-v0"
                            #, observation_mode=args.observation_mode
                            #, file_path=(args.file_path if args.file_path is not None and args.file_path != ""
                                                      #else DEFAULT_FILE_PATH)
                            #, num_products=None
                            #, num_prev_actions=args.prev_actions
                            #, num_prev_obs=args.prev_observations
                            #)

    logger.info("The environment is ready.")
    #  }}} Build Agent and Environment # 

    #  Workflow {{{ # 
    if args.pub_to_local_mapping is None:
        local_mapping: List[int] = list(range(600))
    else:
        with open(args.pub_to_local_mapping) as f:
            local_mapping: List[int] = list( map( int
                                                , f.read().splitlines()
                                                )
                                           )
    training_set: List[int] = local_mapping[500+args.trainseta:500+args.trainsetb]
    test_set: List[int] = local_mapping[args.testseta:args.testsetb]

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
            success_list: Set[int] = traverse_environment( env, training_set
                                                         , model
                                                         , logger, except_list
                                                         , max_nb_steps=max_nb_steps
                                                         )
            if epch==0:
                except_list |= success_list
        model.train(False)
        # [ 2,  8,  9, 15, 16, 18, 21, 22, 32, 38, 40, 58, 63, 66, 70, 76, 85, 90, 91, 93]
        traverse_environment( env, test_set
                            , model, logger
                             #, except_list=[ 81, 34, 48, 20, 29
                             #, 11, 73, 83, 64, 62
                             #, 78, 36, 80, 41, 33
                             #, 99,  3,  0, 55, 96
                             #, 50, 37, 75, 23, 94
                             #, 19,  6, 67, 97, 13
                             #, 87, 45, 52, 95, 88
                             #, 51, 47, 12, 89, 49
                             #,  4, 84, 30, 71, 35
                             #, 42, 65, 61, 10, 26
                             #, 92, 82,  5, 25, 74
                             #, 46, 86, 77, 17, 31
                             #, 24, 53, 39, 72, 60
                             #,  1, 59, 56, 54, 57
                             #, 98, 28, 43, 27, 44
                             #, 68, 79, 69,  7, 14
                             #]
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
    #  }}} Workflow # 

if __name__ == "__main__":
    main()
