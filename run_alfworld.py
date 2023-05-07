#!/usr/bin/python3

import numpy as np
import alfworld.agents.environment as environment
#import alfworld.agents.modules.generic as generic
import functools
from sentence_transformers import SentenceTransformer
import string

import alfworld_agent
import history
import agent_protos
import alfworld_prompts

import yaml
import argparse
import datetime
import os
import sys

import logging
from typing import Set, List, Tuple, Dict
from typing import Any

#  Static Prompts Config {{{ # 
training_set_type: List[str] = [ "puttwo", "examine", "heat", "puttwo", "puttwo"
                               , "clean", "cool", "clean", "put", "cool"
                               , "puttwo", "examine", "examine", "cool", "cool"
                               , "heat", "put", "examine", "put", "puttwo"
                               , "put", "clean", "heat", "puttwo", "puttwo"
                               , "puttwo", "put", "puttwo", "put", "put"
                               , "puttwo", "put", "heat", "cool", "put"
                               , "clean", "puttwo", "puttwo", "put", "heat"
                               , "puttwo", "heat", "examine", "put", "puttwo"
                               , "clean", "put", "put", "puttwo", "heat"
                               ]
test_set_type: List[str] = [ "put", "clean", "cool", "puttwo", "heat"
                           , "put", "clean", "puttwo", "heat", "clean"
                           , "clean", "put", "put", "cool", "cool"
                           , "heat", "heat", "clean", "put", "clean"
                           , "examine", "examine", "put", "examine", "cool"
                           , "examine", "put", "clean", "clean", "put"
                           , "puttwo", "examine", "heat", "puttwo", "puttwo"
                           , "heat", "heat", "put", "heat", "heat"
                           , "examine", "heat", "puttwo", "put", "puttwo"
                           , "examine", "clean", "put", "put", "clean"
                           , "put", "cool", "heat", "clean", "cool"
                           , "heat", "put", "cool", "puttwo", "examine"
                           , "clean", "heat", "puttwo", "cool", "heat"
                           , "heat", "examine", "clean", "cool", "put"
                           , "heat", "cool", "heat", "cool", "examine"
                           , "examine", "clean", "put", "put", "put"
                           , "clean", "puttwo", "heat", "clean", "clean"
                           , "puttwo", "examine", "cool", "put", "cool"
                           , "clean", "clean", "puttwo", "put", "cool"
                           , "cool", "cool", "put", "cool", "cool"
                           , "heat", "clean", "clean", "examine", "heat"
                           , "puttwo", "clean", "heat", "clean", "clean"
                           , "put", "examine", "puttwo", "examine", "clean"
                           , "clean", "cool", "put", "heat", "heat"
                           , "puttwo", "clean", "puttwo", "cool", "put"
                           , "cool", "clean", "clean", "clean", "examine"
                           , "examine", "examine", "puttwo", "clean"
                           ]
per_type_prompts: Dict[str, Tuple[str, str]]\
        = { "put": (alfworld_prompts.put_1_0, alfworld_prompts.put_1_1)
          , "clean": (alfworld_prompts.clean_1_0, alfworld_prompts.clean_1_1)
          , "cool": (alfworld_prompts.cool_1_0, alfworld_prompts.cool_1_1)
          , "puttwo": (alfworld_prompts.puttwo_1_0, alfworld_prompts.puttwo_1_1)
          , "heat": (alfworld_prompts.heat_1_0, alfworld_prompts.heat_1_1)
          , "examine": (alfworld_prompts.examine_1_0, alfworld_prompts.examine_1_1)
          }
#  }}} Static Prompts Config # 

def traverse_environment( env: environment.AlfredTWEnv
                        , max_task_id: int
                        , task_types: List[str]
                        , model: alfworld_agent.Agent
                        , logger: logging.Logger
                        , except_list: Set[int] = set()
                        , max_nb_steps: int = 50
                        ) -> Set[int]:
    #  function traverse_environment {{{ # 
    success_list: Set[int] = set()

    nb_stepss: List[int] = []
    rewards: List[float] = []
    succeedss: List[int] = []
    for i, t in zip(range(max_task_id), task_types):
        observation: Tuple[str]
        info: Dict[str, Any]
        observation, info = env.reset()
        if i in except_list:
            continue

        init_state: str
        goal: str
        _, init_state, goal = observation[0].split("\n\n")
        assert goal.startswith("Your task is to: ")
        goal = goal[17:]
        trajectory = ""

        logger.debug("%s. %s.", init_state, goal)

        task_name: str = "/".join(info["extra.gamefile"][0].split("/")[-3:-1])

        available_actions: List[str] = info["admissible_commands"][0]

        model.reset()
        if hasattr(model, "set_static_prompts"):
            model.set_static_prompts(per_type_prompts[t])

        nb_steps = 0
        nb_nothing_steps = 0

        reward = (0.,)
        total_reward = 0.
        succeeds = False
        while nb_steps<max_nb_steps:
            action: str = model( init_state
                               , goal
                               , trajectory
                               , tuple(available_actions)
                               , reward[0]
                               )

            if action!="NOTHINGG":
                observation, reward, done, info = env.step([action])

                #trajectory += "> {:}\n".format(action)
                observation: str = observation[0]
                if observation.startswith("You arrive at loc "):
                    observation = observation[observation.find(". ")+2:]
                trajectory += "{:}\n".format(observation)
                logger.debug(observation)

                available_actions = info["admissible_commands"][0]
                total_reward += reward[0]

                nb_steps += 1
                if done[0]:
                    succeeds = info["won"][0]
                    break
            else:
                nb_nothing_steps += 1

        model.end( init_state
                 , goal
                 , trajectory
                 , tuple(available_actions)
                 , reward[0]
                 )

        if succeeds:
            success_list.add(i)

        nb_stepss.append(nb_steps)
        rewards.append(total_reward)
        succeedss.append(int(succeeds))
        logger.info("\x1b[43mEND!\x1b[0m %s, %s", init_state, goal)
        logger.info( "\x1b[42mEND!\x1b[0m TaskId: %d, TaskName: %s, #Steps: %d(%d), Reward: %.2f, Succeds: %s"
                   , i, task_name, nb_steps, nb_nothing_steps, total_reward, str(succeeds)
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

    # Env Options
    parser.add_argument("--alfworld-config", default="alfworld_config.yaml", type=str)

    # Matcher Options
    parser.add_argument( "--sentence-transformer"
                       , default="all-MiniLM-L12-v2", type=str
                       , choices=[ "all-MiniLM-L12-v2"
                                 , "all-mpnet-base-v2"
                                 ]
                       )

    # Replay Options
    parser.add_argument("--load-replay", type=str)
    parser.add_argument("--save-replay", type=str)
    parser.add_argument("--item-capacity", type=int)
    parser.add_argument("--action-capacity", type=int)
    parser.add_argument( "--matcher"
                       , default="4insrel3istrel3trjrel"
                       , type=str
                       , choices=[ "4insrel3istrel3trjrel"
                                 , "4inspata3recpiou3obvpat"
                                 ]
                       )
    parser.add_argument("--gamma", default=1., type=float)
    parser.add_argument("--step-penalty", default=0., type=float)
    parser.add_argument("--update-mode", default="mean", type=str, choices=["mean", "const"])
    parser.add_argument("--learning-rate", default=0.1, type=float)
    parser.add_argument("--n-step-flatten", type=int)

    # Agent Options
    parser.add_argument("--prompt-template", type=str)
    parser.add_argument("--max-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--stop", type=str)
    parser.add_argument("--request-timeout", default=3., type=float)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--manual", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--speech", action="store_true")

    # Trainging Options
    parser.add_argument("--starts-from", default=0, type=int)
    parser.add_argument("--epochs", default=3, type=int)

    parser.add_argument("--except", nargs="+", type=int)
    parser.add_argument("--trainseta", default=0, type=int)
    parser.add_argument("--trainsetb", default=20, type=int)
    parser.add_argument("--testseta", default=0, type=int)
    parser.add_argument("--testsetb", default=10, type=int)

    args = parser.parse_args()
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

    stdout_handler.addFilter(logging.Filter("alfworld"))
    sdebug_handler.addFilter(logging.Filter("alfworld"))
    odebug_handler.addFilter(logging.Filter("openaiE"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)
    logger.addHandler(odebug_handler)

    logger = logging.getLogger("alfworld")
    #  }}} Config Logger # 

    #  Build Agent and Environment {{{ # 
    sentence_transformer = SentenceTransformer(args.sentence_transformer)
    matcher_functions: Dict[str, history.LambdaMatcherConstructor[alfworld_agent.Key]]\
            = { "4insrel3istrel3trjrel": history.LambdaMatcherConstructor( [ functools.partial( history.DenseInsMatcher
                                                                                              , transformer=sentence_transformer
                                                                                              )
                                                                           , functools.partial( history.DenseInsMatcher
                                                                                              , transformer=sentence_transformer
                                                                                              , index=0
                                                                                              )
                                                                           , functools.partial( history.DenseTrajectoryMatcher
                                                                                              , transformer=sentence_transformer
                                                                                              )
                                                                           ]
                                                                         , [0.4, 0.3, 0.3]
                                                                         ).get_lambda_matcher
              , "4inspata3recpiou3obvpat": history.LambdaMatcherConstructor( [ history.AlfInsPatMatcher
                                                                             , history.ReceptacleIoUMatcher
                                                                             , history.AlfObvPatMatcher
                                                                             ]
                                                                           , [0.4, 0.3, 0.3]
                                                                           ).get_lambda_matcher
              }
    history_replay: history.HistoryReplay[alfworld_agent.Key, alfworld_agent.Action]\
            = history.HistoryReplay( args.item_capacity
                                   , args.action_capacity
                                   , matcher=matcher_functions[args.matcher]
                                   , gamma=args.gamma
                                   , step_penalty=args.step_penalty
                                   , update_mode=args.update_mode
                                   , learning_rate=args.learning_rate
                                   , n_step_flatten=args.n_step_flatten
                                   )
    history_replay.load_yaml(args.load_replay)

    with open(os.path.join(args.prompt_template, "prompt_ptha.txt")) as f:
        prompt_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "input_template_a.txt")) as f:
        input_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "advice_template.txt")) as f:
        advice_template = string.Template(f.read())
    with open(os.path.join(args.prompt_template, "canonical_examplar1_a_clean.txt")) as f:
        canonical1: str = f.read()
    with open(os.path.join(args.prompt_template, "canonical_examplar2_a_clean.txt")) as f:
        canonical2: str = f.read()
    template_group = agent_protos.TemplateGroup( whole_template=prompt_template
                                               , input_template=input_template
                                               , advice_template=advice_template
                                               , canonical1=canonical1
                                               , canonical2=canonical2
                                               )
    with open(args.config) as f:
        openaiconfig: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
    if args.speech:
        api_key: str = openaiconfig["spc_token"]
    else:
        api_key: str = openaiconfig["api_key"]
    model = alfworld_agent.AutoAgent( history_replay=history_replay
                                    , prompt_templates=template_group
                                    , api_key=api_key
                                    , max_tokens=args.max_tokens
                                    , temperature=args.temperature
                                    , stop=args.stop
                                    , request_timeout=args.request_timeout
                                    , static=args.static
                                    , manual=args.manual
                                    , train=args.train
                                    , with_speech=args.speech
                                    )
    #model = alfworld_agent.ManualAgent()

    with open(args.alfworld_config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    split = "eval_out_of_distribution"
    env: environment.AlfredTWEnv = getattr(environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    if args.train:
        train_env: environment.AlfredTWEnv\
                = getattr(environment, config["env"]["type"])(config, train_eval="train")
        train_env = train_env.init_env(batch_size=1)

    logger.info("The environment is ready.")
    #  }}} Build Agent and Environment # 

    #  Workflow {{{ # 
    except_list: Set[int] = set() if getattr(args, "except") is None\
                                else set(getattr(args, "except"))
    except_list |= set(range(args.trainseta))
    if not args.train:
        starts_from = 0
        nb_epochs = 1
    else:
        starts_from: int = args.starts_from
        nb_epochs: int = args.epochs
    max_nb_steps = 50
    for epch in range(starts_from, nb_epochs):
        if args.train:
            model.train(True)
            success_list: Set[int] = traverse_environment( train_env, args.trainsetb
                                                         , training_set_type[:args.trainsetb]
                                                         , model
                                                         , logger, except_list
                                                         , max_nb_steps=max_nb_steps
                                                         )
            if epch % 3 == 0:
                except_list |= success_list

        model.train(False)
        traverse_environment( env, args.testsetb
                            , test_set_type[:args.testsetb]
                            , model
                            , logger, set(range(args.testseta))
                            , max_nb_steps=max_nb_steps
                            )

        if args.train:
            history_replay.save_yaml(args.save_replay % epch)

        epoch_str = "Epoch {:}".format(epch)
        logger.info("\x1b[31m━━━━━━━━━━━━━━━━━━━%s━━━━━━━━━━━━━━━━━━━━\x1b[0m", epoch_str)
        #logger.info( "Size: %d, Avg AD Size: %d"
                   #, len(history_replay)
                   #, sum( map( lambda rcd: len(rcd["action_dict"])
                             #, history_replay._record.values()
                             #)
                        #)\
                                #/ float(len(history_replay))
                   #)
        logger.info("\x1b[31m━━━━━━━━━━━━━━━━━━━%s━━━━━━━━━━━━━━━━━━━━\x1b[0m", "━" * len(epoch_str))
    #  }}} Workflow # 

if __name__ == "__main__":
    main()
