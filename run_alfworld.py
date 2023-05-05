#!/usr/bin/python3

import numpy as np
import alfworld.agents.environment as environment
#import alfworld.agents.modules.generic as generic

import alfworld_agent

import yaml
import argparse
import datetime
import os
import sys

import logging
from typing import Set, List, Tuple, Dict
from typing import Any

def traverse_environment( env: environment.AlfredTWEnv
                        , max_task_id: int
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
    for i in range(max_task_id):
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
        available_actions: List[str] = info["admissible_commands"][0]

        model.reset()

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
        logger.info( "\x1b[42mEND!\x1b[0m TaskId: %d, #Steps: %d(%d), Reward: %.2f, Succeds: %s"
                   , i, nb_steps, nb_nothing_steps, total_reward, str(succeeds)
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

    # Replay Options

    # Agent Options
    parser.add_argument("--train", action="store_true")

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
    model = alfworld_agent.ManualAgent()

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
                                                         , model
                                                         , logger, except_list
                                                         , max_nb_steps=max_nb_steps
                                                         )
            if epch % 3 == 0:
                except_list |= success_list

        model.train(False)
        traverse_environment( env, args.testsetb
                            , model
                            , logger, set(range(args.testseta))
                            , max_nb_steps=max_nb_steps
                            )

        if args.train:
            pass # TODO

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
