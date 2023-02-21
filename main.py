#!/usr/bin/python3

import logging
import argparse

import os.path
import sys

import agent
import android_env
from android_env.wrappers import VhIoWrapper
from transformers import AutoTokenizer
import dm_env

from typing import Dict
import numpy as np

def main():
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)

    parser.add_argument("--task-path", type=str)
    parser.add_argument("--avd-name", type=str)
    parser.add_argument("--tokenizer-path", type=str)

    parser.add_argument("--prompt-template", type=str)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options # 

    #  Config Logger {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(args.log_dir, "normal.log"))
    debug_handler = logging.FileHandler(os.path.join(args.log_dir, "debug.log"))
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    #  }}} Config Logger # 

    # TODO
    with open(args.prompt_template) as f:
        prompt_template: str = f.read()
    model = agent.Agent(prompt_template=prompt_template)

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
                          )
    env = VhIoWrapper( env
                     , AutoTokenizer.from_pretrained(args.tokenizer_path)
                     , nb_click_frames=3
                     , nb_scroll_frmaes=10
                     )

    max_nb_steps = 100
    for i in range(env.nb_tasks):
        model.reset()
        step: dm_env.TimeStep = env.switch_task(i)
        command: str = "\n".join(env.command())
        instruction: str = "\n".join(env.task_instructions())

        nb_steps = 0
        reward: float = step.reward
        succeeds: bool = True
        while not step.last():
            action: Dict[str, np.ndarray]\
                    = model( command
                           , step.observation["view_hierarchy"]
                           , instruction
                           )
            step = env.step(action)
            if len(env.task_instructions())>0:
                instruction = "\n".join(env.task_instructions())
            reward += step.reward

            nb_steps += 1
            if nb_steps>=max_nb_steps:
                succeeds = False
                break

        logger.info( "TaskId: %d, #Steps: %d, Reward: %f, Succeds: %s"
                   , i, nb_steps, reward, str(succeeds)
                   )

if __name__ == "__main__":
    main()
