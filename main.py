#!/usr/bin/python3

import logging
import argparse

import os
import os.path
import sys
import yaml
import datetime
import string

import agent
import android_env
from android_env.wrappers import VhIoWrapper
from transformers import AutoTokenizer
import dm_env

from typing import Dict
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

def main():
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument("--task-path", type=str)
    parser.add_argument("--avd-name", type=str)
    parser.add_argument("--tokenizer-path", type=str)

    parser.add_argument("--prompt-template", type=str)
    parser.add_argument("--max-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--request-timeout", default=3., type=float)

    parser.add_argument("--replay-file", type=str)
    parser.add_argument("--dump-path", type=str)

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

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    #stdout_handler.addFilter(logging.Filter("main"))
    stdout_handler.addFilter(logging.Filter("agent"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)

    logger = logging.getLogger("agent")
    #  }}} Config Logger # 

    #  Build Agent and Environment {{{ # 
    with open(args.prompt_template) as f:
        prompt_template = string.Template(f.read())
    with open(args.config) as f:
        openaiconfig: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
    model = agent.AutoAgent( prompt_template=prompt_template
                           , api_key=openaiconfig["api_key"]
                           , max_tokens=args.max_tokens
                           , temperature=args.temperature
                           , request_timeout=args.request_timeout
                           )
    #model = agent.ManualAgent()
    #model = agent.ReplayAgent(args.replay_file)

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

    logger.info("The environment is ready.")
    #  }}} Build Agent and Environment # 

    #os.makedirs(args.dump_path, exist_ok=True)

    max_nb_steps = 15
    for i in range(env.nb_tasks):
    #for i in [10]:
        model.reset()
        step: dm_env.TimeStep = env.switch_task(i)
        command: str = "\n".join(env.command())
        instruction: str = "\n".join(env.task_instructions())

        nb_steps = 0
        #dump( args.dump_path, nb_steps, command
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
                instruction = "\n".join(env.task_instructions())
            reward += step.reward

            nb_steps += 1
            #dump( args.dump_path, nb_steps, command
                #, step.observation["pixels"]
                #, step.observation["view_hierarchy"]
                #, instruction
                #)

            if nb_steps>=max_nb_steps:
                succeeds = False
                break

        logger.info( "\x1b[42mEND!\x1b[0m TaskId: %d, TaskName: %s, #Steps: %d, Reward: %.1f, Succeds: %s"
                   , i, env.task_id, nb_steps, reward, str(succeeds)
                   )

if __name__ == "__main__":
    main()