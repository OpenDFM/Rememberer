#!/usr/bin/python3

import sys
sys.path.append("../WebShop")

import gym
import web_agent_site.envs
from web_agent_site.utils import DEFAULT_FILE_PATH
import webshop_agent

import logging
import argparse
import datetime
import os

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
# def reset() -> Tuple[ str # observation
#                     , None
#                     ]
#  }}} Interfaces of WebAgentTextEnv # 

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

    # Agent Options
    pass

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
    model = webshop_agent.ManualAgent(args.observation_mode)

    env = gym.make( "WebAgentTextEnv-v0"
                  , observation_mode="text"
                  , file_path=(args.file_path if args.file_path is not None and args.file_path != ""
                                            else DEFAULT_FILE_PATH)
                  , num_products=None
                  , num_prev_actions=args.prev_actions
                  , num_prev_obs=args.prev_obs
                  )
    #  }}} Build Agent and Environment # 

    #  Workflow {{{ # 
    max_nb_tasks = 10
    max_nb_steps = 15
    for i in range(max_nb_tasks):
        model.reset()
        observation: str = env.reset()[0]
        task: str = env.get_instruction_text()

        nb_steps = 0
        nb_nothing_steps = 0

        reward = 0.
        total_reward = 0.
        succeeds = False
        while nb_steps<max_nb_steps:
            action: webshop_agent.Action = model( task
                                                , observation
                                                , reward
                                                , total_reward
                                                )
            if action!="NOTHINGG":
                observation, reward, done, _ = env.step(action)
                total_reward += reward
                nb_steps += 1
                if done:
                    succeeds = True
                    break
            else:
                nb_nothing_steps += 1

        logger.info("\x1b[43mEND!\x1b[0m %s", task)
        logger.info( "\x1b[42mEND!\x1b[0m TaskIdx: %d, #Steps: %d(%d), Reward: %.1f, Succeds: %s"
                   , i, nb_steps, nb_nothing_steps, total_reward, str(succeeds)
                   )
    #  }}} Workflow # 

#if __name__ == "__main__":
    #main()
