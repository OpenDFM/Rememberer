#!/usr/bin/python3

from typing import Sequence, Optional, Union
from typing import NamedTuple, List, Dict
import numpy as np
import abc

import string
import datetime
import time
import yaml
import os
import sys

import logging
import argparse

import openai
import itertools

run_logger = logging.getLogger("agent")

class Step(NamedTuple):
    reward: int # 0 or 1

class BanditsEnv:
    #  class BanditsEnv {{{ # 
    def __init__( self
                , probs: Sequence[float]
                , seed: int = 999
                ):
        #  function __init__ {{{ # 
        """
        Args:
            probs (Sequence[float]): the probabilities of each arm
            seed (int): random seed
        """

        self._nb_arms: int = len(probs)
        self._probabilities: np.ndarray = np.array(probs, dtype=np.float32)
        self._seed: int = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        #  }}} function __init__ # 

    def reset(self, seed: Optional[int] = None) -> Step:
        #  function reset {{{ # 
        """
        Args:
            seed (Optional[int]): an optional new random seed

        Returns:
            Step: step information
        """

        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        return Step(reward=0)
        #  }}} function reset # 

    def step(self, action: int) -> Step:
        #  function step {{{ # 
        """
        Args:
            action (int): arm index as the action

        Returns:
            Step: step information
        """

        if action<0:
            return Step(0)

        action: np.int64 = np.clip(action, 0, len(self)-1)
        predicate: float = self._rng.random()
        if predicate<self._probabilities[action]:
            reward = 1
        else:
            reward = 0

        return Step(reward)
        #  }}} function step # 

    def __len__(self) -> int:
        return self._nb_arms
    #  }}} class BanditsEnv # 

class Agent(abc.ABC):
    def __call__(self, last_reward: int, total_reward: int) -> int:
        raise NotImplementedError()
    def reset(self):
        pass

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self, nb_arms: int) -> int:
        self._nb_arms: int = nb_arms
    def __call__(self, last_reward: int, total_reward: int) -> int:
        action: str = input( "Last reward: {:d}. Total reward: {:d}. Please input an arm index (There are totally {:d} arms):"\
                                .format(last_reward, total_reward, self._nb_arms)
                           )
        return int(action)
    #  }}} class ManualAgent # 

class HistoryReplay:
    #  class HistoryReplay {{{ # 
    ActionDict = Dict[ int
                     , Dict[ str
                           , Union[int, float]
                           ]
                     ]

    def __init__( self
                , item_capacity: int # H
                , action_capacity: int # A
                , gamma: float = 1. # γ
                , step_penalty: float = 0. # ω
                , learning_rate: float = 0.1 # α
                , n_step_flatten: int = 1 # n
                ):
        #  method __init__ {{{ # 
        """
        Args:
            item_capacity (int): the item capacity of the history pool signed
              as H
            action_capacity (int): the action capacity of each item in the
              history pool signed as A
            gamma (float): the discount in calculation of the value function
              signed as γ
            step_penalty (float): an optional penalty for the step counts
              signed as ω
            learning_rate (float): learning rate signed as α
            n_step_flatten (int): flatten the calculation of the estimated q
              value up to `n_step_flatten` steps signed as n
        """

        self._record: Dict[ None
                          , HistoryReplay.ActionDict
                          ] = {None: {}}

        self._gamma: float = gamma
        self._step_penalty: float = step_penalty
        self._learning_rate: float = learning_rate
        self._n_step_flatten: int = n_step_flatten
        self._multi_gamma: float = gamma ** self._n_step_flatten
        #  }}} method __init__ # 

    def __getitem__(self, last_reward: int) -> ActionDict:
        #  method __getitem__ {{{ # 
        """
        Args:
            last_reward (int): the observation

        Returns:
            ActionDict: the action-state value estimations
        """

        return self._record[None]
        #  }}} method __getitem__ # 

    def update(self, steps: List[Step], actions: List[int]):
        #  method update {{{ # 
        """
        Args:
            steps (List[Step]): list with length L of Step as the record of the
              observations and rewards
            actions (List[int]): list with length (L-1) of int as the record of
              the taken actions
        """

        assert len(steps)==len(actions)+1
        rewards = np.array( list( map( lambda st: st.reward
                                     , steps
                                     )
                                )
                          , dtype=np.float32
                          ) # (L,)
        convolved_rewards = np.convolve( rewards
                                       , np.logspace( 0, self._n_step_flatten
                                                    , num=self._n_step_flatten
                                                    , endpoint=False
                                                    , base=self._gamma
                                                    )[::-1] # (n,)
                                       , mode="full"
                                       )[self._n_step_flatten:] # (L-1,)
        for obsvt, act, n_obsvt\
                , rwd, accml_rwd in itertools.zip_longest( steps[:-1]
                                                         , actions
                                                         , steps[self._n_step_flatten:]
                                                         , rewards[1:]
                                                         , convolved_rewards
                                                         ):
            action_dict: HistoryReplay.ActionDict = self._record[None] # obsvt
            if act not in action_dict:
                action_dict[act] = { "reward": 0.
                                   , "qvalue": 0.
                                   , "number": 0
                                   }
            action_record: Dict[str, Union[int, float]] = action_dict[act]
            number: int = action_record["number"]
            number_: int = number + 1
            action_record["number"] = number_

            action_record["reward"] = float(number)/number_ * action_record["reward"]\
                                    + 1./number_ * float(rwd)

            qvalue: float = action_record["qvalue"]
            new_estimation: float = float(accml_rwd)\
                                  + self._multi_gamma * max( map( lambda rcd: rcd["qvalue"]
                                                                , self._record[None].values() # n_obsvt
                                                                )
                                                           )
            update: float = new_estimation-qvalue
            action_record["qvalue"] += self._learning_rate*update
        #  }}} method update # 

    def __str__(self) -> str:
        return yaml.dump(self._record, Dumper=yaml.Dumper)
    #  }}} class HistoryReplay # 

class AutoAgent(Agent):
    #  class AutoAgent {{{ # 
    def __init__( self
                , nb_arms: int
                , history_replay: HistoryReplay
                , prompt_template: string.Template
                , api_key: str
                , model: str = "text-davinci-003"
                , max_tokens: int = 5
                , temperature: float = 0.1
                , request_timeout: float = 3.
                ):
        #  method __init__ {{{ # 
        self._nb_arms: int = nb_arms

        self._prompt_template: string.Template = prompt_template
        self._api_key: str = api_key
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._request_timeout: float = request_timeout

        self._last_request_time: datetime.datetime = datetime.datetime.now()

        self._history_replay: HistoryReplay = history_replay
        self._action_history: List[int] = []

        openai.api_key = api_key
        #  }}} method __init__ # 

    def reset(self):
        self._action_history = []

    def __call__(self, last_reward: int, total_reward: int) -> int:
        #  method __call__ {{{ # 
        action_dict: HistoryReplay.ActionDict = self._history_replay[last_reward]
        history_str = "\n".join( ["| Actions | Rewards | Accumulated Rewards |"]\
                               + [ "| "\
                                 + " | ".join( [ str(act)
                                               , "{:.2f}".format(rcd["reward"])
                                               , "{:.2f}".format(rcd["qvalue"])
                                               ]
                                             )\
                                 + " |" for act, rcd in action_dict.items()
                                 ]
                               )
        prompt: str = self._prompt_template.safe_substitute(
                                                nb_arms=self._nb_arms
                                              , history=history_str
                                              , actions=" ".join( map( str
                                                                     , self._action_history
                                                                     )
                                                                )
                                              , reward=last_reward
                                              , total=total_reward
                                              )

        try:
            request_time = datetime.datetime.now()
            timedelta: datetime.timedelta = request_time - self._last_request_time
            timedelta: float = timedelta.total_seconds()
            if 3.1 - timedelta > 0.:
                time.sleep(3.1-timedelta)
            completion = openai.Completion.create( model=self._model
                                                 , prompt=prompt
                                                 , max_tokens=self._max_tokens
                                                 , temperature=self._temperature
                                                 , request_timeout=self._request_timeout
                                                 )
            self._last_request_time = datetime.datetime.now()

            action_text: str = completion.choices[0].text.strip()
            #action_text: str = input(prompt)
        except:
            return -1

        run_logger.debug( "Returns: {\"text\": %s, \"reason\": %s}"
                        , repr(completion.choices[0].text)
                        , repr(completion.choices[0].finish_reason)
                        )

        try:
            action = int(np.clip(int(action_text)-1, 0, self._nb_arms-1))
        except:
            action = -1

        self._action_history.append(action)
        return action
        #  }}} method __call__ # 
    #  }}} class AutoAgent # 

if __name__ == "__main__":
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument("--bandits-config", default="bandits.yaml", type=str)

    parser.add_argument("--prompt-template", type=str)
    parser.add_argument("--max-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--request-timeout", default=3., type=float)

    #parser.add_argument("--replay-file", type=str)
    #parser.add_argument("--dump-path", type=str)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options # 

    #  Logger Config {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    timestampt: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    file_handler = logging.FileHandler( os.path.join( args.log_dir
                                                    , "normal-{:}.log".format(timestampt)
                                                    )
                                      )
    debug_handler = logging.FileHandler( os.path.join( args.log_dir
                                                     , "debug-{:}.log".format(timestampt)
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

    stdout_handler.addFilter(logging.Filter("agent"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    #  }}} Logger Config # 

    #  Build Environment and Agent {{{ # 
    with open("bandits.yaml") as f:
        probabilities: List[float] = yaml.load(f, Loader=yaml.Loader)["bandits"]
    env = BanditsEnv( probs=probabilities
                    , seed=999
                    )

    history_replay = HistoryReplay( 1, len(env)
                                  , gamma=0.
                                  , step_penalty=0.
                                  , learning_rate=0.1
                                  , n_step_flatten=1
                                  )

    with open(args.prompt_template) as f:
        prompt_template = string.Template(f.read())
    with open(args.config) as f:
        openaiconfig: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
    model = AutoAgent( nb_arms=len(env)
                     , history_replay=history_replay
                     , prompt_template=prompt_template
                     , api_key=openaiconfig["api_key"]
                     , max_tokens=args.max_tokens
                     , temperature=args.temperature
                     , request_timeout=args.request_timeout
                     )
    #model = ManualAgent(nb_arms=len(env))
    #  }}} Build Environment and Agent # 

    #  Workflow {{{ # 
    max_nb_steps = 15
    nb_turns = 15
    for i in range(nb_turns):
        step_record: List[Step] = []
        action_record: List[int] = []

        model.reset()
        step: Step = env.reset()
        step_record.append(step)

        reward: int = step.reward
        for j in range(max_nb_steps):
            action: int = model(step.reward, reward)
            step = env.step(action)
            reward += step.reward

            step_record.append(step)
            action_record.append(action)

        history_replay.update(step_record, action_record)
        run_logger.info(str(history_replay))

        run_logger.info( "\x1b[42mEND!\x1b[0m TrajecId: %d, Reward: %d"
                       , i, reward
                       )
    #  }}} Workflow # 
