#!/usr/bin/python3

from typing import Sequence, Optional, Union
from typing import NamedTuple, List, Dict, Deque, Set
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
#import itertools
import collections

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

    def refresh(self, seed: Optional[int] = None) -> Step:
        #  function refresh {{{ # 
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
        #  }}} function refresh # 

    def reset(self) -> Step:
        return Step(reward=0)

    def step(self, action: int) -> Step:
        #  function step {{{ # 
        """
        Args:
            action (int): arm index as the action

        Returns:
            Step: step information
        """

        if action==0:
            return Step(0)

        action: np.int64 = np.clip(action, 1, len(self))-1
        predicate: float = self._rng.random()
        if predicate<self._probabilities[action]:
            reward = 1
        else:
            reward = 0

        run_logger.debug("%d, %.3f, %.2f", action, predicate, self._probabilities[action])

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
        self._multi_gamma: float = gamma ** n_step_flatten
        self._filter: np.ndarray = np.logspace( 0, n_step_flatten
                                              , num=n_step_flatten
                                              , endpoint=False
                                              , base=self._gamma
                                              )[::-1] # (n,)

        self._step_penalty: float = step_penalty
        self._learning_rate: float = learning_rate
        self._n_step_flatten: int = n_step_flatten

        self._action_buffer: Deque[Optional[int]] = collections.deque(maxlen=self._n_step_flatten)
        self._observation_buffer: Deque[int] = collections.deque(maxlen=self._n_step_flatten+1)
        self._reward_buffer: Deque[int] = collections.deque(maxlen=self._n_step_flatten)
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

    def update(self, step: Step, action: Optional[int] = None):
        #  method update {{{ # 
        """
        Args:
            step (Step): the new state transitted to after `action` is
              performed
            action (Optional[int]): the performed action, may be null if it is
              the initial state
        """

        self._action_buffer.append(action)
        self._observation_buffer.append(step.reward)
        self._reward_buffer.append(step.reward)
        if len(self._observation_buffer)<self._observation_buffer.maxlen:
            return

        observation: int = self._observation_buffer[0]
        action: int = self._action_buffer[0]
        observation_: int = self._observation_buffer[-1]
        reward: int = self._reward_buffer[0]

        action_dict: HistoryReplay.ActionDict = self._record[None] # observation
        if action not in action_dict:
            action_dict[action] = { "reward": 0.
                                  , "qvalue": 0.
                                  , "number": 0
                                  }
        action_record: Dict[str, Union[int, float]] = action_dict[action]

        number: int = action_record["number"]
        number_: int = number + 1
        action_record["number"] = number_

        action_record["reward"] = float(number)/number_ * action_record["reward"]\
                                + 1./number_ * float(reward)

        qvalue: float = action_record["qvalue"]
        new_estimation: float = float( np.convolve( np.asarray(self._reward_buffer, dtype=np.float32) # (n,)
                                                  , self._filter # (n,)
                                                  , mode="valid"
                                                  )[0]
                                     )\
                              + self._multi_gamma * max( map( lambda rcd: rcd["qvalue"]
                                                            , self._record[None].values() # observation_
                                                            )
                                                       )
        update: float = new_estimation-qvalue
        action_record["qvalue"] += self._learning_rate*update
        #  }}} method update # 

    def new_trajectory(self):
        #  method new_trajectory {{{ # 
        if len(self._action_buffer)<=1:
            self._action_buffer.clear()
            self._observation_buffer.clear()
            self._reward_buffer.clear()
            return

        if self._action_buffer[0] is None:
            self._action_buffer.popleft()
            self._reward_buffer.popleft()

        rewards = np.asarray(self._reward_buffer, dtype=np.float32) # (n',)
        convolved_rewards = np.convolve( rewards, self._filter
                                       , mode="full"
                                       )[self._n_step_flatten-1:] # (n',)

        for obsvt, act, rwd, cvl_rwd in zip( self._observation_buffer[:-1]
                                           , self._action_buffer
                                           , self._reward_buffer
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
            new_estimation: float = float(cvl_rwd)
            update: float = new_estimation-qvalue
            action_record["qvalue"] += self._learning_rate*update

        self._action_buffer.clear()
        self._observation_buffer.clear()
        self._reward_buffer.clear()
        #  }}} method new_trajectory # 

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
                , advantage_threshold: float = 0.1
                , exploration_threshold: int = 3
                , model: str = "text-davinci-003"
                , max_tokens: int = 5
                , temperature: float = 0.1
                , request_timeout: float = 3.
                ):
        #  method __init__ {{{ # 
        self._nb_arms: int = nb_arms
        self._action_list: str = " ".join(map(str, range(1, self._nb_arms+1)))

        self._advantage_threshold: float = advantage_threshold
        self._exploration_threshold: int = exploration_threshold

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
        self._rng: np.random.Generator = np.random.default_rng()
        #  }}} method __init__ # 

    def reset(self):
        self._action_history = []
        self._history_replay.new_trajectory()

    def __call__(self, last_reward: int, total_reward: int) -> int:
        #  method __call__ {{{ # 
        action_dict: HistoryReplay.ActionDict = self._history_replay[last_reward]

        actions_by_advantage: List[int] = list(sorted( action_dict
                                                     , key=(lambda act: action_dict[act]["reward"])
                                                     , reverse=True
                                                     )
                                              )
        if len(actions_by_advantage)==0:
            action_advices: Set[int] = set()
        elif len(actions_by_advantage)==1:
            action_advices: Set[int] = set(actions_by_advantage)
        else:
            action_advices: Set[int] = set(actions_by_advantage[0:1])\
                                            if actions_by_advantage[0]-actions_by_advantage[1]>self._advantage_threshold\
                                          else set(actions_by_advantage[0:2])
        avg_exploration_time: np.float64 = np.mean( np.asarray(
                                                        list( map( lambda rcd: rcd["number"]
                                                                 , action_dict.values()
                                                                 )
                                                            )
                                                      )
                                                  )
        #run_logger.info(str(action_advices))
        less_explored_actions: Set[int] = set( filter( lambda act:
                                                          ( avg_exploration_time-action_dict[act]["number"]\
                                                                  if act in action_dict\
                                                                  else avg_exploration_time
                                                          ) >= self._exploration_threshold
                                                     , range(1, self._nb_arms+1)
                                                     )
                                             )
        interaction: Set[int] = action_advices & less_explored_actions
        symmetric: Set[int] = action_advices ^ less_explored_actions
        #priors: np.ndarray = self._rng.permutation(np.array(list(interaction), dtype=np.int32))
        #inferiors: np.ndarray = self._rng.permutation(np.array(list(symmetric), dtype=np.int32))
        #action_advices = np.concatenate([priors, inferiors])
        weight = np.concatenate( [ np.full((len(interaction),), 2.)
                                 , np.full((len(symmetric),), 1.)
                                 ]
                               )
        if len(weight)>0:
            weight /= np.sum(weight)
            action_advices = self._rng.choice( np.concatenate( [ np.array(list(interaction), dtype=np.int32)
                                                               , np.array(list(symmetric), dtype=np.int32)
                                                               ]
                                                             )
                                             , size=(len(interaction)+len(symmetric),)
                                             , replace=False
                                             , p=weight
                                             )
        else:
            action_advices = np.empty((0,))

        prompt: str = self._prompt_template.safe_substitute(
                                                nb_arms=self._nb_arms
                                              , action_list=self._action_list
                                              , advice=" ".join(map(str, action_advices))
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

            run_logger.debug( "Returns: {\"text\": %s, \"reason\": %s}"
                            , repr(completion.choices[0].text)
                            , repr(completion.choices[0].finish_reason)
                            )

            action = int(np.clip(int(action_text), 1, self._nb_arms))
        except:
            action = 0

        self._action_history.append(action)
        run_logger.debug("Action: %d", action)

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
    max_nb_steps = 25
    model.reset()
    step: Step = env.reset()
    history_replay.update(step)
    reward: int = step.reward
    for i in range(max_nb_steps):
        action: int = model(step.reward, reward)
        step = env.step(action)
        reward += step.reward

        history_replay.update(step, action)

    run_logger.info(str(history_replay))

    run_logger.info( "\x1b[42mEND!\x1b[0m TrajecId: %d, Reward: %d"
                   , i, reward
                   )
    #  }}} Workflow # 
