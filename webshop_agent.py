import abc
import logging

from typing import List
from typing import Callable, Optional
import agent_protos

import vh_to_html
import history
import numpy as np
import tiktoken

logger = logging.getLogger("webshop")
ocounter = 0
ologger = logging.getLogger("openaiE")

Key = str
Action = str

class Agent(abc.ABC):
    #  class Agent {{{ # 
    def __init__(self, env_mode: str):
        #  method __init__ {{{ # 
        self._action_history: List[Action] = []
        self._env_mode: str = env_mode

        self._preprocess_observation: Callable[[str], List[str]]
        if env_mode=="html":
            self._preprocess_observation = vh_to_html.simplify_html
        elif env_mode=="text":
            self._preprocess_observation = vh_to_html.convert_simple_page
        elif env_mode=="text_rich":
            self._preprocess_observation = lambda url: [url]
        elif env_mode=="url":
            self._preprocess_observation = lambda url: [url]
        #  }}} method __init__ # 

    def reset(self):
        self._action_history.clear()

    def __call__( self
                , task: str
                , observation: str
                , reward: float
                , total_reward: float
                , available_actions: List[str]
                ) -> Action:
        #  method __call__ {{{ # 
        """
        Args:
            task (str): task instruction
            observation (str): observation
            reward (float): the last reward
            total_reward (float): the total history reward
            available_actions (List[str]): available_actions on the current observation

        Returns:
            Action: the action to take
        """

        action_str: Action = self._get_action( task
                                             , self._preprocess_observation(observation)
                                             , reward
                                             , total_reward
                                             , available_actions
                                             )

        self._action_history.append(action_str)
        return action_str
        #  }}} method __call__ # 

    @abc.abstractmethod
    def _get_action( self
                   , task: str
                   , observation: str
                   , reward: float
                   , total_reward: float
                   , available_actions: List[str]
                   ) -> Action:
        raise NotImplementedError()
    #  }}} class Agent # 

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self, env_mode: str):
        super(ManualAgent, self).__init__(env_mode)

    def _get_action( self
                   , task: str
                   , observation: str
                   , reward: float
                   , total_reward: float
                   , available_actions: List[str]
                   ) -> Action:
        #  method _get_action {{{ # 
        print("Task:")
        print(task)
        print("Observation:")
        print("\n".join(observation))
        print("Action History:")
        print("\n".join(self._action_history))
        print("Last Reward:")
        print("{:.1f}".format(reward))
        print("Total Reward:")
        print("{:.1f}".format(total_reward))
        print("Available Action:")
        print(", ".join(available_actions))

        action_str: str = input("Please input the next action:")
        return action_str
        #  }}} method _get_action # 
    #  }}} class ManualAgent # 

class AutoAgent( Agent
               , agent_protos.OpenAIClient[Action]
               , agent_protos.HistoryReplayClient[Key, Action]
               ):
    #  class AutoAgent {{{ # 
    def __init__( self
                , history_replay: history.HistoryReplay[Key, Action]
                , prompt_templates: agent_protos.TemplateGroup
                , api_key: str
                , model: str = "text-davinci-003"
                , max_tokens: int = 20
                , temperature: float = 0.1
                , stop: Optional[str] = None
                , request_timeout: float = 5.
                , manual: bool = False
                , train: bool = True
                , with_speech: bool = False
                , env_mode: str = "text_rich"
                ):
        #  method __init__ {{{ # 
        super(AutoAgent, self).__init__(env_mode)
        super(Agent, self).__init__( prompt_templates
                                   , api_key
                                   , model
                                   , max_tokens
                                   , temperature
                                   , stop
                                   , request_timeout
                                   , 3.1
                                   , with_speech
                                   , manual
                                   )

        # TODO: adjust the length limit according to the preamble
        # self._input_length_limit: int = 3700

        self._tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model)
        super(agent_protos.OpenAIClient, self).__init__( history_replay
                                                       , train
                                                       , self._tokenizer
                                                       )
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        self._history_replay.new_trajectory()

    def _parse_action(self, response: str) -> Action:
        return response

    def _get_action( self
                   , task: str
                   , observation: str
                   , reward: float
                   , total_reward: float
                   , available_actions: List[str]
                   ) -> Action:
        #  method _get_action {{{ # 
        #  Replay Updating {{{ # 
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history)>0\
                                          else None
            self._history_replay.update( observation
                                       , reward
                                       , last_action
                                       )
        #  }}} Replay Updating # 

        # TODO
        #  }}} method _get_action # 
    #  }}} class AutoAgent # 
