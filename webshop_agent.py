import abc
import logging

from typing import List, Tuple
from typing import Callable, Optional
import agent_protos

import vh_to_html
import history
import numpy as np
import tiktoken

logger = logging.getLogger("webshop")

Key = Tuple[str, str, str] # (observation, task, available_actions)
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

        if action_str!="NOTHINGG":
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

    def train(self, train: bool):
        pass
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

        self._config_temperature: float = temperature
        #temperature = self._config_temperature if train else 0.
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
        self._input_length_limit: int = 3870

        self._tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model)
        super(agent_protos.OpenAIClient, self).__init__( history_replay
                                                       , train
                                                       , self._tokenizer
                                                       )
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        self._history_replay.new_trajectory()

    def _instantiate_input_template( self
                                   , task: str
                                   , observation: str
                                   , action_history: List[Action]
                                   , reward: float
                                   , total_reward: float
                                   , available_actions: str
                                   ):
        #  method _instantiate_input_template {{{ # 
        return self._prompt_templates.input_template.safe_substitute(
                                                        task=task
                                                      , observation=\
                                                              "\n".join(
                                                                  map( lambda l: "  " + l
                                                                     , observation.splitlines()
                                                                     )
                                                                )
                                                      , actions=\
                                                              "\n".join(
                                                                  map( lambda act: "- " + act
                                                                     , action_history[-min(5, len(action_history)):]
                                                                     )
                                                                )
                                                      , reward="{:.1f}".format(reward)
                                                      , total_reward="{:.1f}".format(total_reward)
                                                      , available_actions=\
                                                              "\n".join(
                                                                  map( lambda act: "- " + act
                                                                     , available_actions.splitlines()
                                                                     )
                                                                )
                                                      )
        #  }}} method _instantiate_input_template # 

    def _random_action(self, key: Key) -> Action:
        #  method _random_action {{{ # 
        available_actions: List[str] = key[-1].splitlines()
        action: np.int64 = self._rng.integers(len(available_actions))
        return "click[{:}]".format(available_actions[action])
        #  }}} method _random_action # 

    def _action_to_string(self, action: Action, value: float) -> str:
        return "{:} -> {:.1f}".format(action, value)

    def _examplar_to_string( self
                           , index: int
                           , key: Key
                           , info_dict: history.HistoryReplay.InfoDict[Action]
                           , encouraged: str
                           , discouraged: str
                           ) -> str:
        #  method _examplar_to_string {{{ # 
        examplar: str = "Example {:d}:\n\n".format(index+1)\
                      + self._instantiate_input_template( task=key[1]
                                                        , observation=key[0]
                                                        , action_history=info_dict["action_history"]
                                                        , reward=info_dict["last_reward"]
                                                        , total_reward=info_dict["total_reward"]
                                                        , available_actions=key[2]
                                                        )\
                      + "\n"\
                      + self._prompt_templates.advice_template.safe_substitute(
                                                                encouraged=encouraged
                                                              , discouraged=discouraged
                                                              )
        return examplar
        #  }}} method _examplar_to_string # 

    def _parse_action(self, response: str) -> Action:
        #  method _parse_action {{{ # 
        encouraged_result: str = response.split("Disc", maxsplit=1)[0]
        encouraged_result = encouraged_result.split(":", maxsplit=1)[1]
        encouraged_result = encouraged_result.strip().splitlines()[0]
        encouraging_texts: List[str] = encouraged_result.split("->", maxsplit=1)

        action_text: str = encouraging_texts[0].strip()
        return action_text
        #  }}} method _parse_action # 

    def _get_action( self
                   , task: str
                   , observation: List[str]
                   , reward: float
                   , total_reward: float
                   , available_actions: List[str]
                   ) -> Action:
        #  method _get_action {{{ # 
        observation: str = "\n".join(observation)
        available_actions: str = "\n".join(available_actions)

        #  Replay Updating {{{ # 
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history)>0\
                                          else None
            self._history_replay.update( (observation, task, available_actions)
                                       , reward
                                       , last_action
                                       )
        #  }}} Replay Updating # 

        #  Construct New Input {{{ # 
        new_input: str = self._instantiate_input_template( task=task
                                                         , observation=observation
                                                         , action_history=self._action_history
                                                         , reward=reward
                                                         , total_reward=total_reward
                                                         , available_actions=available_actions
                                                         )
        nb_new_input_tokens: int = len(self._tokenizer.encode(new_input))
        example_tokens_limit: int = self._input_length_limit - nb_new_input_tokens
        #  }}} Construct New Input # 

        #  Construct Examplars {{{ # 
        examplars: List[str] = self._get_examplars( (observation, task, available_actions)
                                                  , example_tokens_limit
                                                  , 2
                                                  )

        example_str: str = "\n".join(reversed(examplars)).strip()
        #  }}} Construct Examplars # 

        prompt: str = self._prompt_templates.whole_template.safe_substitute( examples=example_str
                                                                           , new_input=new_input
                                                                           )
        action: Optional[Action] = self._get_response(prompt)
        if action is None:
            action_text: str = "NOTHINGG"
        else:
            action_text: str = action

        logger.debug("Action: %s", action_text)
        return action_text
        #  }}} method _get_action # 

    def train(self, train: bool):
        super(agent_protos.OpenAIClient, self).train(train)
        #self._temperature = self._config_temperature if self._train else 0.
    #  }}} class AutoAgent # 
