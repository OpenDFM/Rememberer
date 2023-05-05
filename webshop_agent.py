import abc
import logging

from typing import List, Tuple
from typing import Callable, Optional
import agent_protos

import vh_to_html
import history
import numpy as np
import tiktoken
import itertools

logger = logging.getLogger("webshop")

Key = Tuple[str, str, str] # (observation, task, available_actions)
Action = Tuple[str, str] # (action, reason)

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
    def end( self
           , task: str
           , observation: str
           , reward: float
           , total_reward: float
           , available_actions: List[str]
           ):
        pass

    def __call__( self
                , task: str
                , observation: str
                , reward: float
                , total_reward: float
                , available_actions: List[str]
                ) -> str:
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

        action_tuple: Action = self._get_action( task
                                             , self._preprocess_observation(observation)
                                             , reward
                                             , total_reward
                                             , available_actions
                                             )
        action_str: str = action_tuple[0]

        if action_str!="NOTHINGG":
            self._action_history.append(action_tuple)
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
        return action_str, "something"
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
                , static: bool = False
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

        self._input_length_limit: int = 3700

        self._tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model)
        super(agent_protos.OpenAIClient, self).__init__( history_replay
                                                       , train
                                                       , self._tokenizer
                                                       )

        self._static: bool = static
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        #self._history_replay.new_trajectory()
    def end( self
           , task: str
           , observation: str
           , reward: float
           , total_reward: float
           , available_actions: List[str]
           ):
        #  method end {{{ # 
        observation: str = "\n".join(self._preprocess_observation(observation))
        available_actions: str = "\n".join(available_actions)
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history)>0\
                                          else None
            self._history_replay.update( (observation, task, available_actions)
                                       , reward
                                       , last_action
                                       , last_step=True
                                       )
        #  }}} method end # 

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
                                                                     , map( " ".join
                                                                          , action_history[-min(5, len(action_history)):]
                                                                          )
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

    def _random_action(self, key: Key, encourages: bool = False) -> Action:
        #  method _random_action {{{ # 
        available_actions: List[str] = key[-1].splitlines()
        action: np.int64 = self._rng.integers(len(available_actions))
        if encourages:
            if available_actions[action]=="search":
                action_str: str = "search[{:}]".format(key[1])
                reason: str = ""
            else:
                action_str: str = "click[{:}]".format(available_actions[action])
                if available_actions[action]=="< prev":
                    reason: str = "The current item doesn't offer the desired options and I need to go back to check other items."
                elif available_actions[action]=="back to search":
                    reason: str = "The current item doesn't offer the desired options and I need to search for other items."
                elif available_actions[action]=="buy now":
                    reason: str = "All the options are ready now and I will click \"buy now\" to complete the shopping."
                else:
                    reason: str = "{:} conforms to the instruction.".format(available_actions[action])
        else:
            action_str: str = "click[{:}]".format(available_actions[action])
            if available_actions[action]=="search":
                reason: str = "The search button shouldn't be clicked."
            elif available_actions[action]=="features":
                reason: str = "There is no need to check the features."
            elif available_actions[action]=="description":
                reason: str = "There is no need to check the description."
            elif available_actions[action]=="reviews":
                reason: str = "There is no need to review."
            elif available_actions[action]=="buy now":
                reason: str = "Not all the requirements are ready now."
            elif available_actions[action]=="< prev":
                reason: str = "The current item offers the desired options and I don't need to go back to check other items."
            elif available_actions[action]=="back to search":
                reason: str = "The current item offers the desired options and I don't need to search for other items."
            else:
                reason: str = "{:} is not the desired item.".format(available_actions[action])
        return (action_str, reason)
        #  }}} method _random_action # 

    def _action_to_string(self, action: Action, value: float) -> str:
        return "{:} -> {:.1f} {:}".format(action[0], value, action[1])

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
        encouraged_result: List[str] = encouraged_result.strip().splitlines()

        encouraged_texts: List[Tuple[str, float, str]] = []
        for rst in encouraged_result:
            action_text: str
            action_tail: str
            action_text, action_tail = rst.split("->", maxsplit=1)

            action_text = action_text.strip()

            action_tail: List[str] = action_tail.strip().split(maxsplit=1)
            score: float = float(action_tail[0].strip())
            element_html: str = action_tail[1].strip() if len(action_tail)>1 else "" # action reason

            encouraged_texts.append((action_text, score, element_html))

        highest_result: Tuple[str, float, str]\
                = list( itertools.islice( sorted( encouraged_texts
                                                , key=(lambda itm: itm[1])
                                                , reverse=True
                                                )
                                        , 1
                                        )
                      )[0]
        return highest_result[0], highest_result[2]
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
        if self._static:
            examplars: List[str] = [ "Example 2:\n\n" + self._prompt_templates.canonical2
                                   , "Example 1:\n\n" + self._prompt_templates.canonical1
                                   ]
        else:
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
            reason: str = ""
        else:
            action_text: str
            reason: str
            action_text, reason = action

        logger.debug("Action: %s %s", action_text, reason)
        return (action_text, reason)
        #  }}} method _get_action # 

    def train(self, train: bool):
        super(agent_protos.OpenAIClient, self).train(train)
        #self._temperature = self._config_temperature if self._train else 0.
    #  }}} class AutoAgent # 
