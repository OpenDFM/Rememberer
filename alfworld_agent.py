import abc
import logging

from typing import Tuple, List
from typing import Optional

import tiktoken
import numpy as np

import agent_protos
import history

logger = logging.getLogger("alfworld")

Key = Tuple[str, str, str, Tuple[str, ...]] # (init_env, task, trajectory, available_actions)
Action = Tuple[str, str] # (action, reason)

class Agent(abc.ABC):
    #  class Agent {{{ # 
    def __init__(self):
        self._action_history: List[Action] = []

    def reset(self):
        self._action_history.clear()
    def end( self
           , init_env: str
           , task: str
           , trajectory: str
           , available_actions: Tuple[str, ...]
           , reward: float
           ):
        pass

    def __call__( self
                , init_env: str
                , task: str
                , trajectory: str
                , available_actions: Tuple[str, ...]
                , reward: float
                ) -> str:
        #  method __call__ {{{ # 
        """
        Args:
            init_env (str): the description of the initial state of the
              environment
            task (str): the task goal
            trajectory (str): the trajectory. the taken actions are prefixed
              with `> `. the last line is the last observation return.
            reward (float): the reward of the last action

        Returns:
            str: the action to take
        """

        action_tuple: Action = self._get_action( init_env
                                               , task
                                               , trajectory.strip()
                                               , available_actions
                                               , reward
                                               )

        action_str: str = action_tuple[0]

        if action_str!="NOTHINGG":
            self._action_history.append(action_tuple)
        return action_str
        #  }}} method __call__ # 

    @abc.abstractmethod
    def _get_action( self
                   , init_env: str
                   , task: str
                   , trajectory: str
                   , available_actions: List[str]
                   , reward: float
                   ) -> Action:
        raise NotImplementedError()

    def train(self, train: bool):
        pass
    #  }}} class Agent # 

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self):
        super(ManualAgent, self).__init__()

    def _get_action( self
                   , init_env: str
                   , task: str
                   , trajectory: str
                   , available_actions: Tuple[str, ...]
                   , reward: float
                   ) -> Action:
        #  method _get_action {{{ # 
        print("Init State")
        print(init_env)
        print("Task:")
        print(task)
        print("Trajectory:")
        print(trajectory)

        action_str: str = input("Please input the next action:")
        return action_str, "something"
        #  }}} method _get_action # 
    #  }}} class ManualAgent # 

class AutoAgent( Agent
               , agent_protos.OpenAIClient[Action]
               , agent_protos.HistoryReplayClient[Key, Agent]
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
                ):
        #  method __init__ {{{ # 
        super(AutoAgent, self).__init__()
        super(Agent, self).__init__( prompt_templates
                                   , api_key
                                   , model
                                   , max_tokens
                                   , temperature
                                   , stop
                                   , request_timeout
                                   , 1.1
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
        self._static_prompts: Tuple[str, str] = ("", "")

        self._position: str = "room middle"
        self._carrying: str = ""
        self._facing: List[str] = []
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        self._position = "room middle"
        self._carrying = ""
        self._facing = []
    def end( self
           , init_env: str
           , task: str
           , trajectory: str
           , available_actions: Tuple[str]
           , reward: float
           ):
        #  method end {{{ # 
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history) > 0\
                                          else None
            self._history_replay.update( (init_env, task, trajectory, available_actions)
                                       , reward
                                       , last_action
                                       , last_step=True
                                       )
        #  }}} method end # 

    def _instantiate_input_template( self
                                   , init_env: str
                                   , task: str
                                   , trajectory: str
                                   , action_history: List[Action]
                                   , available_actions: Tuple[Action]
                                   ):
        #  method _instantiate_input_template {{{ # 
        return self._prompt_templates.input_template.safe_substitute(
                                                        task=task
                                                      , init_env=init_env
                                                      , actions=\
                                                              "\n".join(
                                                                  map( " ".join
                                                                     , action_history #[-min(5, len(action_history)):]
                                                                     )
                                                                )
                                                      , trajectory=trajectory
                                                      , available_actions="\n".join(available_actions)
                                                      )
        #  }}} method _instantiate_input_template # 
    def _random_action(self, key: Key, encourages: bool = False) -> Action:
        #  method _random_action {{{ # 
        available_actions: Tuple[str, ...] = key[-1]
        action: np.int64 = self._rng.integers(len(available_actions))
        action_str: str = available_actions[action]
        #reason: str = ("I need to " if encourages else "I shouldn't ") + action_str

        new_carrying: str = self._carrying
        new_position: str = self._position
        if action_str.startswith("go to "):
            new_position = action_str[6:]
        elif action_str.startswith("put "):
            new_carrying = ""
        elif action_str.startswith("take "):
            new_carrying = action_str[5:action_str.find("from")-1]

        if len(self._facing)>0 and self._facing[0]=="closed":
            facing_str = "The {:} is closed.".format(self._position)
        else:
            facing_str = "There {:} {:} on {:}."\
                            .format( "is" if len(self._facing)<=1 else "are"
                                   , "nothing" if len(self._facing)==0\
                                             else ( self._facing[0] if len(self._facing)==1\
                                                                  else ", ".join(self._facing[:-1])\
                                                                     + ", and " + self._facing[-1]
                                                  )
                                   , self._position
                                   )
        additional: str = "I carry {:}. I am at {:}. {:} I will carry {:}. I will be at {:}."\
                            .format( "nothing" if self._carrying=="" else self._carrying
                                   , self._position
                                   , facing_str
                                   , "nothing" if new_carrying=="" else new_carrying
                                   , new_position
                                   )
        return action_str, additional
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
                      + self._instantiate_input_template( init_env=key[0]
                                                        , task=key[1]
                                                        , trajectory=key[2]
                                                        , action_history=info_dict["action_history"]
                                                        , available_actions=key[3]
                                                        )\
                      + "\n"\
                      + self._prompt_templates.advice_template.safe_substitute(
                                                                encouraged=encouraged
                                                              , discouraged=discouraged
                                                              )
        return examplar
        #  }}} method _examplar_to_string # 
    def _parse_action(self, response: str) -> Action:
        return agent_protos.parse_action_with_optional(response)

    def _get_action( self
                   , init_env: str
                   , task: str
                   , trajectory: str
                   , available_actions: Tuple[str, ...]
                   , reward: float
                   ) -> Action:
        #  method _get_action {{{ # 
        #  Replay Updating {{{ # 
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history) > 0\
                                          else None
            self._history_replay.update( (init_env, task, trajectory, available_actions)
                                       , reward
                                       , last_action
                                       )
        #  }}} Replay Updating # 

        if len(trajectory)>0:
            last_observation: str = trajectory.splitlines()[-1]
            observation_pattern: str
            _, observation_pattern = history.AlfObvPatMatcher._get_pattern(last_observation)
            if observation_pattern in {"On the", "The _ is open.", "You open"}:
                you_see_position: int = last_observation.find("you see ")
                if you_see_position!=-1:
                    self._facing = list( map( lambda itm: " ".join(itm.strip().split()[-2:])
                                            , last_observation[you_see_position+8:].split(", ")
                                            )
                                       )
                    if self._facing[0]=="nothing":
                        self._facing = []
            elif observation_pattern=="The _ is closed.":
                self._facing = "closed"
            elif observation_pattern=="You pick":
                object_: str = last_observation[16:last_observation.find("from")-1]
                self._carrying = object_
                if object_ in self._facing:
                    self._facing.remove(object_)
            elif observation_pattern=="You put":
                self._facing.append(self._carrying)
                self._carrying = ""

        logger.debug("%s, %s, %s", self._carrying, self._position, self._facing)

        #  Construct New Input {{{ # 
        new_input: str = self._instantiate_input_template( init_env=init_env
                                                         , task=task
                                                         , trajectory=trajectory
                                                         , action_history=self._action_history
                                                         , available_actions=available_actions
                                                         )
        nb_new_input_tokens: int = len(self._tokenizer.encode(new_input))
        example_tokens_limit: int = self._input_length_limit - nb_new_input_tokens
        #  }}} Construct New Input # 

        #  Construct Exemplars {{{ # 
        if self._static:
            examplars: List[str] = [ "Example 2:\n\n" + self._static_prompts[1]
                                   , "Example 1:\n\n" + self._static_prompts[0]
                                   ]
        else:
            examplars: List[str] = self._get_examplars( (init_env, task, trajectory, available_actions)
                                                      , example_tokens_limit
                                                      , 2
                                                      )

        example_str: str = "\n".join(reversed(examplars)).strip()
        #  }}} Construct Exemplars # 

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

        if action_text!="NOTHINGG":
            if action_text.startswith("go to "):
                self._position = action_text[6:]

        logger.debug("Action: %s %s", action_text, reason)
        return (action_text, reason)
        #  }}} method _get_action # 

    def train(self, train: bool):
        super(agent_protos.OpenAIClient, self).train(train)

    def set_static_prompts(self, prompts: Tuple[str, str]):
        self._static_prompts = prompts
    #  }}} class AutoAgent # 
