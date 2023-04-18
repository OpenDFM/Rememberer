import vh_to_html
import re
import tiktoken
import itertools

import lxml.etree
import lxml.html
from android_env.wrappers import VhIoWrapper
from typing import Dict, Pattern, Match, List, Tuple
from typing import Optional
import numpy as np
import history
import agent_protos

import abc
import logging
import datetime
import time

logger = logging.getLogger("wikihow")

Key = Tuple[str, str, str] # (observation, task, instruction)
Action = Tuple[str, str]

class Agent(abc.ABC):
    #  class Agent {{{ # 
    def __init__( self
                #, prompt_template: str
                ):
        #  method __init__ {{{ # 
        """
        Args:
            #prompt_template (str): template of the prompt
        """

        #self._prompt_template: str = prompt_template

        self._action_pattern: Pattern[str] =\
                re.compile(r"^(?P<atype>\w+)\((?P<arg1>\w+)(?:,\s*(?P<arg2>.+))?\)$")
        self._action_history: List[Action] = []
        #  }}} method __init__ # 

    def reset(self):
        self._action_history.clear()

    def __call__( self
                , task: str
                , screen: lxml.etree.Element
                , instruction: str
                , reward: float
                , total_reward: float
                ) -> Dict[str, np.ndarray]:
        #  method __call__ {{{ # 
        """
        Args:
            task (str): task description
            screen (lxml.etree.Element): screen view hierarchy
            instruction (str): step instruction
            reward (float): the last reward
            total_reward (float): the total history reward

        Returns:
            Dict[str, np.ndarray]: dict like
              {
                "action_type": NOTHING
              } or
              {
                "action_type": CLICK
                "element_id": int
              } or
              {
                "action_type": INPUT
                "element_id": int
                "text": str
              } or
              {
                "action_type": SCROLL
                "direction": Direction
              }
              all the values in `action` are wrapped in np.ndarray.
        """

        html_elements: List[lxml.html.Element] =\
                vh_to_html.convert_tree(screen)[0]

        screen_representation: List[str] = []
        for html in html_elements:
            screen_representation.append( lxml.html.tostring( html
                                                            , pretty_print=True
                                                            , encoding="unicode"
                                                            ).strip()\
                                                             .replace("\n", "&#10;")\
                                                             .replace("\r", "&#13;")
                                        )
        screen_representation: str = "\n".join(screen_representation)

        action_tuple: Action = self._get_action( task
                                          , screen_representation.strip()
                                          , instruction
                                          , reward
                                          , total_reward
                                          )
        action_str: str = action_tuple[0]

        if action_str=="NOTHINGG":
            return { "action_type": np.array(VhIoWrapper.ActionType.NOTHING)
                   , "records": False
                   }

        self._action_history.append(action_tuple)

        if action_str=="GOBACK":
            return {"action_type": np.array(VhIoWrapper.ActionType.GOBACK)}

        action_match: Match[str] = self._action_pattern.match(action_str)
        if action_match is not None:
            action_type: Optional[str] = action_match.group("atype")
            argument1: Optional[str] = action_match.group("arg1")
            argument2: Optional[str] = action_match.group("arg2")
            if action_type=="CLICK":
                if len(html_elements)>0\
                        and argument1 is not None\
                        and argument1.isdecimal():
                    return { "action_type": np.array(VhIoWrapper.ActionType.CLICK)
                           , "element_id": np.clip( np.array(int(argument1))
                                                  , 0
                                                  , len(html_elements)-1
                                                  )
                           }
            if action_type=="INPUT":
                if len(html_elements)>0\
                        and argument1 is not None\
                        and argument1.isdecimal()\
                        and argument2 is not None:
                    return { "action_type": np.array(VhIoWrapper.ActionType.INPUT)
                           , "element_id": np.clip( np.array(int(argument1))
                                                  , 0
                                                  , len(html_elements)-1
                                                  )
                           , "text": np.array(argument2, dtype=np.object_)
                           }
            if action_type=="SCROLL":
                if argument1 is not None\
                        and argument1.upper() in { "LEFT"
                                                 , "UP"
                                                 , "RIGHT"
                                                 , "DOWN"
                                                 }:
                    return { "action_type": np.array(VhIoWrapper.ActionType.SCROLL)
                           , "direction": np.array(VhIoWrapper.ScrollDirection[argument1.upper()])
                           }
        return {"action_type": np.array(VhIoWrapper.ActionType.NOTHING)}
        #  }}} method __call__ # 

    @abc.abstractmethod
    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
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
                   , task: str
                   , screen: str
                   , instruction: str
                   , reward: float
                   , total_reward: float
                   ) -> Action:
        #  method _get_action {{{ # 
        print("Task:")
        print(task)
        print("Action History:")
        print("\n".join(map(lambda itm: itm[0], self._action_history)))
        print("Screen:")
        print(screen)
        print("Instruction:")
        print(instruction)
        print("Last Reward:")
        print("{:.1f}".format(reward))
        print("Total Reward:")
        print("{:.1f}".format(total_reward))

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
                , request_timeout: float = 3.
                , manual: bool = False
                , train: bool = True
                , with_speech: bool = False
                ):
        #  method __init__ {{{ # 
        """
        Args:
            history_replay (history.HistoryReplay[Key, Action]): history replay
            prompt_templates (agent_protos.TemplateGroup): templates for the prompt

            api_key (str): openai api key
            model (str): the model to use
            max_tokens (int): max number of tokens to generate
            temperature (float): generating temperature
            stop (Optional[str]): stop sequence for the model
            request_timeout (float): waiting time for the client to timeout

            manual (bool): if a human is waiting the prompt to decide instead
              of sending it to the model
            train (bool): indicats whether the history replay should be updated
              or not

            with_speech (bool): whether the speech wrapper should be used
              instead or not
        """

        super(AutoAgent, self).__init__()
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
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        self._history_replay.new_trajectory()

    def _instantiate_input_template( self
                                   , command: str
                                   , html: str
                                   , instruction: str
                                   , action_history: List[Action]
                                   , reward: float
                                   , total_reward: float
                                   ) -> str:
        #  method _instantiate_input_template {{{ # 
        return self._prompt_templates.input_template.safe_substitute(
                                                        command=command
                                                      , html=html
                                                      , instruction=instruction
                                                      , actions=\
                                                            "\n".join(
                                                                map( " ".join
                                                                   , action_history[-min(5, len(action_history)):]
                                                                   )
                                                              )
                                                      , reward="{:.1f}".format(reward)
                                                      , total_reward="{:.1f}".format(total_reward)
                                                      )
        #  }}} method _instantiate_input_template # 

    def _random_action( self, key: Key) -> Action:
        #  method _random_action {{{ # 
        screen: str = key[0]
        elements: List[str] = screen.splitlines()
        action: np.int64 = self._rng.integers(len(elements)+4)

        directions = ["LEFT", "UP", "RIGHT", "DOWN"]
        if action<4:
            return ( "SCROLL({:})".format(directions[action])
                   , ""
                   )
        return ( "CLICK({:d})".format(action-4)
               , elements[action-4].strip()
               )
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
                      + self._instantiate_input_template( command=key[1]
                                                        , html=key[0]
                                                        , instruction=key[2]
                                                        , action_history=info_dict["action_history"]
                                                        , reward=info_dict["last_reward"]
                                                        , total_reward=info_dict["total_reward"]
                                                        )\
                      + "\n"\
                      + self._prompt_templates.advice_template.safe_substitute(
                                                                 encouraged=encouraged
                                                               , discouraged=discouraged
                                                               )

        return examplar
        #  }}} method _examplar_to_string # 

    def _parse_action(self, response: str) -> Action:
        #  Parse Action Text {{{ # 
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
            element_html: str = action_tail[1].strip() if len(action_tail)>1 else ""

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
        #return action_text, element_html
        #  }}} Parse Action Text # 

    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
                   , reward: float
                   , total_reward: float
                   ) -> Action:
        #  method _get_action {{{ # 
        #  Replay Updating {{{ # 
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history)>0\
                                          else None
            self._history_replay.update( (screen, task, instruction)
                                       , reward
                                       , last_action
                                       )
        #  }}} Replay Updating # 

        #  Construct New Input {{{ # 
        new_input: str = self._instantiate_input_template( command=task
                                                         , html=screen
                                                         , instruction=instruction
                                                         , action_history=self._action_history
                                                         , reward=reward
                                                         , total_reward=total_reward
                                                         ).strip()
        nb_new_input_tokens: int = len(self._tokenizer.encode(new_input))
        example_tokens_limit: int = self._input_length_limit - nb_new_input_tokens
        #  }}} Construct New Input # 

        #  Construct Examplars {{{ # 
        examplars: List[str] = self._get_examplars( (screen, task, instruction)
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
            element_html: str = ""
        else:
            action_text: str
            element_html: str
            action_text, element_html = action

        logger.debug("Action: %s %s", action_text, element_html)

        return (action_text, element_html)
        #  }}} method _get_action # 

    def train(self, train: bool):
        super(agent_protos.OpenAIClient, self).train(train)
    #  }}} class AutoAgent # 

class ReplayAgent(Agent):
    #  class ReproducingAgent {{{ # 
    def __init__(self, replay_files: List[str]):
        #  method __init__ {{{ # 
        super(ReplayAgent, self).__init__()

        self._replay: List[List[Action]] = []
        for rpl_f in replay_files:
            logger.debug("File: %s", rpl_f)
            self._replay.append([])
            with open(rpl_f) as f:
                for l in f:
                    #log_item: Dict[str, str] = json.loads(l)
                    #self._replay[-1].append(log_item["text"].strip())
                    logger.debug("Replay: %s", l.strip())
                    items: List[str] = l.strip().split("<->", maxsplit=1)
                    action: str = items[0].strip()
                    element: str = "" if len(items)==1 else items[1].strip()
                    self._replay[-1].append((action, element))

        self._replay_index: int = -1
        self._index: int = -1

        self._last_request_time: datetime.datetime = datetime.datetime.now()
        #  }}} method __init__ # 

    def reset(self):
        super(ReplayAgent, self).reset()
        self._replay_index += 1
        self._replay_index %= len(self._replay)
        self._index = -1

    def _get_action(self, *args) -> Action:
        #  method _get_action {{{ # 
        request_time = datetime.datetime.now()
        timedelta: datetime.timedelta = request_time - self._last_request_time
        timedelta: float = timedelta.total_seconds()
        if 3.1 - timedelta > 0.:
            time.sleep(3.1-timedelta)
        self._last_request_time = datetime.datetime.now()
        self._index += 1
        self._index %= len(self._replay[self._replay_index])
        logger.debug("Action: %s %s", self._replay[self._replay_index][self._index][0]
                                    , self._replay[self._replay_index][self._index][1]
                    )
        return self._replay[self._replay_index][self._index]
        #  }}} method _get_action # 
    #  }}} class ReproducingAgent # 
