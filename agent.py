import vh_to_html
import re
import openai

import lxml.etree
import lxml.html
from android_env.wrappers import VhIoWrapper
from typing import Dict, Pattern, Match, List
from typing import Optional
import numpy as np

import abc
import logging
import datetime
import time

logger = logging.getLogger("agent")

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
        self._action_history: List[str] = []
        #  }}} method __init__ # 

    def reset(self):
        self._action_history.clear()

    def __call__( self
                , task: str
                , screen: lxml.etree.Element
                , instruction: str
                ) -> Dict[str, np.ndarray]:
        #  method __call__ {{{ # 
        """
        Args:
            task (str): task description
            screen (lxml.etree.Element): screen view hierarchy
            instruction (str): step instruction

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
                                                            )
                                        )
        screen_representation: str = "".join(screen_representation)

        action_str: str = self._get_action( task
                                          , screen_representation.strip()
                                          , instruction
                                          )

        self._action_history.append(action_str)

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
                   ) -> str:
        raise NotImplementedError()
    #  }}} class Agent # 

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self):
        super(ManualAgent, self).__init__()

    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
                   ) -> str:
        #  method _get_action {{{ # 
        print("Task:")
        print(task)
        print("Screen:")
        print(screen)
        print("Instruction:")
        print(instruction)
        print("Action History:")
        print("\n".join(self._action_history))

        action_str: str = input("Please input the next action:")
        return action_str
        #  }}} method _get_action # 
    #  }}} class ManualAgent # 

class AutoAgent(Agent):
    #  class AutoAgent {{{ # 
    def __init__( self
                , prompt_template: str
                , api_key: str
                , model: str = "text-davinci-003" # or maybe "engine"? not sure
                , max_tokens: int = 20
                , temperature: float = 0.1
                , request_timeout: float = 3.
                ):
        #  method __init__ {{{ # 
        """
        Args:
            prompt_template (str): template of the prompt
        """

        super(AutoAgent, self).__init__()

        self._prompt_template: str = prompt_template
        self._api_key: str = api_key
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._request_timeout: float = request_timeout

        self._last_request_time: datetime.datetime = datetime.datetime.now()

        openai.api_key = api_key
        #  }}} method __init__ # 

    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
                   ) -> str:
        #  method _get_action {{{ # 
        prompt = self._prompt_template.format( command=task
                                             , html=screen
                                             , instruction=instruction
                                             , actions="\n".join(self._action_history)
                                             )
        try:
            completion = openai.Completion.create( model=self._model
                                                 , prompt=prompt
                                                 , max_tokens=self._max_tokens
                                                 , temperature=self._temperature
                                                 , request_timeout=self._request_timeout
                                                 )
            request_time = datetime.datetime.now()
            timedelta: datetime.timedelta = request_time - self._last_request_time
            timedelta: float = timedelta.total_seconds()
            if 3.1 - timedelta > 0.:
                time.sleep(3.1-timedelta)
        except:
            return "NOTHING"

        logger.debug( "Return: {text: %s, reason: %s}"
                    , repr(completion.choices[0].text)
                    , repr(completion.choices[0].finish_reason)
                    )

        return completion.choices[0].text.strip()
        #  }}} method _get_action # 
    #  }}} class AutoAgent # 
