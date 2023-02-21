import vh_to_html
import re

import lxml.etree
import lxml.html
from android_env.wrappers import VhIoWrapper
from typing import Dict, Pattern, Match, List
from typing import Optional
import numpy as np

class Agent:
    def __init__( self
                , prompt_template: str
                ):
        #  method __init__ {{{ # 
        """
        Args:
            prompt_template (str): template of the prompt
        """

        self._prompt_template: str = prompt_template

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

        print("Task:")
        print(task)
        print("Screen:")
        print(screen_representation.strip())
        print("Instruction:")
        print(instruction)
        print("Action History:")
        print("\n".join(self._action_history))

        action_str: str = input("Please input the next action:")
        self._action_history.append(action_str)

        action_match: Match[str] = self._action_pattern.match(action_str)
        if action_match is not None:
            action_type: Optional[str] = action_match.group("atype")
            argument1: Optional[str] = action_match.group("arg1")
            argument2: Optional[str] = action_match.group("arg2")
            if action_type=="CLICK":
                if argument1 is not None\
                        and argument1.isdecimal():
                    return { "action_type": np.array(VhIoWrapper.ActionType.CLICK)
                           , "element_id": np.clip( np.array(int(argument1))
                                                  , 0
                                                  , len(html_elements)
                                                  )
                           }
            if action_type=="INPUT":
                if argument1 is not None\
                        and argument1.isdecimal()\
                        and argument2 is not None:
                    return { "action_type": np.array(VhIoWrapper.ActionType.INPUT)
                           , "element_id": np.clip( np.array(int(argument1))
                                                  , 0
                                                  , len(html_elements)
                                                  )
                           , "text": np.array(argument2, dtype=np.object_)
                           }
            if action_type=="SCROLL":
                if argument1 is not None\
                        and argument1 in { "LEFT"
                                         , "UP"
                                         , "RIGHT"
                                         , "DOWN"
                                         }:
                    return { "action_type": np.array(VhIoWrapper.ActionType.SCROLL)
                           , "direction": np.array(VhIoWrapper.Direction[argument1])
                           }
        return {"action_type": np.array(VhIoWrapper.ActionType.NOTHING)}
        #  }}} method __call__ # 
