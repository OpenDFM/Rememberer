import vh_to_html
import lxml.etree
from android_env.wrappers import VhIoWrapper
from typing import Dict

class Agent:
    def __init__( self
                , prompt_template: str
                ):
        #  method __init__ {{{ # 
        """
        Args:
            prompt_template (str): template of the prompt
        """

        self._prompt_pattern: str = prompt_pattern
        #  }}} method __init__ # 

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
        #  }}} method __call__ # 
