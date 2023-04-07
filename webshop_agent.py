import abc
import logging

from typing import List
import vh_to_html

logger = logging.getLogger("webshop")
ocounter = 0
ologger = logging.getLogger("openaiE")

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
                ) -> Action:
        #  method __call__ {{{ # 
        """
        Args:
            task (str): task instruction
            observation (str): observation
            reward (float): the last reward
            total_reward (float): the total history reward
        """

        action_str: Action = self._get_action( task
                                             , self._preprocess_observation(observation)
                                             , reward
                                             , total_reward
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

        action_str: str = input("Please input the next action:")
        return action_str
        #  }}} method _get_action # 
    #  }}} class ManualAgent # 
