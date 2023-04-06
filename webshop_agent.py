import abc
import logging

from typing import List

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
                                             , observation
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
        print(observation)
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
