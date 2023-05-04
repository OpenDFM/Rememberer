import abc
import logging

logger = logging.getLogger("alfworld")

#Key = 
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
           ):
        pass

    def __call__( self
                , init_env: str
                , task: str
                , trajectory: str
                ) -> str:
        #  method __call__ {{{ # 
        """
        Args:
            init_env (str): the description of the initial state of the
              environment
            task (str): the task goal
            trajectory (str): the trajectory. the taken actions are prefixed
              with `> `. the last line is the last observation return.

        Returns:
            str: the action to take
        """

        action_tuple: Action = self._get_action( init_env
                                               , task
                                               , trajectory
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
                   ) -> Action:
        raise NotImplementedError()

    def train(self, train: bool):
        pass
    #  }}} class Agent # 
