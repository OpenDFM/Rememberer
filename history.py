from typing import Dict, Tuple
import numpy as np
import abc
import dm_env

class Matcher(abc.ABC):
    #  class Matcher {{{ # 
    def __init__(self, query: "HistoryReplay.Key"):
        #  method __init__ {{{ # 
        self._query = query
        #  }}} method __init__ # 

    def __call__(self, key: "HistoryReplay.Key") -> float:
        raise NotImplementedError
    #  }}} class Matcher # 

class HistoryReplay:
    #  class HistoryReplay {{{ # 
    Key = Tuple[ str # screen representation
               , str # task description
               , str # step instruction
               ]
    ActionDict = Dict[ str
                     , Dict[ str
                           , Union[int, float]
                           ]
                     ]

    def __init__( self
                , item_capacity: int
                , action_capacity: int
                , matcher: type(Matcher)
                , gamma: float = 1.
                , step_penalty: float = 0.
                , update_mode: str = "mean"
                , learning_rate: float = 0.1
                , n_step_flatten: int = 1
                ):
        #  method __init__ {{{ # 
        """
        Args:
            item_capacity (int): the item capacity of the history pool
            action_capacity (int): the action capacity of each item in the
              history pool
            matcher (type(Matcher)): matcher constructor

            gamma (float): the discount in calculation of the value function
            step_penalty (float): an optional penalty for the step counts

            update_mode (str): "mean" or "const"
            learning_rate (float): learning rate
            n_step_flatten (int): flatten the calculation of the estimated q
              value up to `n_step_flatten` steps
        """

        self._record: Dict[ HistoryReplay.Key
                          , HistoryReplay.ActionDict
                          ] = {}

        self._item_capacity: int = item_capacity
        self._action_capacity: int = action_capacity
        self._matcher: type(Matcher) = matcher

        self._gamma: float = gamma
        self._multi_gamma: float = gamma ** n_step_flatten
        self._filter: np.ndarray = np.logspace( 0, n_step_flatten
                                              , num=n_step_flatten
                                              , endpoint=False
                                              , base=self._gamma
                                              )[::-1] # (n,)

        self._step_penalty: float = step_penalty

        self._update_mode: str = update_mode
        self._learning_rate: float = learning_rate
        self._n_step_flatten: int = n_step_flatten

        self._action_buffer: Deque[Optional[str]] = collections.deque(maxlen=self._n_step_flatten)
        self._observation_buffer: Deque[HistoryReplay.Key]\
                = collections.deque(maxlen=self._n_step_flatten+1)
        self._reward_buffer: Deque[float] = collections.deque(maxlen=self._n_step_flatten)
        #  }}} method __init__ # 

    def __getitem__(self, request: Key) ->\
            List[ Tuple[ ActionDict
                       , float
                       ]
                ]:
        #  method __getitem__ {{{ # 
        """
        Args:
            request (Key): the observation

        Returns:
            List[Tuple[ActionDict, float]]: the retrieved action-state value
              estimations sorted by matching scores
        """

        matcher: Matcher = self._matcher(request)
        match_scores: List[float] =\
                list( map( matcher
                         , self._record.keys()
                         )
                    )
        candidates: List[ Tuple[ ActionDict
                               , float
                               ]
                        ] = list( sorted( zip( self._record.keys()
                                             , match_scores
                                             )
                                        , key=(lambda itm: itm[1])
                                        )
                                )
        return candidates
        #  }}} method __getitem__ # 

    def update(self, step: dm_env.TimeStep, action: Optional[str]):
        #  method update {{{ # 
        # TODO
        pass
        #  }}} method update # 
    #  }}} class HistoryReplay # 
