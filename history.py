from typing import Dict, Tuple, Deque, List
from typing import Union, Optional
import abc
#import dm_env

import numpy as np
import collections

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
                , item_capacity: Optional[int]
                , action_capacity: Optional[int]
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
            item_capacity (Optional[int]): the optional item capacity limit of
              the history pool
            action_capacity (Optional[int]): the optional action capacity of
              each item in the history pool
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

        self._item_capacity: Optional[int] = item_capacity
        self._action_capacity: Optional[int] = action_capacity
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

        self._similarity_matrix: np.ndarray = np.zeros( (self._item_capacity, self._item_capacity)
                                                      , dtype=np.float32
                                                      )
        self._index_pool: Deque[int] = collections.deque(range(self._item_capacity))
        #self._index_dict: Dict[HistoryReplay.Key, int] = {}
        self._keys: List[HistoryReplay] = []
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
        candidates: List[ Tuple[ HistoryReplay.ActionDict
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

    def update(self, step: Key, reward: float, action: Optional[str]):
        #  method update {{{ # 
        """
        Args:
            step (Key): the new state transitted to after `action` is performed
            reward (float): the reward corresponding to the new state
            action (Optional[str]): the performed action, may be null if it is
              the initial state
        """

        self._action_buffer.append(action)
        self._observation_buffer.append(step)
        self._reward_buffer.append(reward)
        if len(self._observation_buffer)<self._observation_buffer.maxlen:
            return

        step = self._observation_buffer[0]
        action: str = self._action_buffer[0]
        #step_: HistoryReplay.Key = self._observation_buffer[-1]
        reward: float = self._reward_buffer[0]

        if not self._insert_key(step):
            return

        action_dict: HistoryReplay.ActionDict = self._record[step]
        self._update_action_record(action_dict, action, reward, 0.)
        self._prune_action(action_dict)
        #  }}} method update # 

    def new_trajectory(self):
        #  method new_trajectory {{{ # 
        if len(self._action_buffer)<1\
                or len(self._action_buffer)==1 and self._action_buffer[0] is None:
            self._action_buffer.clear()
            self._observation_buffer.clear()
            self._reward_buffer.clear()
            return

        if self._action_buffer[0] is None:
            self._action_buffer.popleft()
            self._reward_buffer.popleft()

        rewards = np.asarray(self._reward_buffer, dtype=np.float32)
        convolved_rewards = np.convolve( rewards, self._filter
                                       , mode="full"
                                       )[self._n_step_flatten-1:]

        for k, act, rwd, cvl_rwd in zip( self._observation_buffer[:-1]
                                       , self._action_buffer
                                       , self._reward_buffer
                                       , convolved_rewards
                                       ):
            if not self._insert_key(k):
                continue

            action_dict: HistoryReplay.ActionDict = self._record[k]
            self._update_action_record(action_dict, act, rwd, float(cvl_rwd))
            self._prune_action(action_dict)
        #  }}} method new_trajectory # 

    def _insert_key(self, key: Key) -> bool:
        #  method _insert_key {{{ # 
        if key not in self._record:
            #  Insertion Policy (Static Capacity Limie) {{{ # 
            matcher: Matcher = self._matcher(key)
            similarities: np.ndarray = np.asarray(list(map(matcher, self._keys)))

            if self._item_capacity is not None and self._item_capacity>0\
                    and len(self._record)==self._item_capacity:

                max_new_similarity_index: np.int64 = np.argmax(similarities)
                max_old_similarity_index: Tuple[ np.int64
                                               , np.int64
                                               ] = np.unravel_index( np.argmax(self._similarity_matrix)
                                                                   , self._similarity_matrix.shape
                                                                   )
                if similarities[max_new_similarity_index]>=self._similarity_matrix[max_old_similarity_index]:
                    # drop the new one
                    return False
                # drop an old one according to the number of action samples
                action_dict1: HistoryReplay.ActionDict = self._record[self._keys[max_old_similarity_index[0]]]
                nb_samples1: int = sum(map(lambda d: d["number"], action_dict1.values()))

                action_dict2: HistoryReplay.ActionDict = self._record[self._keys[max_old_similarity_index[1]]]
                nb_samples2: int = sum(map(lambda d: d["number"], action_dict2.values()))

                drop_index: np.int64 = max_old_similarity_index[0] if nb_samples1>=nb_samples2 else max_old_similarity_index[1]

                del self._record[self._keys[drop_index]]
                self._keys[drop_index] = key
                similarities[drop_index] = 0.
                self._similarity_matrix[drop_index, :] = similarities
                self._similarity_matrix[:, drop_index] = similarities
                self._record[key] = {}
            else:
                new_index: int = len(self._record)
                self._keys.append(key)
                self._similarity_matrix[new_index, :new_index] = similarities
                self._similarity_matrix[:new_index, new_index] = similarities
                self._record[key] = {}
            #  }}} Insertion Policy (Static Capacity Limie) # 
        return True
        #  }}} method _insert_key # 

    def _update_action_record( self
                             , action_dict: ActionDict
                             , action: str
                             , reward: float
                             , new_estimation: float
                             )\
            -> Dict[str, Union[int, float]]:
        #  method _update_action_record {{{ # 
        if action not in action_dict:
            action_dict[action] = { "reward": 0.
                                  , "number": 0
                                  }
        action_record = action_dict[action]

        number: int = action_record["number"]
        number_: int = number + 1
        action_record["number"] = number_

        if self._update_mode=="mean":
            action_record["reward"] = float(number)/number_ * action_record["reward"]\
                                    + 1./number_ * reward
        elif self._update_mode=="const":
            action_record["reward"] += self._learning_rate * (reward-action_record["reward"])
        #  }}} method _update_action_record # 

    def _prune_action(self, action_dict: ActionDict):
        #  method _remove_action {{{ # 
        if self._action_capacity is not None and self._action_capacity>0\
                and len(action_dict)>self._action_capacity:
            worst_action: str = min( action_dict
                                   , key=(lambda act: action_dict[act]["reward"])
                                   )
            del action_dict[worst_action]
        #  }}} method _remove_action # 
    #  }}} class HistoryReplay # 
