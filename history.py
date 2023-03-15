from typing import Dict, Tuple, Deque, List
from typing import Union, Optional
import abc
#import dm_env

import numpy as np
import collections
import itertools
import yaml

import logging

logger = logging.getLogger("agent.history")

class Matcher(abc.ABC):
    #  class Matcher {{{ # 
    def __init__(self, query: "HistoryReplay.Key"):
        #  method __init__ {{{ # 
        self._query: HistoryReplay.Key = query
        #  }}} method __init__ # 

    def __call__(self, key: "HistoryReplay.Key") -> float:
        raise NotImplementedError
    #  }}} class Matcher # 

class LCSNodeMatcher(Matcher):
    #  class LCSNodeMatcher {{{ # 
    def __init__(self, query: "HistoryReplay.Key"):
        #  method __init__ {{{ # 
        super(LCSNodeMatcher, self).__init__(query)

        screen: str
        screen, _, _ = self._query
        self._node_sequence: List[str] = list( map( lambda n: n[1:n.index(" ")]
                                                  , screen.splitlines()
                                                  )
                                             )
        #  }}} method __init__ # 

    def __call__(self, key: "HistoryReplay.Key") -> float:
        #  method __call__ {{{ # 
        key_screen: str = key[0]
        #  }}} method __call__ # 
        key_node_sequence: List[str] = list( map( lambda n: n[1:n.index(" ")]
                                                , key_screen.splitlines()
                                                )
                                           )

        n: int = len(self._node_sequence)
        m: int = len(key_node_sequence)
        lcs_matrix: np.ndarray = np.zeros((n+1, m+1), dtype=np.int32)
        for i, j in itertools.product( range(1, n+1)
                                     , range(1, m+1)
                                     ):
            lcs_matrix[i, j] = lcs_matrix[i-1, j-1] + 1 if self._node_sequence[i]==key_node_sequence[j]\
                                                        else max( lcs_matrix[i-1, j]
                                                                , lcs_matrix[i, j-1]
                                                                )
        lcs: np.int32 = lcs_matrix[n, m]
        length: int = max(n, m)
        return float(lcs)/length
    #  }}} class LCSNodeMatcher # 

class HistoryReplay:
    #  class HistoryReplay {{{ # 
    Key = Tuple[ str # screen representation
               , str # task description
               , str # step instruction
               ]
    InfoDict = Dict[ str
                   , Union[ float
                          , int
                          , List[str]
                          ]
                   ]
    ActionDict = Dict[ str
                     , Dict[ str
                           , Union[int, float]
                           ]
                     ]
    Record = Dict[str, Union[InfoDict, ActionDict]]

    def __init__( self
                , item_capacity: Optional[int]
                , action_capacity: Optional[int]
                , matcher: type(Matcher)
                , gamma: float = 1.
                , step_penalty: float = 0.
                , update_mode: str = "mean"
                , learning_rate: float = 0.1
                , n_step_flatten: int = 1
                , action_history_update_mode: str = "shortest"
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

            action_history_update_mode (str): "longest", "shortest", "newest",
              or "oldest"
        """

        self._record: Dict[ HistoryReplay.Key
                          , HistoryReplay.Record
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

        self._action_history_update_mode: str = action_history_update_mode

        self._action_buffer: Deque[Optional[str]] = collections.deque(maxlen=self._n_step_flatten)
        self._action_history: List[str] = []
        self._observation_buffer: Deque[HistoryReplay.Key]\
                = collections.deque(maxlen=self._n_step_flatten+1)
        self._reward_buffer: Deque[float] = collections.deque(maxlen=self._n_step_flatten+1)
        self._total_reward: float = 0.
        self._total_reward_buffer: Deque[float] = collections.deque(maxlen=self._n_step_flatten+1)

        self._similarity_matrix: np.ndarray = np.zeros( (self._item_capacity, self._item_capacity)
                                                      , dtype=np.float32
                                                      )
        self._index_pool: Deque[int] = collections.deque(range(self._item_capacity))
        #self._index_dict: Dict[HistoryReplay.Key, int] = {}
        self._keys: List[HistoryReplay] = []
        #  }}} method __init__ # 

    def __getitem__(self, request: Key) ->\
            List[ Tuple[ Key
                       , Record
                       , float
                       ]
                ]:
        #  method __getitem__ {{{ # 
        """
        Args:
            request (Key): the observation

        Returns:
            List[Tuple[Key, Record, float]]: the retrieved action-state value
              estimations sorted by matching scores
        """

        matcher: Matcher = self._matcher(request)
        match_scores: List[float] =\
                list( map( matcher
                         , self._record.keys()
                         )
                    )
        candidates: List[ Tuple[ HistoryReplay.Record
                               , float
                               ]
                        ] = list( sorted( zip( self._record.keys()
                                             , map(lambda k: self._record[k], self._record.keys())
                                             , match_scores
                                             )
                                        , key=(lambda itm: itm[2])
                                        , reverse=True
                                        )
                                )
        return candidates
        #  }}} method __getitem__ # 

    def update( self
              , step: Key, reward: float, action: Optional[str]
              ):
        #  method update {{{ # 
        """
        Args:
            step (Key): the new state transitted to after `action` is performed
            reward (float): the reward corresponding to the new state
            action (Optional[str]): the performed action, may be null if it is
              the initial state
        """

        self._action_buffer.append(action)
        if action is not None:
            self._action_history.append(action)
        self._observation_buffer.append(step)
        self._reward_buffer.append(reward)
        self._total_reward += reward
        self._total_reward_buffer.append(self._total_reward)
        if len(self._observation_buffer)<self._observation_buffer.maxlen:
            return

        step = self._observation_buffer[0]
        action: str = self._action_buffer[0]
        #step_: HistoryReplay.Key = self._observation_buffer[-1]
        reward: float = self._reward_buffer[1]

        action_history: List[str] = self._action_history[:-self._n_step_flatten]
        last_reward: float = self._reward_buffer[0]
        total_reward: float = self._total_reward_buffer[0]

        if not self._insert_key( step
                               , action_history
                               , last_reward
                               , total_reward
                               ):
            return

        action_dict: HistoryReplay.ActionDict = self._record[step]["action_dict"]
        self._update_action_record(action_dict, action, reward, 0.)
        self._prune_action(action_dict)
        #  }}} method update # 

    def new_trajectory(self):
        #  method new_trajectory {{{ # 
        if len(self._action_buffer)<1\
                or len(self._action_buffer)==1 and self._action_buffer[0] is None:
            self._action_buffer.clear()
            self._action_history.clear()
            self._observation_buffer.clear()
            self._reward_buffer.clear()
            self._total_reward_buffer.clear()

            return

        if self._action_buffer[0] is None:
            self._action_buffer.popleft()
            #self._reward_buffer.popleft()

        rewards = np.asarray(self._reward_buffer[1:], dtype=np.float32)
        convolved_rewards = np.convolve( rewards, self._filter
                                       , mode="full"
                                       )[self._n_step_flatten-1:]

        end_point: Optional[int] = -len(self._action_buffer)

        for k, act, rwd, cvl_rwd\
                , e_p, l_rwd, ttl_rwd in zip( self._observation_buffer[:-1]
                                            , self._action_buffer
                                            , self._reward_buffer
                                            , convolved_rewards
                                            , range(end_point, 0)
                                            , self._reward_buffer[:-1]
                                            , self._total_reward_buffer[:-1]
                                            ):
            action_history: List[str] = self._action_history[:e_p]
            if not self._insert_key( k
                                   , action_history
                                   , l_rwd
                                   , ttl_rwd
                                   ):
                continue

            action_dict: HistoryReplay.ActionDict = self._record[k]["action_dict"]
            self._update_action_record(action_dict, act, float(rwd), float(cvl_rwd))
            self._prune_action(action_dict)

        self._action_buffer.clear()
        self._action_history.clear()
        self._observation_buffer.clear()
        self._reward_buffer.clear()
        self._total_reward_buffer.clear()
        #  }}} method new_trajectory # 

    def _insert_key( self, key: Key
                   , action_history: List[str]
                   , last_reward: float
                   , total_reward: float
                   ) -> bool:
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
                self._record[key] = { "other_info": { "action_history": action_history
                                                    , "last_reward": last_reward
                                                    , "total_reward": total_reward
                                                    , "number": 1
                                                    }
                                    , "action_dict": {}
                                    }
            else:
                new_index: int = len(self._record)
                self._keys.append(key)
                self._similarity_matrix[new_index, :new_index] = similarities
                self._similarity_matrix[:new_index, new_index] = similarities
                self._record[key] = { "other_info": { "action_history": action_history
                                                    , "last_reward": last_reward
                                                    , "total_reward": total_reward
                                                    , "number": 1
                                                    }
                                    , "action_dict": {}
                                    }
            #  }}} Insertion Policy (Static Capacity Limie) # 
        else:
            other_info: HistoryReplay.InfoDict = self._record[key]["other_info"]

            if self._action_history_update_mode=="longest"\
                    and len(action_history) >= other_info["action_history"]:
                other_info["action_history"] = action_history
            elif self._action_history_update_mode=="shortest"\
                    and len(action_history) <= other_info["action_history"]:
                other_info["action_history"] = action_history
            elif self._action_history_update_mode=="newest":
                other_info["action_history"] = action_history
            elif self._action_history_update_mode=="oldest":
                pass

            number: int = other_info["number"]
            number_: int = number + 1
            other_info["number"] = number_

            if self._update_mode=="mean":
                other_info["last_reward"] = float(number)/number_ * other_info["last_reward"]\
                                          + 1./number_ * last_reward
                other_info["total_reward"] = float(number)/number_ * other_info["total_reward"]\
                                           + 1./number_ * total_reward
            elif self._update_mode=="const":
                other_info["last_reward"] += self._learning_rate * (last_reward-other_info["last_reward"])
                other_info["total_reward"] += self._learning_rate * (total_reward-other_info["total_reward"])
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

    def __str__(self) -> str:
        return yaml.dump(self._record, Dumper=yaml.Dumper)
    def load_yaml(self, yaml_file: str):
        #  method load_yaml {{{ # 
        with open(yaml_file) as f:
            self._record = yaml.load(f, Loader=yaml.Loader)
        #  }}} method load_yaml # 
    #  }}} class HistoryReplay # 
