from typing import Dict, Tuple, Deque, List
from typing import Union, Optional, Callable, Sequence, TypeVar, Generic, Hashable, Any
import abc
#import dm_env

import numpy as np
import collections
import itertools
import yaml
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score
import torch

import logging

#logger = logging.getLogger("agent.history")
hlogger = logging.getLogger("history")

Key = TypeVar("Key", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)

class Matcher(abc.ABC, Generic[Key]):
    #  class Matcher {{{ # 
    def __init__(self, query: Key):
        #  method __init__ {{{ # 
        self._query: Key = query
        #  }}} method __init__ # 

    def __call__(self, key: Key) -> float:
        raise NotImplementedError
    #  }}} class Matcher # 

MatcherConstructor = Callable[[Key], Matcher[Key]]

class LCSNodeMatcher(Matcher[Tuple[str, Any]]):
    #  class LCSNodeMatcher {{{ # 
    def __init__(self, query: Tuple[str, ...]):
        #  method __init__ {{{ # 
        super(LCSNodeMatcher, self).__init__(query)

        screen: str = self._query[0]
        self._node_sequence: List[str] = list( map( lambda n: n[1:n.index(" ")]
                                                  , screen.splitlines()
                                                  )
                                             )
        #  }}} method __init__ # 

    def __call__(self, key: Tuple[str, ...]) -> float:
        #  method __call__ {{{ # 
        key_screen: str = key[0]
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
            lcs_matrix[i, j] = lcs_matrix[i-1, j-1] + 1 if self._node_sequence[i-1]==key_node_sequence[j-1]\
                                                        else max( lcs_matrix[i-1, j]
                                                                , lcs_matrix[i, j-1]
                                                                )
        lcs: np.int32 = lcs_matrix[n, m]
        length: int = max(n, m)
        similarity: float = float(lcs)/length

        hlogger.debug("Req: %s", " ".join(self._node_sequence))
        hlogger.debug("Key: %s", " ".join(key_node_sequence))
        hlogger.debug( "LCS: %d, L1: %d, L2: %d, Sim: %.2f"
                    , lcs, n, m, similarity
                    )

        return similarity
        #  }}} method __call__ # 
    #  }}} class LCSNodeMatcher # 

class InsPatMatcher(Matcher[Tuple[Any, str]]):
    #  class InsPatMatcher {{{ # 
    _score_matrix: np.ndarray\
            = np.array( [ [1., .1, 0., 0., 0., 0.]
                        , [.1, 1., .3, .3, 0., 0.]
                        , [0., .3, 1., .8, .3, .3]
                        , [0., .3, .8, 1., .3, .3]
                        , [0., 0., .3, .3, 1., .8]
                        , [0., 0., .3, .3, .8, 1.]
                        ]
                      , dtype=np.float32
                      )

    def __init__(self, query: Tuple[Any, str]):
        #  method __init__ {{{ # 
        super(InsPatMatcher, self).__init__(query)

        instruction: str = self._query[-1]

        self._pattern_id: int
        self._pattern_name: str
        self._pattern_id, self._pattern_name = InsPatMatcher._get_pattern(instruction)

        hlogger.debug( "Ins: %s, Pat: %d.%s"
                     , instruction
                     , self._pattern_id
                     , self._pattern_name
                     )
        #  }}} method __init__ # 

    def __call__(self, key: Tuple[Any, str]) -> float:
        #  method __call__ {{{ # 
        if self._pattern_id==-1:
            return 0.

        key_instruction: str = key[-1]
        key_pattern_id: int
        key_pattern_name: str
        key_pattern_id, key_pattern_name = InsPatMatcher._get_pattern(key_instruction)

        hlogger.debug( "Key: %s, Pat: %d.%s"
                     , key_instruction
                     , key_pattern_id
                     , key_pattern_name
                     )

        if key_pattern_id==-1:
            return 0.
        similarity: np.float32 = InsPatMatcher._score_matrix[ self._pattern_id
                                                            , key_pattern_id
                                                            ]

        hlogger.debug("Sim: %.2f", similarity)
        return float(similarity)
        #  }}} method __call__ # 

    @staticmethod
    def _get_pattern(instruction: str) -> Tuple[int, str]:
        #  method _get_pattern {{{ # 
        if instruction=="":
            return 0, "search"
        if instruction.startswith("Access the "):
            if instruction[11:].startswith("article"):
                return 1, "article"
            if instruction[11:].startswith("page of category"):
                return 3, "categ"
            if instruction[11:].startswith("about page"):
                return 5, "about"
        elif instruction.startswith("Check the "):
            if instruction[10:].startswith("author page"):
                return 2, "author"
            if instruction.startswith("reference list"):
                return 4, "reference"
        return -1, "unknown"
        #  }}} method _get_pattern # 
    #  }}} class InsPatMatcher # 

class PagePatMatcher(Matcher[Tuple[Any, str]]):
    #  class PagePatMatcher {{{ # 
    """
    Page Pattern Matcher (pgpat) for WebShop pages.
    """

    _score_matrix: np.ndarray\
            = np.array( [ [1., 0., 0., 0.]
                        , [0., 1., 0., 0.]
                        , [0., 0., 1., .3]
                        , [0., 0., .3, 1.]
                        ]
                      , dtype=np.float32
                      )

    def __init__(self, query: Tuple[Any, str]):
        #  method __init__ {{{ # 
        super(PagePatMatcher, self).__init__(query)

        available_actions: List[str] = self._query[-1].splitlines()
        self._pattern_id: int
        self._pattern_name: str
        self._pattern_id, self._pattern_name = PagePatMatcher._get_pattern(available_actions)

        hlogger.debug( "AAN: %s, Pat: %d.%s"
                     , self._query[-1]
                     , self._pattern_id
                     , self._pattern_name
                     )
        #  }}} method __init__ # 

    def __call__(self, key: Tuple[Any, str]) -> float:
        #  method __call__ {{{ # 
        if self._pattern_id==-1:
            return 0.

        key_actions: List[str] = key[-1].splitlines()
        key_pattern_id: int
        key_pattern_name: str
        key_pattern_id, key_pattern_name = PagePatMatcher._get_pattern(key_actions)

        hlogger.debug( "Key: %s, Pat: %d.%s"
                     , key[-1]
                     , key_pattern_id
                     , key_pattern_name
                     )

        if key_pattern_id==-1:
            return 0.
        similarity: np.float32 = PagePatMatcher._score_matrix[ self._pattern_id
                                                             , key_pattern_id
                                                             ]
        hlogger.debug("Sim: %.2f", similarity)
        return float(similarity)
        #  }}} method __call__ # 

    @staticmethod
    def _get_pattern(available_actions: List[str]) -> Tuple[int, str]:
        #  method _get_pattern {{{ # 
        if len(available_actions)==1 and available_actions[0]=="search":
            return 0, "search"
        if len(available_actions)==2\
                and available_actions[0]=="back to search"\
                and available_actions[1]=="< prev":
            return 3, "other"
        if len(available_actions)>=3\
                and all( act[:2]=="b0"\
                     and act[2].isdecimal()\
                     and act[2].isascii()\
                     for act in available_actions[2:]
                       ):
            return 1, "results"
        if len(available_actions)>=3\
                and "buy now" in available_actions:
            return 2, "goods"
        return -1, "unknown"
        #  }}} method _get_pattern # 
    #  }}} class PagePatMatcher # 

class InsPageRelMatcher(Matcher[Tuple[str, str, Any]]):
    #  class InsPageRelMatcher {{{ # 
    """
    Matcher for WebShop calculating the correlation between the task
    instruction and the page observation.
    """

    def __init__( self
                , query: Tuple[str, str, Any]
                , transformer: SentenceTransformer = None
                ):
        #  method __init__ {{{ # 
        super(InsPageRelMatcher, self).__init__(query)

        assert transformer is not None
        self._transformer: SentenceTransformer = transformer

        page: str = self._query[0]
        instruction: str = self._query[1]
        # (1+N, D); N is the lines of the page; D is the encoding dimension
        query_encodings: torch.Tensor =\
                self._transformer.encode( [instruction] + page.splitlines()
                                        , show_progress_bar=False
                                        , convert_to_tensor=True
                                        , normalize_embeddings=True
                                        )
        relevancies: torch.Tensor = dot_score( query_encodings[:1]
                                             , query_encodings[1:]
                                             ) # (1, N)
        relevancies = relevancies.squeeze(0) # (N,)
        relevancies = relevancies.sort(descending=True).values # (N,)
        self._relevancy: torch.Tensor = relevancies[1] # ()

        hlogger.debug("IPRel-Q: %.2f", self._relevancy.item())
        #  }}} method __init__ # 

    def __call__(self, key: Tuple[str, str, Any]) -> float:
        #  method __call__ {{{ # 
        page: str = self._query[0]
        instruction: str = self._query[1]
        key_encodings: torch.Tensor =\
                self._transformer.encode( [instruction] + page.splitlines()
                                        , show_progress_bar=False
                                        , convert_to_tensor=True
                                        , normalize_embeddings=True
                                        ) # (1+N, D)
        relevancies: torch.Tensor = dot_score( key_encodings[:1]
                                             , key_encodings[1:]
                                             ) # (1, N)
        relevancies = relevancies.squeeze(0).sort(descending=True).values # (N,)
        relevancy: torch.Tensor = relevancies[1] # ()

        similarity: torch.Tensor = 1. - (self._relevancy-relevancy).abs() # ()

        hlogger.debug("IPRel-K: %.2f", relevancy.item())
        hlogger.debug("Sim: %.2f", similarity.item())
        return similarity.item()
        #  }}} method __call__ # 
    #  }}} class InsPageRelMatcher # 

class DenseInsMatcher(Matcher[Tuple[Any, str, Any]]):
    #  class DenseInsMatcher {{{ # 
    def __init__( self
                , query: Tuple[Any, str, Any]
                , transformer: SentenceTransformer = None
                , index: int = 1
                ):
        #  method __init__ {{{ # 
        super(DenseInsMatcher, self).__init__(query)

        assert transformer is not None
        self._transformer: SentenceTransformer = transformer

        self._index = index
        instruction: str = self._query[self._index]
        # (1, D)
        self._query_encoding: torch.Tensor =\
                self._transformer.encode( [instruction]
                                        , show_progress_bar=False
                                        , convert_to_tensor=True
                                        , normalize_embeddings=True
                                        )
        #  }}} method __init__ # 

    def __call__(self, key: Tuple[Any, str, Any]) -> float:
        #  method __call__ {{{ # 
        instruction: str = key[self._index]
        # (1, D)
        query_encoding: torch.Tensor =\
                self._transformer.encode( [instruction]
                                        , show_progress_bar=False
                                        , convert_to_tensor=True
                                        , normalize_embeddings=True
                                        )

        relevancy: torch.Tensor = dot_score( self._query_encoding
                                           , query_encoding
                                           ) # (1, 1)
        similarity: float = relevancy.squeeze().cpu().item()
        hlogger.debug("Sim: %.2f", similarity)
        return similarity
        #  }}} method __call__ # 
    #  }}} class DenseInsMatcher # 

class DenseTrajectoryMatcher(Matcher[Tuple[Any, Any, str, Any]]):
    #  class DenseTrajectoryMatcher {{{ # 
    def __init__( self
                , query: Tuple[Any, Any, str, Any]
                , transformer: SentenceTransformer = None
                ):
        #  method __init__ {{{ # 
        super(DenseTrajectoryMatcher, self).__init__(query)

        assert transformer is not None
        self._transformer: SentenceTransformer = transformer

        trajectory: str = self._query[2]
        # (M, D)
        self._query_encoding: torch.Tensor =\
                self._transformer.encode( trajectory.splitlines()
                                        , show_progress_bar=False
                                        , convert_to_tensor=True
                                        , normalize_embeddings=True
                                        )
        #  }}} method __init__ # 

    def __call__(self, key: Tuple[Any, Any, str, Any]) -> float:
        #  method __call__ {{{ # 
        trajectory: str = key[2]
        # (N, D)
        query_encoding: torch.Tensor =\
                self._transformer.encode( trajectory.splitlines()
                                        , show_progress_bar=False
                                        , convert_to_tensor=True
                                        , normalize_embeddings=True
                                        )

        relevancy: torch.Tensor = dot_score( self._query_encoding
                                           , query_encoding
                                           ) # (M, N)
        similarity: torch.Tensor = ( relevancy.max(dim=1).values.mean()
                                   + relevancy.max(dim=0).values.mean()
                                   ) / 2. # ()
        similarity: float = similarity.cpu().item()
        hlogger.debug("Sim: %.2f", similarity)
        return similarity
        #  }}} method __call__ # 
    #  }}} class DenseTrajectoryMatcher # 

class LambdaMatcher(Matcher[Key]):
    #  class LambdaMatcher {{{ # 
    def __init__(self, matchers: List[Matcher[Key]], weights: Sequence[float]):
        self._matchers: List[Matcher[Key]] = matchers
        self._lambdas: np.ndarray = np.array(list(weights), dtype=np.float32)

    def __call__(self, key: Key) -> float:
        scores: np.ndarray = np.asarray( list( map( lambda mch: mch(key)
                                                  , self._matchers
                                                  )
                                             )
                                       , dtype=np.float32
                                       )
        return float(np.sum(self._lambdas*scores))
    #  }}} class LambdaMatcher # 

class LambdaMatcherConstructor(Generic[Key]):
    #  class LambdaMatcherConstructor {{{ # 
    def __init__( self
                , matchers: List[MatcherConstructor[Key]]
                , weights: Sequence[float]
                ):
        self._matchers: List[MatcherConstructor[Key]] = matchers
        self._weights: Sequence[float] = weights

    def get_lambda_matcher(self, query) -> LambdaMatcher[Key]:
        matchers: List[Matcher[Key]] = list( map( lambda mch: mch(query)
                                           , self._matchers
                                           )
                                      )
        return LambdaMatcher(matchers, self._weights)
    #  }}} class LambdaMatcherConstructor # 

class HistoryReplay(Generic[Key, Action]):
    #  class HistoryReplay {{{ # 
    #Key = Tuple[ str # screen representation
               #, str # task description
               #, str # step instruction
               #]
    #Action = Tuple[str, str]
    InfoDict = Dict[ str
                   , Union[ float
                          , int
                          , List[Action]
                          ]
                   ]
    ActionDict = Dict[ Action
                     , Dict[ str
                           , Union[int, float]
                           ]
                     ]
    Record = Dict[str, Union[InfoDict, ActionDict, int]]

    def __init__( self
                , item_capacity: Optional[int]
                , action_capacity: Optional[int]
                , matcher: MatcherConstructor
                , gamma: float = 1.
                , step_penalty: float = 0.
                , update_mode: str = "mean"
                , learning_rate: float = 0.1
                , n_step_flatten: Optional[int] = 1
                , action_history_update_mode: str = "shortest"
                ):
        #  method __init__ {{{ # 
        """
        Args:
            item_capacity (Optional[int]): the optional item capacity limit of
              the history pool
            action_capacity (Optional[int]): the optional action capacity of
              each item in the history pool
            matcher (MatcherConstructor): matcher constructor

            gamma (float): the discount in calculation of the value function
            step_penalty (float): an optional penalty for the step counts

            update_mode (str): "mean" or "const"
            learning_rate (float): learning rate
            n_step_flatten (Optional[int]): flatten the calculation of the estimated q
              value up to `n_step_flatten` steps

            action_history_update_mode (str): "longest", "shortest", "newest",
              or "oldest"
        """

        self._record: Dict[ Key
                          , HistoryReplay.Record
                          ] = {}

        self._item_capacity: Optional[int] = item_capacity
        self._action_capacity: Optional[int] = action_capacity
        self._matcher: MatcherConstructor = matcher

        self._gamma: float = gamma
        if n_step_flatten is not None:
            self._multi_gamma: float = gamma ** n_step_flatten
            self._filter: np.ndarray = np.logspace( 0, n_step_flatten
                                                  , num=n_step_flatten
                                                  , endpoint=False
                                                  , base=self._gamma
                                                  )[::-1] # (n,)

        self._step_penalty: float = step_penalty

        self._update_mode: str = update_mode
        self._learning_rate: float = learning_rate
        self._n_step_flatten: Optional[int] = n_step_flatten

        self._action_history_update_mode: str = action_history_update_mode

        maxlenp1: Optional[int] = self._n_step_flatten+1 if self._n_step_flatten is not None else None
        self._action_buffer: Deque[Optional[Action]] = collections.deque(maxlen=self._n_step_flatten)
        self._action_history: List[Action] = []
        self._observation_buffer: Deque[Key]\
                = collections.deque(maxlen=maxlenp1)
        self._reward_buffer: Deque[float] = collections.deque(maxlen=maxlenp1)
        self._total_reward: float = 0.
        self._total_reward_buffer: Deque[float] = collections.deque(maxlen=maxlenp1)

        self._similarity_matrix: np.ndarray = np.zeros( (self._item_capacity, self._item_capacity)
                                                      , dtype=np.float32
                                                      )
        #self._index_pool: Deque[int] = collections.deque(range(self._item_capacity))
        #self._index_dict: Dict[HistoryReplay.Key, int] = {}
        self._keys: List[HistoryReplay] = []
        self._max_id: int = 0
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
                                        , key=( lambda itm: ( itm[2]
                                                            , sum( map( lambda d: d["number"]
                                                                      , itm[1]["action_dict"].values()
                                                                      )
                                                                 )
                                                            )
                                              )
                                        , reverse=True
                                        )
                                )
        return candidates
        #  }}} method __getitem__ # 

    def update( self
              , step: Key
              , reward: float
              , action: Optional[Action] = None
              , last_step: bool = False
              ):
        #  method update {{{ # 
        """
        Args:
            step (Key): the new state transitted to after `action` is performed
            reward (float): the reward corresponding to the new state
            action (Optional[Action]): the performed action, may be null if it is
              the initial state
            last_step (bool): whether this is the last step
        """

        self._action_buffer.append(action)
        if action is not None:
            self._action_history.append(action)
        self._observation_buffer.append(step)
        self._reward_buffer.append(reward)
        self._total_reward += reward
        self._total_reward_buffer.append(self._total_reward)

        if self._observation_buffer.maxlen is not None\
                and len(self._observation_buffer)==self._observation_buffer.maxlen:

            step = self._observation_buffer[0]
            action: Action = self._action_buffer[0]
            step_: Key = self._observation_buffer[-1]
            reward: float = self._reward_buffer[1]

            action_history: List[Action] = self._action_history[:-self._n_step_flatten]
            last_reward: float = self._reward_buffer[0]
            total_reward: float = self._total_reward_buffer[0]

            if not self._insert_key( step
                                   , action_history
                                   , last_reward
                                   , total_reward
                                   ):
                return

            new_estimation: np.float64 = np.convolve( np.asarray(self._reward_buffer, dtype=np.float32)[1:]
                                                    , self._filter
                                                    , mode="valid"
                                                    )[0]

            action_dict: HistoryReplay.ActionDict = self._record[step]["action_dict"]
            self._update_action_record(action_dict, action, reward, float(new_estimation), step_)
            self._prune_action(action_dict)

        if last_step:
            self._clear_buffer()
        #  }}} method update # 

    def _clear_buffer(self):
        #  method new_trajectory {{{ # 
        if len(self._action_buffer)<1\
                or len(self._action_buffer)==1 and self._action_buffer[0] is None:
            self._action_buffer.clear()
            self._action_history.clear()
            self._observation_buffer.clear()
            self._reward_buffer.clear()
            self._total_reward_buffer.clear()
            self._total_reward = 0.

            return

        if self._action_buffer[0] is None:
            self._action_buffer.popleft()
            #self._reward_buffer.popleft()

        rewards = np.asarray(self._reward_buffer, dtype=np.float32)[1:]
        if self._n_step_flatten is not None:
            convolved_rewards = np.convolve( rewards, self._filter
                                           , mode="full"
                                           )[self._n_step_flatten-1:]
        else:
            convolved_rewards = np.convolve( rewards
                                           , np.logspace( 0, len(rewards)
                                                        , num=len(rewards)
                                                        , endpoint=False
                                                        , base=self._gamma
                                                        )[::-1]
                                           , mode="full"
                                           )[len(rewards)-1:]

        end_point: Optional[int] = -len(self._action_buffer)

        for k, act, rwd, cvl_rwd\
                , e_p, l_rwd, ttl_rwd in zip( list(self._observation_buffer)[:-1]
                                            , self._action_buffer
                                            , self._reward_buffer
                                            , convolved_rewards
                                            , range(end_point, 0)
                                            , list(self._reward_buffer)[:-1]
                                            , list(self._total_reward_buffer)[:-1]
                                            ):
            action_history: List[Action] = self._action_history[:e_p]
            if not self._insert_key( k
                                   , action_history
                                   , l_rwd
                                   , ttl_rwd
                                   ):
                continue

            action_dict: HistoryReplay.ActionDict = self._record[k]["action_dict"]
            self._update_action_record(action_dict, act, float(rwd), float(cvl_rwd), None)
            self._prune_action(action_dict)

        self._action_buffer.clear()
        self._action_history.clear()
        self._observation_buffer.clear()
        self._reward_buffer.clear()
        self._total_reward_buffer.clear()
        self._total_reward = 0.
        #  }}} method new_trajectory # 

    def _insert_key( self, key: Key
                   , action_history: List[Action]
                   , last_reward: float
                   , total_reward: float
                   ) -> bool:
        #  method _insert_key {{{ # 

        hlogger.debug("Record: %d, Keys: %d", len(self._record), len(self._keys))

        if key not in self._record:
            #  Insertion Policy (Static Capacity Limie) {{{ # 
            matcher: Matcher[Key] = self._matcher(key)
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
                action_dict1: HistoryReplay.ActionDict = self._record[self._keys[max_old_similarity_index[0]]]["action_dict"]
                nb_samples1: int = sum(map(lambda d: d["number"], action_dict1.values()))

                action_dict2: HistoryReplay.ActionDict = self._record[self._keys[max_old_similarity_index[1]]]["action_dict"]
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
                                    , "id": self._max_id
                                    }
                self._max_id += 1
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
                                    , "id": self._max_id
                                    }
                self._max_id += 1
            #  }}} Insertion Policy (Static Capacity Limie) # 
        else:
            other_info: HistoryReplay.InfoDict = self._record[key]["other_info"]

            if self._action_history_update_mode=="longest"\
                    and len(action_history) >= len(other_info["action_history"]):
                other_info["action_history"] = action_history
            elif self._action_history_update_mode=="shortest"\
                    and len(action_history) <= len(other_info["action_history"]):
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
                             , action: Action
                             , reward: float
                             , new_estimation: float
                             , end_step: Optional[Key]
                             )\
            -> Dict[str, Union[int, float]]:
        #  method _update_action_record {{{ # 
        if action not in action_dict:
            action_dict[action] = { "reward": 0.
                                  , "qvalue": 0.
                                  , "number": 0
                                  }
        action_record = action_dict[action]

        number: int = action_record["number"]
        number_: int = number + 1
        action_record["number"] = number_

        #  New Estimation of Q Value {{{ # 
        if end_step is not None:
            if end_step in self._record:
                action_dict: HistoryReplay.ActionDict = self._record[end_step]["action_dict"]
            else:
                record: HistoryReplay.Record = self[end_step][0][1]
                action_dict: HistoryReplay.ActionDict = record["action_dict"]
            qvalue_: float = max(map(lambda act: act["qvalue"], action_dict.values()))
            qvalue_ *= self._multi_gamma
        else:
            qvalue_: float = 0.
        new_estimation = new_estimation + qvalue_
        #  }}} New Estimation of Q Value # 

        if self._update_mode=="mean":
            action_record["reward"] = float(number)/number_ * action_record["reward"]\
                                    + 1./number_ * reward

            action_record["qvalue"] = float(number)/number_ * action_record["qvalue"]\
                                    + 1./number_ * new_estimation
        elif self._update_mode=="const":
            action_record["reward"] += self._learning_rate * (reward-action_record["reward"])
            action_record["qvalue"] += self._learning_rate * (new_estimation-action_record["qvalue"])
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

        keys = list(self._record.keys())
        similarity_matrix = np.zeros( (len(keys), len(keys))
                                    , dtype=np.float32
                                    )
        for i in range(len(keys)):
            similarity_matrix[i, :i] = similarity_matrix[:i, i]

            matcher: Matcher[Key] = self._matcher(keys[i])
            similarity_matrix[i, i+1:] = np.asarray(
                                            list( map( matcher
                                                     , keys[i+1:]
                                                     )
                                                )
                                          )

        if self._item_capacity is not None\
                and self._item_capacity>0\
                and len(keys)>self._item_capacity:
            hlogger.warning( "Boosting the item capacity from %d to %d"
                           , self._item_capacity, len(keys)
                           )
            self._item_capacity = len(keys)
            self._similarity_matrix = similarity_matrix
        else:
            self._similarity_matrix[:len(keys), :len(keys)] = similarity_matrix

        action_size: int = max( map( lambda rcd: len(rcd["action_dict"])
                                   , self._record.values()
                                   )
                              )
        if self._action_capacity is not None\
                and self._action_capacity>0\
                and action_size > self._action_capacity:
            hlogger.warning( "Boosting the item capacity from %d to %d"
                           , self._action_capacity, action_size
                           )
            self._action_capacity = action_size
        self._keys = keys

        self._max_id = max( map( lambda rcd: rcd["id"]
                               , self._record.values()
                               )
                          ) + 1
        #  }}} method load_yaml # 
    def save_yaml(self, yaml_file: str):
        with open(yaml_file, "w") as f:
            yaml.dump(self._record, f, Dumper=yaml.Dumper)
    def __len__(self) -> int:
        return len(self._record)
    #  }}} class HistoryReplay # 
