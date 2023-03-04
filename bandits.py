#!/usr/bin/python3

from typing import Sequence, Optional
from typing import NamedTuple
import numpy as np
import abc

class Step(NamedTuple):
    reward: int # 0 or 1

class BanditsEnv:
    #  class BanditsEnv {{{ # 
    def __init__( self
                , probs: Sequence[float]
                , seed: int = 999
                ):
        #  function __init__ {{{ # 
        """
        Args:
            probs (Sequence[float]): the probabilities of each arm
            seed (int): random seed
        """

        self._nb_arms: int = len(probs)
        self._probabilities: np.ndarray = np.array(probs, dtype=np.float32)
        self._seed: int = seed
        self._rng: np.random.Generator = np.random.default_rng(seed)
        #  }}} function __init__ # 

    def reset(self, seed: Optional[int] = None) -> Step:
        #  function reset {{{ # 
        """
        Args:
            seed (Optional[int]): an optional new random seed

        Returns:
            Step: step information
        """

        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)

        return Step(reward=0)
        #  }}} function reset # 

    def step(self, action: int) -> Step:
        #  function step {{{ # 
        """
        Args:
            action (int): arm index as the action

        Returns:
            Step: step information
        """

        action: np.int64 = np.clip(action, 0, len(self)-1)
        predicate: float = self._rng.random()
        if predicate<self._probabilities[action]:
            reward: int = 1
        else:
            reward: int = 0

        return Step(reward)
        #  }}} function step # 

    def __len__(self) -> int:
        return self._nb_arms
    #  }}} class BanditsEnv # 

class Agent(abc.ABC):
    def __call__(self, last_reward: int):
        raise NotImplementedError()
