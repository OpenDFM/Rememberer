#!/usr/bin/python3

from typing import Sequence, Optional
from typing import NamedTuple
import numpy as np
import abc

import string
import datetime
import time

import logging

run_logger = logging.getLogger("agent")

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

        if action<0:
            return Step(0)

        action: np.int64 = np.clip(action, 0, len(self)-1)
        predicate: float = self._rng.random()
        if predicate<self._probabilities[action]:
            reward = 1
        else:
            reward = 0

        return Step(reward)
        #  }}} function step # 

    def __len__(self) -> int:
        return self._nb_arms
    #  }}} class BanditsEnv # 

class Agent(abc.ABC):
    def __call__(self, last_reward: int) -> int:
        raise NotImplementedError()
    def reset(self):
        pass

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self, nb_arms: int) -> int:
        self._nb_arms: int = nb_arms
    def __call__(self, last_reward: int) -> int:
        action: str = input("Please input an arm index (There are totally {:d} arms):".format(self._nb_arms))
        return int(action)
    #  }}} class ManualAgent # 

# TODO: HistoryPool

class AutoAgent(Agent):
    #  class AutoAgent {{{ # 
    def __init__( self
                , nb_arms: int
                , prompt_template: string.Template
                , api_key: str
                , model: str = "text-davinci-003"
                , max_tokens: int = 5
                , temperature: float = 0.1
                , request_timeout: float = 3.
                ):
        #  method __init__ {{{ # 
        self._nb_arms: int = nb_arms

        self._prompt_template: string.Template = prompt_template
        self._api_key: str = api_key
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._request_timeout: float = request_timeout

        self._last_request_time: datetime.datetime = datetime.datetime.now()

        openai.api_key = api_key
        #  }}} method __init__ # 

    def __call__(self, last_reward: int) -> int:
        #  method __call__ {{{ # 
        # TODO
        prompt: str = self._prompt_template.safe_substitute(
                                                nb_arms=self._nb_arms
                                              )

        try:
            request_time = datetime.datetime.now()
            timedelta: datetime.timedelta = request_time - self._last_request_time
            timedelta: float = timedelta.total_seconds()
            if 3.1 - timedelta > 0.:
                time.sleep(3.1-timedelta)
            completion = openai.Completion.create( model=self._model
                                                 , prompt=prompt
                                                 , max_tokens=self._max_tokens
                                                 , temperature=self._temperature
                                                 , request_timeout=self._request_timeout
                                                 )
            self._last_request_time = datetime.datetime.now()
        except:
            return -1

        run_logger.debug( "Returns: {\"text\": %s, \"reason\": %s}"
                    , repr(completion.choices[0].text)
                    , repr(completion.choices[0].finish_reason)
                    )

        try:
            action = int(completion.choices[0].text)
        except:
            action = -1
        return action
        #  }}} method __call__ # 
    #  }}} class AutoAgent # 

if __name__ == "__main__":
    #  Command Line Options {{{ # 
    parser = argparse.ArgumentParser()

    parser.add_argument("--log-dir", default="logs", type=str)
    parser.add_argument("--config", default="openaiconfig.yaml", type=str)

    parser.add_argument("--bandits-config", default="bandits.yaml", type=str)

    parser.add_argument("--prompt-template", type=str)
    parser.add_argument("--max-tokens", default=20, type=int)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--request-timeout", default=3., type=float)

    #parser.add_argument("--replay-file", type=str)
    #parser.add_argument("--dump-path", type=str)

    args: argparse.Namespace = parser.parse_args()
    #  }}} Command Line Options # 

    #  Logger Config {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    timestampt: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    file_handler = logging.FileHandler( os.path.join( "bandits.logs"
                                                    , "normal-{:}.log".format(timestampt)
                                                    )
                                      )
    debug_handler = logging.FileHandler( os.path.join( "bandits.logs"
                                                     , "debug-{:}.log".format(timestampt)
                                                     )
                                       )
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    stdout_handler.addFilter(logging.Filter("agent"))

    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    #  }}} Logger Config # 

    #  Build Environment and Agent {{{ # 
    # TODO: load bandits config
    env = BanditsEnv( probs=probabilities
                    , seed=999
                    )

    with open(args.prompt_template) as f:
        prompt_template = string.Template(f.read())
    with open(args.config) as f:
        openaiconfig: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
    model = agent.AutoAgent( prompt_template=prompt_template
                           , api_key=openaiconfig["api_key"]
                           , max_tokens=args.max_tokens
                           , temperature=args.temperature
                           , request_timeout=args.request_timeout
                           )

    #  Workflow {{{ # 
    max_nb_steps = 15
    nb_turns = 15
    for i in range(nb_turns):
        model.reset()
        step: Step = env.reset()

        reward: int = step.reward
        for j in range(max_nb_steps):
            action: int = model(step.reward)
            step = env.step(action)
            reward += step.reward

        run_logger.info( "\x1b[42mEND!\x1b[0m TrajecId: %d, Reward: %d"
                       , i, reward
                       )
    #  }}} Workflow # 
    #  }}} Build Environment and Agent # 
