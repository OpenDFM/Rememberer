# Copyright 2023 SJTU X-Lance Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by Danyang Zhang @X-Lance.

from typing import NamedTuple, List, Tuple, Set
from typing import TypeVar, Optional, Callable, Generic
import abc

import string
import datetime
import time

import openai

import logging
import io
import traceback

import history
import numpy as np
import tiktoken
import itertools

logger = logging.getLogger("agent")
ocounter = 0
ologger = logging.getLogger("openaiE")

class TemplateGroup(NamedTuple):
    whole_template: string.Template
    input_template: string.Template
    advice_template: string.Template
    canonical1: str
    canonical2: str

class Result(NamedTuple):
    text: str
    finish_reason: str

R = TypeVar("Result")
A = TypeVar("Action")

def parse_action_with_optional(response: str) -> Tuple[str, str]:
    #  function parse_action_with_optional {{{ # 
    encouraged_result: str = response.split("Disc", maxsplit=1)[0]
    encouraged_result = encouraged_result.split(":", maxsplit=1)[1]
    encouraged_result: List[str] = encouraged_result.strip().splitlines()

    encouraged_texts: List[Tuple[str, float, str]] = []
    for rst in encouraged_result:
        action_text: str
        action_tail: str
        action_text, action_tail = rst.split("->", maxsplit=1)

        action_text = action_text.strip()

        action_tail: List[str] = action_tail.strip().split(maxsplit=1)
        score: float = float(action_tail[0].strip())
        element_html: str = action_tail[1].strip() if len(action_tail)>1 else ""

        encouraged_texts.append((action_text, score, element_html))

    highest_result: Tuple[str, float, str]\
            = list( itertools.islice( sorted( encouraged_texts
                                            , key=(lambda itm: itm[1])
                                            , reverse=True
                                            )
                                    , 1
                                    )
                  )[0]
    return highest_result[0], highest_result[2]
    #  }}} function parse_action_with_optional # 

class OpenAIClient(abc.ABC, Generic[A]):
    def __init__( self
                , prompt_templates: TemplateGroup
                , api_key: str
                , model: str = "text-davinci-003"
                , max_tokens: int = 20
                , temperature: float = 0.1
                , stop: Optional[str] = None
                , request_timeout: float = 5.
                , request_pause: float = 3.1
                , manual: bool = False
                ):
        #  method __init__ {{{ # 
        """
        Args:
            prompt_templates (TemplateGroup): templates for the prompt

            api_key (str): openai api key
            model (str): the model to use

            max_tokens (int): max number of tokens to generate
            temperature (float): generating temperature
            stop (Optional[str]): stop sequence for the model

            request_timeout (float): waiting time for the client to timeout
            request_pause (float): waiting time between two consecutive request
            manual (bool):
        """

        self._prompt_templates: TemplateGroup = prompt_templates

        self._api_key: str = api_key
        self._model: str = model

        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._stop: Optional[str] = stop

        self._request_timeout: float = request_timeout
        self._request_pause: float = request_pause

        openai.api_key = api_key
        self._completor: Callable[..., R] = openai.Completion.create
        self._extractor: Callable[[R], Result] = lambda cplt: cplt.choices[0]

        self._manual: bool = manual

        self._last_request_time: datetime.datetime = datetime.datetime.now()
        #  }}} method __init__ # 

    def _get_response(self, prompt: str) -> Optional[A]:
        #  method _get_response {{{ # 
        """
        Args:
            prompt (str): the input prompt

        Returns:
            Optional[A]: the completion text
        """

        try:
            if not self._manual:
                request_time = datetime.datetime.now()
                timedelta: datetime.timedelta = request_time - self._last_request_time
                timedelta: float = timedelta.total_seconds()
                if self._request_pause - timedelta > 0.:
                    time.sleep(self._request_pause-timedelta)

                completion: R = self._completor( model=self._model
                                               , prompt=prompt
                                               , max_tokens=self._max_tokens
                                               , temperature=self._temperature
                                               , stop=self._stop
                                               , request_timeout=self._request_timeout
                                               )
                completion: Result = self._extractor(completion)

                self._last_request_time = datetime.datetime.now()

                logger.debug( "Return: {text: %s, reason: %s}"
                            , repr(completion.text)
                            , repr(completion.finish_reason)
                            )

                response: str = completion.text.strip()
            else:
                single_line_response: str = input(prompt)
                response: List[str] = []
                while single_line_response!="":
                    response.append(single_line_response)
                    single_line_response = input()
                response: str = "\n".join(response)

                logger.debug( "Response: %s"
                            , response
                            )

            action: A = self._parse_action(response)
        except Exception as e:
            with io.StringIO() as bfr:
                ocounter = globals()["ocounter"]
                traceback.print_exc(file=bfr)
                ologger.debug("%d: %s", ocounter, bfr.getvalue())
                logger.debug("Response error %d, %s", ocounter, str(type(e)))
                globals()["ocounter"] += 1
            action = None

        return action
        #  }}} method _get_response # 

    @abc.abstractmethod
    def _parse_action(self, response: str) -> A:
        raise NotImplementedError()

class HistoryReplayClient(Generic[history.Key, history.Action]):
    #  class HistoryReplayClient {{{ # 
    def __init__( self
                , history_replay: history.HistoryReplay[history.Key, history.Action]
                , train: bool
                , tokenizer: tiktoken.Encoding
                , norandom: bool = False
                ):
        #  method __init__ {{{ # 
        self._history_replay: history.HistoryReplay[history.Key, history.Action]\
                = history_replay
        self._train: bool = train

        self._rng: np.random.Generator = np.random.default_rng()
        self._tokenizer: tiktoken.Encoding = tokenizer

        self._norandom: bool = norandom
        #  }}} method __init__ # 

    def _get_examplars( self
                      , key: history.Key
                      , example_tokens_limit: int
                      , nb_examplars: int = 2
                      ) -> List[str]:
        #  method _get_examplars {{{ # 
        """
        Args:
            key (history.Key): the key to retrieve
            example_tokens_limit (int): length limit for the examplar strs
            nb_examplars (int): the number of examplars to retrieve

        Returns:
            List[str]: examplar strs
        """

        candidates: List[ Tuple[ history.Key
                               , history.HistoryReplay.Record[history.Action]
                               , float
                               ]
                        ] = self._history_replay[key]

        #  Construct Examplars {{{ # 
        examplars: List[str] = []
        examplar_ids: List[int] = []
        examplar_scores: List[float] = []
        #nb_examplars = 2
        i = 0
        for cdd in candidates:
            #  Contruct one Examplar {{{ # 
            key: history.Key
            record: history.HistoryReplay.Record[history.Action]
            score: float
            key, record, score = cdd
            info_dict: history.HistoryReplay.InfoDict[history.Action] = record["other_info"]

            action_dict: history.HistoryReplay.ActionDict[history.Action] = record["action_dict"]
            actions: List[Tuple[history.Action, float]] =\
                    sorted( map( lambda itm: (itm[0], itm[1]["qvalue"])
                               , action_dict.items()
                               )
                          , key=(lambda itm: itm[1])
                          , reverse=True
                          )

            if actions[0][1]<=0.:
                if self._norandom:
                    encouraged: List[Tuple[history.Action, float]]\
                            = actions[:1]
                else:
                    encouraged: List[Tuple[history.Action, float]]\
                            = [ ( self._random_action(key, True)
                                , self._rng.random()/2.
                                )
                              ]
            else:
                encouraged: List[Tuple[history.Action, float]] = actions[:1]
            encouraged_actions: Set[history.Action] = set(map(lambda itm: itm[0], encouraged))
            encouraged: str = "\n".join( map( lambda act: self._action_to_string(act[0], act[1])
                                            , encouraged
                                            )
                                       )

            if actions[-1][1]>0.:
                if self._norandom:
                    discouraged: List[Tuple[history.Action, float]]\
                            = actions[-1:]
                else:
                    discouraged_action: history.Action = self._random_action(key, False)
                    j = 0
                    while discouraged_action in encouraged_actions:
                        discouraged_action = self._random_action(key, False)
                        j += 1
                        if j>=10:
                            break
                    discouraged: List[Tuple[history.Action, float]]\
                            = [ ( discouraged_action
                                , 0.
                                )
                              ]
                logger.debug("Generated Discouraged: {:}".format(discouraged))
            else:
                discouraged: List[Tuple[history.Action, float]] = list( itertools.takewhile( lambda itm: itm[1]==0.
                                                                      , reversed(actions)
                                                                      )
                                                           )
                logger.debug("Recorded Discouraged: {:}".format(discouraged))
            discouraged: str = "\n".join( map( lambda act: self._action_to_string(act[0], act[1])
                                             , discouraged
                                             )
                                        )

            examplar: str = self._examplar_to_string( i
                                                    , key
                                                    , info_dict
                                                    , encouraged
                                                    , discouraged
                                                    )
            #  }}} Contruct one Examplar # 

            examplar_length: int = len(self._tokenizer.encode(examplar))+1
            if examplar_length<=example_tokens_limit:
                examplars.append(examplar)
                examplar_ids.append(record["id"])
                examplar_scores.append(score)
                example_tokens_limit -= examplar_length
                i += 1
                if i>=nb_examplars:
                    break
        #  }}} Construct Examplars # 

        logger.debug("Egs: %s", " ".join(map(str, examplar_ids)))
        logger.debug("Sms: %s", " ".join(map("{:.2f}".format, examplar_scores)))
        assert len(examplar_ids)>=1

        return examplars
        #  }}} method _get_examplars # 

    @abc.abstractmethod
    def _random_action(self, key: history.Key, encourages: bool = False) -> history.Action:
        raise NotImplementedError()

    @abc.abstractmethod
    def _action_to_string(self, action: history.Action, value: float) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def _examplar_to_string( self
                           , index: int
                           , key: history.Key
                           , info_dict: history.HistoryReplay.InfoDict[history.Action]
                           , encouraged: str
                           , discouraged: str
                           ) -> str:
        raise NotImplementedError()

    def train(self, train: bool):
        self._train = train
    #  }}} class HistoryReplayClient # 
