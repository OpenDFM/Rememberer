import vh_to_html
import re
import openai
import speechopenai
#import json
import itertools
import tiktoken

import lxml.etree
import lxml.html
from android_env.wrappers import VhIoWrapper
from typing import Dict, Pattern, Match, List, NamedTuple, Tuple, Set
from typing import Optional, Callable, TypeVar
import numpy as np
import string
import history

import abc
import logging
import datetime
import time
import traceback
import io

logger = logging.getLogger("agent")
ocounter = 0
ologger = logging.getLogger("openaiE")

Action = Tuple[str, str]

class Agent(abc.ABC):
    #  class Agent {{{ # 
    def __init__( self
                #, prompt_template: str
                ):
        #  method __init__ {{{ # 
        """
        Args:
            #prompt_template (str): template of the prompt
        """

        #self._prompt_template: str = prompt_template

        self._action_pattern: Pattern[str] =\
                re.compile(r"^(?P<atype>\w+)\((?P<arg1>\w+)(?:,\s*(?P<arg2>.+))?\)$")
        self._action_history: List[Action] = []
        #  }}} method __init__ # 

    def reset(self):
        self._action_history.clear()

    def __call__( self
                , task: str
                , screen: lxml.etree.Element
                , instruction: str
                , reward: float
                , total_reward: float
                ) -> Dict[str, np.ndarray]:
        #  method __call__ {{{ # 
        """
        Args:
            task (str): task description
            screen (lxml.etree.Element): screen view hierarchy
            instruction (str): step instruction
            reward (float): the last reward
            total_reward (float): the total history reward

        Returns:
            Dict[str, np.ndarray]: dict like
              {
                "action_type": NOTHING
              } or
              {
                "action_type": CLICK
                "element_id": int
              } or
              {
                "action_type": INPUT
                "element_id": int
                "text": str
              } or
              {
                "action_type": SCROLL
                "direction": Direction
              }
              all the values in `action` are wrapped in np.ndarray.
        """

        html_elements: List[lxml.html.Element] =\
                vh_to_html.convert_tree(screen)[0]

        screen_representation: List[str] = []
        for html in html_elements:
            screen_representation.append( lxml.html.tostring( html
                                                            , pretty_print=True
                                                            , encoding="unicode"
                                                            ).strip()\
                                                             .replace("\n", "&#10;")\
                                                             .replace("\r", "&#13;")
                                        )
        screen_representation: str = "\n".join(screen_representation)

        action_tuple: Action = self._get_action( task
                                          , screen_representation.strip()
                                          , instruction
                                          , reward
                                          , total_reward
                                          )
        action_str: str = action_tuple[0]

        if action_str=="NOTHINGG":
            return { "action_type": np.array(VhIoWrapper.ActionType.NOTHING)
                   , "records": False
                   }

        self._action_history.append(action_tuple)

        if action_str=="GOBACK":
            return {"action_type": np.array(VhIoWrapper.ActionType.GOBACK)}

        action_match: Match[str] = self._action_pattern.match(action_str)
        if action_match is not None:
            action_type: Optional[str] = action_match.group("atype")
            argument1: Optional[str] = action_match.group("arg1")
            argument2: Optional[str] = action_match.group("arg2")
            if action_type=="CLICK":
                if len(html_elements)>0\
                        and argument1 is not None\
                        and argument1.isdecimal():
                    return { "action_type": np.array(VhIoWrapper.ActionType.CLICK)
                           , "element_id": np.clip( np.array(int(argument1))
                                                  , 0
                                                  , len(html_elements)-1
                                                  )
                           }
            if action_type=="INPUT":
                if len(html_elements)>0\
                        and argument1 is not None\
                        and argument1.isdecimal()\
                        and argument2 is not None:
                    return { "action_type": np.array(VhIoWrapper.ActionType.INPUT)
                           , "element_id": np.clip( np.array(int(argument1))
                                                  , 0
                                                  , len(html_elements)-1
                                                  )
                           , "text": np.array(argument2, dtype=np.object_)
                           }
            if action_type=="SCROLL":
                if argument1 is not None\
                        and argument1.upper() in { "LEFT"
                                                 , "UP"
                                                 , "RIGHT"
                                                 , "DOWN"
                                                 }:
                    return { "action_type": np.array(VhIoWrapper.ActionType.SCROLL)
                           , "direction": np.array(VhIoWrapper.ScrollDirection[argument1.upper()])
                           }
        return {"action_type": np.array(VhIoWrapper.ActionType.NOTHING)}
        #  }}} method __call__ # 

    @abc.abstractmethod
    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
                   ) -> Action:
        raise NotImplementedError()
    #  }}} class Agent # 

class ManualAgent(Agent):
    #  class ManualAgent {{{ # 
    def __init__(self):
        super(ManualAgent, self).__init__()

    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
                   , reward: float
                   , total_reward: float
                   ) -> str:
        #  method _get_action {{{ # 
        print("Task:")
        print(task)
        print("Screen:")
        print(screen)
        print("Instruction:")
        print(instruction)
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

R = TypeVar("Result")

class AutoAgent(Agent):
    #  class AutoAgent {{{ # 

    class TemplateGroup(NamedTuple):
        #  class TemplateGroup {{{ # 
        whole_template: string.Template
        input_template: string.Template
        advice_template: string.Template
        #  }}} class TemplateGroup # 

    def __init__( self
                , history_replay: history.HistoryReplay
                , prompt_templates: TemplateGroup
                , api_key: str
                , model: str = "text-davinci-003"
                , max_tokens: int = 20
                , temperature: float = 0.1
                , stop: Optional[str] = None
                , request_timeout: float = 3.
                , manual: bool = False
                , train: bool = True
                , with_speech: bool = False
                ):
        #  method __init__ {{{ # 
        """
        Args:
            history_replay (history.HistoryReplay): history replay
            prompt_templates (TemplateGroup): templates for the prompt

            api_key (str): openai api key
            model (str): the model to use
            max_tokens (int): max number of tokens to generate
            temperature (float): generating temperature
            stop (Optional[str]): stop sequence for the model
            request_timeout (float): waiting time for the client to timeout

            manual (bool): if a human is waiting the prompt to decide instead
              of sending it to the model
            train (bool): indicats whether the history replay should be updated
              or not

            with_speech (bool): whether the speech wrapper should be used
              instead or not
        """

        super(AutoAgent, self).__init__()

        self._prompt_templates: AutoAgent.TemplateGroup = prompt_templates
        self._api_key: str = api_key
        self._model: str = model
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature
        self._stop: Optional[str] = stop
        self._request_timeout: float = request_timeout

        self._input_length_limit: int = 3700

        self._manual: bool = manual
        self._train: bool = train

        self._last_request_time: datetime.datetime = datetime.datetime.now()

        self._history_replay: history_replay.HistoryReplay = history_replay

        if with_speech:
            self._completor: Callable[..., R] = speechopenai.OpenAI(api_key)
            self._extractor: Callable[..., R] = lambda x: x
        else:
            openai.api_key = api_key
            self._completor: Callable[..., R] = openai.Completion.create
            self._extractor: Callable[R, speechopenai.Result] = lambda cplt: cplt.choices[0]

        self._rng: np.random.Generator = np.random.default_rng()
        self._tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model)
        #  }}} method __init__ # 

    def reset(self):
        super(AutoAgent, self).reset()
        self._history_replay.new_trajectory()

    def _random_action( self, screen: str) -> Action:
        #  method _random_action {{{ # 
        elements: List[str] = screen.splitlines()
        action: np.int64 = self._rng.integers(len(elements)+4)

        directions = ["LEFT", "UP", "RIGHT", "DOWN"]
        if action<4:
            return ( "SCROLL({:})".format(directions[action])
                   , ""
                   )
        return ( "CLICK({:d})".format(action-4)
               , elements[action-4].strip()
               )
        #  }}} method _random_action # 

    def _instantiate_input_template( self
                                   , command: str
                                   , html: str
                                   , instruction: str
                                   , action_history: List[Action]
                                   , reward: float
                                   , total_reward: float
                                   ) -> str:
        #  method _instantiate_input_template {{{ # 
        return self._prompt_templates.input_template.safe_substitute(
                                                        command=command
                                                      , html=html
                                                      , instruction=instruction
                                                      , actions=\
                                                            "\n".join(
                                                                map( " ".join
                                                                   , action_history[-min(5, len(action_history)):]
                                                                   )
                                                              )
                                                      , reward="{:.1f}".format(reward)
                                                      , total_reward="{:.1f}".format(total_reward)
                                                      )
        #  }}} method _instantiate_input_template # 

    def _get_action( self
                   , task: str
                   , screen: str
                   , instruction: str
                   , reward: float
                   , total_reward: float
                   ) -> str:
        #  method _get_action {{{ # 
        #  Replay Updating {{{ # 
        if self._train:
            last_action: Optional[Action] = self._action_history[-1]\
                                            if len(self._action_history)>0\
                                          else None
            self._history_replay.update( (screen, task, instruction)
                                       , reward
                                       , last_action
                                       )
        #  }}} Replay Updating # 

        candidates: List[ Tuple[ history.HistoryReplay.Key
                               , history.HistoryReplay.Record
                               , float
                               ]
                        ] = self._history_replay[(screen, task, instruction)]

        #  Construct New Input {{{ # 
        new_input: str = self._instantiate_input_template( command=task
                                                         , html=screen
                                                         , instruction=instruction
                                                         , action_history=self._action_history
                                                         , reward=reward
                                                         , total_reward=total_reward
                                                         ).strip()
        nb_new_input_tokens: int = len(self._tokenizer.encode(new_input))
        example_tokens_limit: int = self._input_length_limit - nb_new_input_tokens
        #  }}} Construct New Input # 

        #  Construct Examplars {{{ # 
        examplars: List[str] = []
        nb_examplars = 2
        i = 0
        for cdd in candidates:
            #  Contruct one Examplar {{{ # 
            key: history.HistoryReplay.Key
            record: history.HistoryReplay.Record
            key, record, _ = cdd
            info_dict: history.HistoryReplay.InfoDict = record["other_info"]

            action_dict: history.HistoryReplay.ActionDict = record["action_dict"]
            actions: List[Tuple[Action, float]] = sorted( map( lambda itm: (itm[0], itm[1]["qvalue"])
                                                          , action_dict.items()
                                                          )
                                                     , key=(lambda itm: itm[1])
                                                     , reverse=True
                                                     )

            if actions[0][1]<=0.:
                encouraged: List[Tuple[str, float]] = [ ( self._random_action(key[0])
                                                        , self._rng.random()/0.5
                                                        )
                                                      ]
            else:
                encouraged: List[Tuple[str, float]] = actions[:1]
            encouraged_actions: Set[str] = set(map(lambda itm: itm[0], encouraged))
            encouraged: str = "\n".join( map( lambda act: "{:} -> {:.1f} {:}".format(act[0][0], act[1], act[0][1])
                                            , encouraged
                                            )
                                       )

            if actions[-1][1]>0.:
                discouraged_action: str = self._random_action(key[0])
                while discouraged_action in encouraged_actions:
                    discouraged_action = self._random_action(key[0])
                discouraged: List[Tuple[str, float]] = [ ( discouraged_action
                                                         , 0.
                                                         )
                                                       ]
            else:
                discouraged: List[Tuple[str, float]] = list( itertools.takewhile( lambda itm: itm[1]==0.
                                                                                , reversed(actions)
                                                                                )
                                                           )
            discouraged: str = "\n".join( map( lambda act: "{:} -> {:.1f} {:}".format(act[0][0], act[1], act[0][1])
                                             , discouraged
                                             )
                                        )

            examplar: str = "Example {:d}:\n\n".format(i+1)\
                          + self._instantiate_input_template( command=key[1]
                                                            , html=key[0]
                                                            , instruction=key[2]
                                                            , action_history=info_dict["action_history"]
                                                            , reward=info_dict["last_reward"]
                                                            , total_reward=info_dict["total_reward"]
                                                            )\
                          + "\n"\
                          + self._prompt_templates.advice_template.safe_substitute(
                                                                     encouraged=encouraged
                                                                   , discouraged=discouraged
                                                                   )
            #  }}} Contruct one Examplar # 

            examplar_length: int = len(self._tokenizer.encode(examplar))
            if examplar_length<=example_tokens_limit:
                examplars.append(examplar)
                example_tokens_limit -= examplar_length
                i += 1
                if i>=nb_examplars:
                    break

        example_str: str = "\n".join(reversed(examplars)).strip()
        #  }}} Construct Examplars # 

        prompt: str = self._prompt_templates.whole_template.safe_substitute( examples=example_str
                                                                           , new_input=new_input
                                                                           )
        try:
            #  Fetch Response {{{ # 
            if not self._manual:
                request_time = datetime.datetime.now()
                timedelta: datetime.timedelta = request_time - self._last_request_time
                timedelta: float = timedelta.total_seconds()
                if 3.1 - timedelta > 0.:
                    time.sleep(3.1-timedelta)

                completion: R = self._completor( model=self._model
                                               , prompt=prompt
                                               , max_tokens=self._max_tokens
                                               , temperature=self._temperature
                                               , stop=self._stop
                                               , request_timeout=self._request_timeout
                                               )
                completion: speechopenai.Result = self._extractor(completion)

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
            #  }}} Fetch Response # 

            #  Parse Action Text {{{ # 
            encouraged_result: str = response.split("Disc", maxsplit=1)[0]
            encouraged_result = encouraged_result.split(":", maxsplit=1)[1]
            encouraged_result = encouraged_result.strip().splitlines()[0]
            encouraging_texts: List[str] = encouraged_result.split("->", maxsplit=1)

            action_text: str = encouraging_texts[0].strip()

            action_tail: List[str] = encouraging_texts[1].strip().split(maxsplit=1)
            element_html: str = action_tail[1].strip() if len(action_tail)>1 else ""
            #  }}} Parse Action Text # 
        except Exception as e:
            #nonlocal ocounter
            with io.StringIO() as bfr:
                ocounter = globals()["ocounter"]
                traceback.print_exc(file=bfr)
                ologger.debug("%d: %s", ocounter, bfr.getvalue())
                logger.debug("Response error %d, %s", ocounter, str(type(e)))
                globals()["ocounter"] += 1
            action_text: str = "NOTHINGG"
            element_html: str = ""

        logger.debug("Action: %s %s", action_text, element_html)

        return (action_text, element_html)
        #  }}} method _get_action # 
    #  }}} class AutoAgent # 

class ReplayAgent(Agent):
    #  class ReproducingAgent {{{ # 
    def __init__(self, replay_files: List[str]):
        #  method __init__ {{{ # 
        super(ReplayAgent, self).__init__()

        self._replay: List[List[Action]] = []
        for rpl_f in replay_files:
            logger.debug("File: %s", rpl_f)
            self._replay.append([])
            with open(rpl_f) as f:
                for l in f:
                    #log_item: Dict[str, str] = json.loads(l)
                    #self._replay[-1].append(log_item["text"].strip())
                    logger.debug("Replay: %s", l.strip())
                    items: List[str] = l.strip().split("<->", maxsplit=1)
                    action: str = items[0].strip()
                    element: str = "" if len(items)==1 else items[1].strip()
                    self._replay[-1].append((action, element))

        self._replay_index: int = -1
        self._index: int = -1

        self._last_request_time: datetime.datetime = datetime.datetime.now()
        #  }}} method __init__ # 

    def reset(self):
        super(ReplayAgent, self).reset()
        self._replay_index += 1
        self._replay_index %= len(self._replay)
        self._index = -1

    def _get_action(self, *args) -> Action:
        #  method _get_action {{{ # 
        request_time = datetime.datetime.now()
        timedelta: datetime.timedelta = request_time - self._last_request_time
        timedelta: float = timedelta.total_seconds()
        if 3.1 - timedelta > 0.:
            time.sleep(3.1-timedelta)
        self._last_request_time = datetime.datetime.now()
        self._index += 1
        self._index %= len(self._replay[self._replay_index])
        logger.debug("Action: %s %s", self._replay[self._replay_index][self._index][0]
                                    , self._replay[self._replay_index][self._index][1]
                    )
        return self._replay[self._replay_index][self._index]
        #  }}} method _get_action # 
    #  }}} class ReproducingAgent # 
