from typing import Union, Optional, Any
from typing import List, NamedTuple, Dict

import requests
import json

import logging

import datetime
import os
import sys
import yaml

logger = logging.getLogger("openai")

_BASE_URL: Dict[str, str] =\
        { "completion": "http://54.193.55.85:10030/v1/completions"
        }

def request( url: str
           , token: str
           , load: Dict[str, Any]
           , timeout: float=5.
           ) -> Dict[str, Any]:
    #  function request {{{ # 
    """
    Args:
        url (str): the requested URL
        token (str): user token
        load (Dict[str, Any]): dict like
          {
            "model": str
            "prompt": str
            "max_tokens": int
            "temperature": float
            "stop": optional str or list with length <=4 of str
          }, more fields may be provided; different fields may be required
          rather than Completion
        timeout (float): request timeout

    Returns:
        Dict[str, Any]: dict like
          {
            "code": str, e.g., "0"
            "message": str, e.g., "Success"
            "result": {
                "text": str
                "is_truncated": bool
                "create_time": float
                "model": str
            }
            "from_cache": bool
            "duration": str, e.g., "0.00s"
          }
    """

    response: requests.Response = requests.post( url
                                               , json=load
                                               , headers={"llm-token": token}
                                               , timeout=timeout
                                               )
    return response.json()
    #  }}} function request # 

class Result(NamedTuple):
    text: str
    finish_reason: str

class OpenAI:
    def __init__(self, token_key: str):
        self._token_key: str = token_key

    def Completion( self
                  , model: str
                  , prompt: str
                  , max_tokens: int
                  , temperature: float
                  , suffix: Optional[str] = None
                  , top_p: int = 1
                  , stream: bool = False
                  , logprobs: Optional[int] = None
                  , stop: Optional[Union[str, List[str]]] = None
                  , presence_penalty: float = 0.
                  , frequency_penalty: float = 0.
                  , logit_bias: Optional[Dict[str, float]] = None
                  , request_timeout: float = 5.
                  , **params
                  ) -> Result:
        #  method Completion {{{ # 
        params.update( { "model": model
                       , "prompt": prompt
                       , "max_tokens": max_tokens
                       , "temperature": temperature
                       , "suffix": suffix
                       , "top_p": top_p
                       , "stream": stream
                       , "logprobs": logprobs
                       , "stop": stop
                       , "presence_penalty": presence_penalty
                       , "frequency_penalty": frequency_penalty
                       , "logit_bias": logit_bias
                       }
                     )
        response: Dict[str, Any] = request( _BASE_URL["completion"]
                                          , self._token_key
                                          , load=params
                                          , timeout=request_timeout
                                          )
        if response["message"]!="Success":
            raise requests.RequestException( "Server Failded with:\n"\
                                           + "model: {:}\n".format(model)\
                                           + "prompt:\n\t{:}\n".format(prompt)\
                                           + "response: {:}\n".format(json.dumps(response))
                                           )
        return Result( response["result"]["text"]
                     , "length" if response["result"]["is_truncated"] else "stop"
                     )
        #  }}} method Completion # 

# {
#   "role": "system" | "user" | "assistant" | "function"
#   "content": str
#   "name": str, required for "function" role
#   "function_call": {
#       "name": str
#       "arguments": dict like {str: anything}
#   }
# }
Message = Dict[ str
              , Union[ str
                     , Dict[ str
                           , Union[ str
                                  , Dict[str, Any]
                                  ]
                           ]
                     ]
              ]

class GPTProxy:
    #  class GPTProxy {{{ # 
    def __init__(self, fetch_api: str):
        self._fetch_api: str = fetch_api

    def _chat( self, model: str
             , messages: List[Message]
             , max_tokens: int = 100
             , temperature: float = 0.1
             , top_p: int = 1
             , stream: bool = False
             , presence_penalty: float = 0.
             , frequency_penalty: float = 0.
             , request_timeout: float = 60.
             , **params
             ) -> Message:
        #  method _call {{{ # 
        header = { "Content-Type":"application/json"
                 , "Authorization": "Bearer {:}".format(self._fetch_api)
                 }
        post_dict = { "model": model
                    , "messages": messages
                    , "max_tokens": max_tokens
                    , "temperature": temperature
                    , "top_p": top_p
                    , "presence_penalty": presence_penalty
                    , "frequency_penalty": frequency_penalty
                    , "request_timeout": request_timeout
                    }
        logger.debug("Request: %s", json.dumps(post_dict))

        response: requests.Response = requests.post( "https://frostsnowjh.com/v1/chat/completions"
                                                   , json=post_dict, headers=header
                                                   )
        logger.debug("Response: %s", repr(response))
        response: Dict[str, Any] = response.json()

        return response["choices"][0]["message"]
        #  }}} method _call # 

    def _completion( self, model: str
                   , messages: str
                   , max_tokens: int = 100
                   , temperature: float = 0.1
                   , top_p: int = 1
                   , stream: bool = False
                   , presence_penalty: float = 0.
                   , frequency_penalty: float = 0.
                   , request_timeout: float = 60.
                   , **params
                   ) -> str:
        #  method _call {{{ # 
        header = { "Content-Type":"application/json"
                 , "Authorization": "Bearer {:}".format(self._fetch_api)
                 }
        post_dict = { "model": model
                    , "messages": messages
                    , "max_tokens": max_tokens
                    , "temperature": temperature
                    , "top_p": top_p
                    , "presence_penalty": presence_penalty
                    , "frequency_penalty": frequency_penalty
                    , "request_timeout": request_timeout
                    }
        logger.debug("Request: %s", json.dumps(post_dict))

        response: requests.Response = requests.post( "https://frostsnowjh.com/v1/completions"
                                                   , json=post_dict, headers=header
                                                   )
        logger.debug("Response: %s", repr(response))
        response: Dict[str, Any] = response.json()

        return response["choices"][0]["text"]
        #  }}} method _call # 

    def chatgpt( self, version: str
               , messages: List[Message]
               , max_tokens: int = 100
               , temperature: float = 0.1
               , top_p: int = 1
               , stream: bool = False
               , presence_penalty: float = 0.
               , frequency_penalty: float = 0.
               , request_timeout: float = 60.
               , **params
               ) -> Message:
        #  method chatgpt {{{ # 
        return self._chat( "gpt-3.5-turbo{:}".format(version)
                         , messages
                         , max_tokens
                         , temperature
                         , top_p
                         , stream
                         , presence_penalty
                         , frequency_penalty
                         , request_timeout
                         , **params
                         )
        #  }}} method chatgpt # 

    def gpt4( self, version: str
            , messages: List[Message]
            , max_tokens: int = 100
            , temperature: float = 0.1
            , top_p: int = 1
            , stream: bool = False
            , presence_penalty: float = 0.
            , frequency_penalty: float = 0.
            , request_timeout: float = 60.
            , **params
            ) -> Message:
        #  method chatgpt {{{ # 
        return self._chat( "gpt-4{:}".format(version)
                         , messages
                         , max_tokens
                         , temperature
                         , top_p
                         , stream
                         , presence_penalty
                         , frequency_penalty
                         , request_timeout
                         , **params
                         )
        #  }}} method chatgpt # 
    #  }}} class GPTProxy # 

if __name__ == "__main__":
    #  Logger Config {{{ # 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    
    file_handler = logging.FileHandler(os.path.join("logs-tmp", "normal-{:}.log".format(datetime_str)))
    debug_handler = logging.FileHandler(os.path.join("logs-tmp", "debug-{:}.log".format(datetime_str)))
    stdout_handler = logging.StreamHandler(sys.stdout)
    sdebug_handler = logging.FileHandler(os.path.join("logs-tmp", "sdebug-{:}.log".format(datetime_str)))
    
    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(logging.INFO)
    sdebug_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    sdebug_handler.setFormatter(formatter)
    
    stdout_handler.addFilter(logging.Filter("openai"))
    sdebug_handler.addFilter(logging.Filter("openai"))
    
    logger.addHandler(file_handler)
    logger.addHandler(debug_handler)
    logger.addHandler(stdout_handler)
    logger.addHandler(sdebug_handler)
    #  }}} Logger Config # 

    with open("openaiconfig.yaml") as f:
        config: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
        fetch_api: str = config["fetch_api"]
    proxy = GPTProxy(fetch_api)

    message = "Hello,"
    response: str = proxy._completion( "text-davinci-003"
                                     , message
                                     , max_tokens=50
                                     , request_timeout=300.
                                     )
    print(response)
