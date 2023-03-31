from typing import Union, Optional, Any
from typing import List, NamedTuple, Dict

import requests
import json

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
