#!/usr/bin/python3

import csv
#import agent_protos
import numpy as np
import itertools

from typing import List, Tuple

def parse_action_with_optional(response: str) -> Tuple[str, float, str]:
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
    return highest_result
    #  }}} function parse_action_with_optional # 

total_returns: List[float] = []
with open("../llmcases/sdebug-20230511@090033,094436.log.END") as f:
    reader = csv.DictReader(f)
    for rcd in reader:
        total_returns.append(float(rcd["Reward"]))

all_q_estimations: List[List[float]] = [[]]
with open("../llmcases/debug-20230511@090033,094436.log.Return") as f:
    for l in f:
        if l.startswith("----"):
            all_q_estimations.append([])
            continue
        returning: str = l[7:l.find(", reason:")]
        returning = eval(returning)
        q_value: float = parse_action_with_optional(returning)[1]
        all_q_estimations[-1].append(q_value)
del all_q_estimations[-1]

errors: List[np.ndarray] = []
for ttl_rtn, q_est in zip(total_returns, all_q_estimations):
    q_est: np.ndarray = np.asarray(q_est)
    error: np.ndarray = q_est-ttl_rtn
    errors.append(error)

errors: np.ndarray = np.concatenate(errors)
print(np.mean(np.abs(errors)))
print(np.sqrt(np.mean(errors**2)))
