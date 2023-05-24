#!/usr/bin/python3

import yaml
import os.path
from typing import Any
from typing import Dict

wikihow_pools = [ "init_pool.qu.2023-05-04T11:11:53.0.yaml"
                , "init_pool.qu.2023-05-04T11:11:53.1.yaml"
                , "init_pool.qu.2023-05-04T11:11:53.2.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.3.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.4.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.5.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.6.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.7.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.8.yaml"
                , "init_pool.qu.2023-05-19T09:43:39.9.yaml"
                ]
webshop_pools = [ "init_pool.wqu.2023-05-11T09:00:03.0.yaml"
                , "init_pool.wqu.2023-05-11T09:00:03.1.yaml"
                , "init_pool.wqu.2023-05-11T09:00:03.2.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.3.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.4.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.5.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.6.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.7.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.8.yaml"
                , "init_pool.wqu.2023-05-19T19:24:25.9.yaml"
                ]

print("WikiHow")
for p in wikihow_pools:
    with open(os.path.join("../history-pools", p)) as f:
        pool: Dict[str, Any] = yaml.load(f, Loader=yaml.Loader)
    observations: int = len(pool)
    experiences: int = sum( map( lambda rcd: len(rcd["action_dict"])
                               , pool.values()
                               )
                          )
    experience_times: int = sum( map( lambda rcd: sum( map( lambda d: d["number"]
                                                          , rcd["action_dict"].values()
                                                          )
                                                     )
                                    , pool.values()
                                    )
                               )
    print(observations, experiences, experience_times)

print("WebShop")
for p in webshop_pools:
    with open(os.path.join("../history-pools", p)) as f:
        pool: Dict[str, Any] = yaml.load(f, Loader=yaml.Loader)
    observations: int = len(pool)
    experiences: int = sum( map( lambda rcd: len(rcd["action_dict"])
                               , pool.values()
                               )
                          )
    experience_times: int = sum( map( lambda rcd: sum( map( lambda d: d["number"]
                                                          , rcd["action_dict"].values()
                                                          )
                                                     )
                                    , pool.values()
                                    )
                               )
    print(observations, experiences, experience_times)
