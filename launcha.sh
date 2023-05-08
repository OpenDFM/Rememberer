#!/bin/bash

export ALFWORLD_DATA=$HOME/文档/Mobile-Env/代码/alfworld/data

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python run_alfworld.py --log-dir logs\
				   	   --alfworld-config alfworld_config.yaml\
					   --load-replay history-pools/init_pool.aq.yaml\
					   --save-replay history-pools/init_pool.aqu."$date_str".%d.yaml\
					   --action-capacity 10\
					   --matcher 4inspata3recpiou3obvpat\
					   --n-step-flatten 5\
					   --prompt-template prompts\
					   --max-tokens 200\
					   --stop "Discouraged"\
					   --request-timeout 10.\
					   --train\
				   	   --starts-from 0\
				   	   --epochs 3\
				   	   --trainseta 0\
				   	   --trainsetb 17\
					   --except 3 4 10 11 12 13 14\
				   	   --testseta 0\
				   	   --testsetb 5
