#!/bin/bash

export ALFWORLD_DATA=$HOME/文档/Mobile-Env/代码/alfworld/data

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python run_alfworld.py --log-dir logs-alfworld\
				   	   --alfworld-config alfworld_config.yaml\
					   --load-replay history-pools/init_pool.aq.yaml\
					   --save-replay history-pools/init_pool.aqu."$date_str".%d.yaml\
					   --action-capacity 10\
					   --matcher 4insrel3istrel3trjrel\
					   --prompt-template prompts\
					   --max-tokens 200\
					   --stop "Discouraged"\
					   --request-timeout 10.\
				   	   --starts-from 0\
				   	   --epochs 3\
				   	   --trainseta 0\
				   	   --trainsetb 10\
				   	   --testseta 0\
				   	   --testsetb 5
