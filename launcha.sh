#!/bin/bash

#date_str=$(date +%Y-%m-%dT%H:%M:%S)

python run_alfworld.py --log-dir logs-alfworld\
				   	   --alfworld-config alfworld_config.yaml\
				   	   --starts-from 0\
				   	   --epochs 3\
				   	   --trainseta 0\
				   	   --trainsetb 10\
				   	   --testseta 0\
				   	   --testsetb 5
