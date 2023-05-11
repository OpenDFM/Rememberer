#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool3.wqu.2023-05-11T14:22:06.2.yaml\
				  --save-replay history-pools/init_pool3.wqu."$date_str".%d.yaml\
				  --item-capacity 500\
				  --action-capacity 10\
				  --matcher pgpat+insrel\
				  --n-step-flatten 1\
				  --prompt-template prompts/\
				  --max-tokens 200\
				  --stop "Discouraged"\
				  --request-timeout 20.\
				  --starts-from 0\
				  --epochs 3\
				  --trainseta 0\
				  --trainsetb 10\
				  --testseta 0\
				  --testsetb 100
