#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool.wq.yaml\
				  --save-replay history-pools/init_pool.wqu."$date_str".%d.yaml\
				  --item-capacity 500\
				  --action-capacity 20\
				  --matcher pgpat+insrel\
				  --prompt-template prompts/\
				  --max-tokens 200\
				  --stop "Discouraged"\
				  --request-timeout 10.\
				  --starts-from 0\
				  --epochs 3\
				  --trainseta 0\
				  --trainsetb 10\
				  --testseta 0\
				  --testsetb 100
