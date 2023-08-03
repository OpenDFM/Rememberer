#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool3.wqu.2023-08-03T11:20:02.2.yaml\
				  --save-replay history-pools/init_pool3.wqu."$date_str".%d.yaml\
				  --item-capacity 500\
				  --action-capacity 20\
				  --matcher pgpat+insrel\
				  --prompt-template prompts/\
				  --max-tokens 200\
				  --request-timeout 20.\
				  --starts-from 1\
				  --epochs 3\
				  --trainseta 0\
				  --trainsetb 10\
				  --testseta 0\
				  --testsetb 100
