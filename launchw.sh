#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs-webshop\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool2.wqu.2023-04-25T10:42:08.2.yaml\
				  --save-replay history-pools/init_pool2.wqu."$date_str".%d.yaml\
				  --item-capacity 500\
				  --action-capacity 10\
				  --matcher pgpat+insrel\
				  --prompt-template prompts/\
				  --max-tokens 200\
				  --stop "Discouraged"\
				  --request-timeout 10.\
				  --starts-from 0\
				  --epochs 3\
				  --trainset 10\
				  --testseta 50\
				  --testsetb 100
