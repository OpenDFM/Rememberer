#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs-webshop\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool.wq.yaml\
				  --save-replay history-pools/init_pool.wqu."$date_str".%d.yaml\
				  --item-capacity 500\
				  --action-capacity 10\
				  --matcher pgpat+iprel+insrel\
				  --prompt-template prompts/\
				  --max-tokens 50\
				  --stop "Discouraged"\
				  --request-timeout 5.\
				  --pub-to-local-mapping pub-to-mine.list\
				  --train\
				  --starts-from 0\
				  --epochs 6\
				  --trainset 20\
				  --testseta 0\
				  --testsetb 5
