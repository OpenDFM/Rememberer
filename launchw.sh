#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool3.wq.yaml\
				  --save-replay history-pools/init_pool3.wqu."$date_str".%d.a.yaml\
				  --load-replay history-pools/init_pool3.wq.yaml\
				  --save-replay history-pools/init_pool3.wqu."$date_str".%d.b.yaml\
				  --item-capacity 500\
				  --action-capacity 20\
				  --matcher pgpat+insrel\
				  --double-q-learning\
				  --prompt-template prompts/\
				  --max-tokens 200\
				  --request-timeout 20.\
				  --train\
				  --starts-from 0\
				  --epochs 1\
				  --trainseta 0\
				  --trainsetb 10\
				  --testseta 0\
				  --testsetb 1
