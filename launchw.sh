#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

python webshop.py --log-dir logs\
				  --observation-mode text_rich\
				  --load-replay history-pools/init_pool3.wqu.2023-08-08T00:14:30.2.a.yaml\
				  --load-replay history-pools/init_pool3.wqu.2023-08-08T00:14:30.2.b.yaml\
				  --save-replay history-pools/init_pool3.wqu."$date_str".%d.a.yaml\
				  --save-replay history-pools/init_pool3.wqu."$date_str".%d.b.yaml\
				  --item-capacity 500\
				  --action-capacity 20\
				  --matcher pgpat+insrel\
				  --double-q-learning\
				  --prompt-template prompts/\
				  --max-tokens 200\
				  --request-timeout 20.\
				  --train\
				  --starts-from 3\
				  --epochs 6\
				  --except 501 504 506 507 508\
				  --trainseta 0\
				  --trainsetb 10\
				  --testseta 0\
				  --testsetb 1
