#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python wikihow.py --log-dir logs/\
			      --train-path ../wikihow/wikihow-canonical.trainset\
			      --task-path ../wikihow/wikihow-microcanon\
			      --avd-name Pixel_2_API_30_ga_x64\
			      --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			      --load-replay history-pools/init_pool.q.yaml\
			      --load-replay history-pools/init_pool.q.yaml\
			      --save-replay history-pools/init_pool.qu."$date_str".a.yaml\
			      --save-replay history-pools/init_pool.qu."$date_str".b.yaml\
			      --item-capacity 500\
			      --action-capacity 20\
			      --matcher lcs+inspat\
			      --prompt-template prompts/\
			      --max-tokens 100\
			      --stop "Discouraged"\
			      --request-timeout 10.\
			      --starts-from 0\
			      --epochs 3
