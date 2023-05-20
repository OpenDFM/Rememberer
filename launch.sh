#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python wikihow.py --log-dir logs/\
			      --train-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer.complementary.trainset\
			      --task-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer\
			      --avd-name Pixel_2_API_30_ga_x64_1\
			      --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			      --load-replay history-pools/init_pool.q.yaml\
			      --save-replay history-pools/init_pool.qu."$date_str".%d.yaml\
			      --item-capacity 500\
			      --action-capacity 20\
			      --matcher lcs+inspat\
			      --prompt-template prompts/\
			      --max-tokens 100\
			      --stop "Discouraged"\
			      --request-timeout 20.\
			      --starts-from 3\
			      --epochs 10
