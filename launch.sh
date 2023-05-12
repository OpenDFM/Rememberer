#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python main.py --log-dir logs/\
			   --train-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer.complementary.trainset\
			   --task-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer.valset.complementary\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --load-replay history-pools/init_pool.q.yaml\
			   --save-replay history-pools/init_pool.qu."$date_str".%d.yaml\
			   --item-capacity 500\
			   --action-capacity 10\
			   --matcher lcs+inspat\
			   --n-step-flatten 1\
			   --prompt-template prompts/\
			   --max-tokens 100\
			   --stop "Discouraged"\
			   --request-timeout 10.\
			   --train\
			   --starts-from 0\
			   --epochs 3
