#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python main.py --log-dir logs-historyless/\
			   --train-path ../android_env/apps/wikihow/templates.miniout.microbuffer.complementary.trainset\
			   --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer.valset.complementary\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --load-replay history-pools/init_pool3.qu.2023-05-04T19:23:50.2.yaml\
			   --save-replay history-pools/init_pool3.qu."$date_str".%d.yaml\
			   --item-capacity 500\
			   --action-capacity 10\
			   --matcher lcs+inspat\
			   --prompt-template prompts/\
			   --max-tokens 100\
			   --stop "Discouraged"\
			   --request-timeout 10.\
			   --starts-from 0\
			   --epochs 3
