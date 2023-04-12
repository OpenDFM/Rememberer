#!/bin/bash

#date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python main.py --log-dir logs-historyless/\
			   --train-path ../android_env/apps/wikihow/templates.miniout.microbuffer.annotated\
			   --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer.valset\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --load-replay history-pools/init_pool.qu.2023-04-12T15:43:10.0.yaml\
			   --save-replay history-pools/init_pool.qu.2023-04-12T15:43:10.%d.yaml\
			   --item-capacity 500\
			   --action-capacity 10\
			   --matcher lcs+inspat\
			   --prompt-template prompts/\
			   --max-tokens 50\
			   --stop "Discouraged"\
			   --request-timeout 5.\
			   --train\
			   --starts-from 1\
			   --epochs 3\
			   --except 0 1 2 3 5 6 7 8 9
