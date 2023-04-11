#!/bin/bash

date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python main.py --log-dir logs-historyless/\
			   --train-path ../android_env/apps/wikihow/templates.miniout.microbuffer.annotated.subset\
			   --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer.valset.subset\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			   --load-replay history-pools/init_pool.q.yaml\
			   --save-replay history-pools/init_pool.qu.$date_str.%d.yaml\
			   --item-capacity 500\
			   --action-capacity 10\
			   --matcher lcs+inspat\
			   --prompt-template prompts/\
			   --max-tokens 50\
			   --stop "Discouraged"\
			   --request-timeout 5.\
			   --train\
			   --epochs 3
