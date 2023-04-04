#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --log-dir logs-historyless/\
			   --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer.easy.subset\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path ../deep_rl_zoo/weights/vilt-b32-mlm-tiny-tkn\
			   --load-replay history-pools/annotated_pool-auto.q.yaml\
			   --save-replay history-pools/annotated_pool-auto.qu.yaml\
			   --item-capacity 500\
			   --action-capacity 10\
			   --matcher lcs+inspat\
			   --prompt-template prompts/\
			   --max-tokens 50\
			   --stop "Discouraged"\
			   --request-timeout 5.
