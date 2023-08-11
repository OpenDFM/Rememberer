#!/bin/bash

#date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python wikihow.py --log-dir logs/\
			      --train-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer.complementary.trainset\
			      --task-path ../../../android_env/apps/wikihow/templates.miniout.microbuffer.valset.subset\
			      --avd-name Pixel_2_API_30_ga_x64_1\
			      --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			      --load-replay history-pools/init_pool.qu.2023-08-11T15:13:44.0.a.yaml\
			      --load-replay history-pools/init_pool.qu.2023-08-11T15:13:44.0.b.yaml\
			      --save-replay history-pools/init_pool.qu.2023-08-11T15:13:44.%d.a.yaml\
			      --save-replay history-pools/init_pool.qu.2023-08-11T15:13:44.%d.b.yaml\
			      --item-capacity 500\
			      --action-capacity 20\
			      --matcher lcs+inspat\
				  --double-q-learning\
			      --prompt-template prompts/\
			      --max-tokens 100\
			      --request-timeout 20.\
				  --train\
			      --starts-from 1\
			      --epochs 6\
				  --except 0 1 2 3 4 5
