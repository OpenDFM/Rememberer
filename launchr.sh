#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path ../deep_rl_zoo/weights/vilt-b32-mlm-tiny-tkn\
			   --replay-file replay-data/buy_stocks_W2\(A\).jsonl\
			   				 replay-data/fish_W2\(A\).jsonl\
			   				 replay-data/make_lei_W2\(A\).jsonl\
			   --dump-path visualization/buy_stocks_W2\(A\)\
			   			   visualization/fish_W2\(A\)\
						   visualization/make_lei_W2\(A\)
