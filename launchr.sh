#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer.hard\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path ../deep_rl_zoo/weights/vilt-b32-mlm-tiny-tkn\
			   --replay-file replay-data/buy_stocks_W4\(eA\).jsonl\
			   				 replay-data/bypass_youtube_W4\(eA\).jsonl\
			   				 replay-data/deal_sister_W4\(eA\).jsonl\
                             replay-data/do_ruby_W4\(eA\).jsonl\
                             replay-data/get_girls_W4\(eA\).jsonl\
                             replay-data/make_lei_W4\(eA\).jsonl\
			   --dump-path visualization/buy_stocks_W4\(eA\)\
			   			   visualization/bypass_youtube_W4\(eA\)\
						   visualization/deal_sister_W4\(eA\)\
                           visualization/do_ruby_W4\(eA\)\
                           visualization/get_girls_W4\(eA\)\
                           visualization/make_lei_W4\(eA\)
