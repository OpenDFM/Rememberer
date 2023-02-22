#!/bin/bash

python main.py --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path ../deep_rl_zoo/weights/vilt-b32-mlm-tiny-tkn\
			   --prompt-template prompt_pt.txt
