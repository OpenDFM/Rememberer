#!/bin/bash

export TOKENIZERS_PARALLELISM=false
python main.py --task-path ../android_env/apps/wikihow/templates.miniout.microbuffer\
			   --avd-name Pixel_2_API_30_ga_x64_1\
			   --tokenizer-path ../deep_rl_zoo/weights/vilt-b32-mlm-tiny-tkn\
			   --replay-file get_a_loan_without_private_mortgage_insurance_P28pmiP29-8.jsonl\
			   --dump-path visualization/get_loan
