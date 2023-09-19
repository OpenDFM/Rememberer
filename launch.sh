#!/bin/bash
# Copyright 2023 SJTU X-Lance Lab
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Created by Danyang Zhang @X-Lance.

date_str=$(date +%Y-%m-%dT%H:%M:%S)

export TOKENIZERS_PARALLELISM=false
python wikihow.py --log-dir logs/\
			      --train-path ../wikihow/wikihow-canonical.trainset\
			      --task-path ../wikihow/wikihow-microcanon\
			      --avd-name Pixel_2_API_30_ga_x64\
			      --tokenizer-path weights/vilt-b32-mlm-tiny-tkn\
			      --load-replay history-pools/init_pool.q.yaml\
			      --load-replay history-pools/init_pool.q.yaml\
			      --save-replay history-pools/init_pool.qu."$date_str".a.yaml\
			      --save-replay history-pools/init_pool.qu."$date_str".b.yaml\
			      --item-capacity 500\
			      --action-capacity 20\
			      --matcher lcs+inspat\
			      --prompt-template prompts/\
			      --max-tokens 100\
			      --stop "Discouraged"\
			      --request-timeout 10.\
			      --starts-from 0\
			      --epochs 3
