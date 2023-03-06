#!/bin/bash

export all_proxy="socks5h://127.0.0.1:1080"
python bandits.py --log-dir bandits.logs\
				  --bandits-config bandits.yaml\
				  --prompt-template prompt_bd.txt
