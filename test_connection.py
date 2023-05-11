import openai
import yaml
from typing import Dict

with open("openaiconfig.yaml") as f:
    config: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
openai.api_key = config["api_key"]

with open("llmcases/debug-20230420@191814.log.api_version.-1") as f:
    prompt = f.read()

completion = openai.Completion.create( model="text-davinci-003"
                                     , prompt=prompt
                                     , request_timeout=20.
                                     )
print(completion.choices)
