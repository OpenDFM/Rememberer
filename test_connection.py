import openai
import yaml
from typing import Dict

with open("openaiconfig.yaml") as f:
    config: Dict[str, str] = yaml.load(f, Loader=yaml.Loader)
openai.api_key = config["api_key"]

completion = openai.Completion.create( model="text-davinci-003"
                                     , prompt="Hello,"
                                     , request_timeout=3.
                                     )
print(completion.choices)
