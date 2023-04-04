#!/usr/bin/python3

import sys
sys.path.append("../WebShop")

import gym
from web_agent_site.envs import WebAgentTextEnv

# # Interfaces of WebAgentTextEnv:
# 
# def init( observation_mode: str = "html" # "html": raw html
#                                          # "text": " [SEP] " joined element text contents
#                                          # "text_rich": "\n" joined discription with bounded text contents
#                                          #              bounding labels are
#                                          #              [button][button_] and [clicked
#                                          #              button][clicked button_];
#                                          #              non-product-link as [clicked
#                                          #              button] will be prefixed with a
#                                          #              discription as "You have clicked {t}.\n"
#                                          # "url": url
#     , file_path: str = utils.DEFAULT_FILE_PATH # path to a json as the data file
#     , num_prev_actions: int = 0 # the number of history actions to append
#                                 # after the observation; actions are appended
#                                 # in a reverse order
#     , num_prev_obs: int = 0 # the number of history observations to append
#                             # after the current observation; observations are
#                             # appended in a reverse order; observations are
#                             # suffixed interleavingly with the actions like:
#                             # 
#                             # <current_obs> [SEP] act_{n-1} [SEP] obs_{n-1} [SEP] ... [SEP] obs_0
#     )
# 
# def step( action: str # search[keywords]
#                       # click[element], element should in
#                       # self.text_to_clickable and shouldn't be search
#         ) -> Tuple[ str # observation
#                   , float # reward
#                   , bool # done or not
#                   , None
#                   ]
# 
# def get_available_actions()\
#         -> Dict[str, bool | List[str]]
#         # {
#         #   "has_search_bar": bool
#         #   "clickables": List[str]
#         # }
# def get_instruction_text() -> str
# def observation -> str
# def state -> Dict[str, str]
#           # {
#           #     "url": str
#           #     "html": str
#           #     "instruction_text": str
#           # }
# text_to_clickable: Dict[str, Any] # {element_text: bs4 element}
# instruction_text: str
# 
# def reset() -> Tuple[ str # observation
#                     , None
#                     ]

#def main():
env = gym.make( "WebAgentTextEnv-v0"
              , observation_mode="text"
              , num_products=None
              )

#if __name__ == "__main__":
    #main()
