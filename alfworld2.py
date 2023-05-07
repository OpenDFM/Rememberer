import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
print(obs)
print(info)
while True:
    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
    print(admissible_commands)
    #random_actions = [np.random.choice(admissible_commands[0])]
    random_actions = [input()]

    # step
    if random_actions[0]=="next":
        obs, info = env.reset()
        print(obs)
        print(info)
    else:
        obs, scores, dones, info = env.step(random_actions)
        print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))
        #print(obs, scores, dones, info)
        #input()
        if dones[0]:
            obs, info = env.reset()
            print(obs)
            print(info)
