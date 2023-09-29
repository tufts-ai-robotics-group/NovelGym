
import os
import argparse
import tianshou as ts
import gymnasium as gym
from config import OBS_GEN_ARGS, OBS_TYPES
from net.basic import BasicNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from obs_convertion import LidarAll, OnlyFacingObs

from args import parser, NOVELTIES, AVAILABLE_ENVS

from utils.make_env import make_env

args = parser.parse_args()
verbose = True


# novelty
novelty_name = args.novelty
novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single.yaml"]
if novelty_path != "":
    config_file_paths.append(novelty_path)

seed = args.seed

env_name = args.env

# observation generator
RepGenerator = OBS_TYPES[args.obs_type]
rep_gen_args = OBS_GEN_ARGS.get(args.obs_type, {})

env = make_env(
    env_name,
    config_file_paths,
    RepGenerator=RepGenerator,
    rep_gen_args={
        "num_reserved_extra_objects": 2 if novelty_name == "none" else 0,
        "item_encoder_config_path": "config/items.json",
        **rep_gen_args
    },
    render_mode="human",
    base_env_args={"logged_agents": ["agent_0"]}
)

env.reset(seed=seed)

for episode in range(1000):
    cum_rew = 0
    discount_rew = 0
    obs, info = env.reset()
    env.render()
    env.rep_gen.item_encoder.save_json("config/items2.json")
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")

    agent = env.env.agent_manager.agents["agent_0"]
    policy = agent.agent.policy

    for step in range(1000):
        print(obs)
        print(obs.shape)
        print("actions: ", "; ".join([
            f"{i}: {action}" 
            for i, (action, _) in 
                enumerate(agent.action_set.actions)
            ])
        )
        # action = policy(obs)
        # action = env.action_space.sample()
        env.render()
        action = int(input("action: "))
        obs, reward, terminated, truncated, info = env.step(action)
        cum_rew += reward
        discount_rew = reward + 0.99 * discount_rew
        print(f"reward: {reward}; cum_rew: {cum_rew}; discount_rew: {discount_rew}")
        
        
        env.render()
        if terminated or truncated:
            break

print("Done!")

