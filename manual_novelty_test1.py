
import os
import argparse
from envs.polycraft_simplified import SAPolycraftRL
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from obs_convertion import LidarAll, OnlyFacingObs

from args import parser, NOVELTIES

args = parser.parse_args()
verbose = True


# novelty
novelty_name = args.novelty
novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single.json"]
config_file_paths.append(novelty_path)

seed = args.seed

env = SAPolycraftRL(
    config_file_paths=config_file_paths,
    agent_name="agent_0",
    task_name="main",
    show_action_log=True,
    enable_render=True,
)
env.reset(seed=seed)

for episode in range(1000):
    obs, info = env.reset()
    env.render()
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")

    agent = env.env.agent_manager.agents["agent_0"]
    policy = agent.agent.policy

    for step in range(1000):
        print(obs)
        print("actions: ", "; ".join([
            f"{i}: {action}" 
            for i, (action, _) in 
                enumerate(agent.action_set.actions)
            ])
        )
        # action = policy(obs)
        # action = env.action_space.sample()
        action = int(input("action: "))
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward: ", reward)
        
        if verbose:
            env.render()
        if terminated or truncated:
            break

print("Done!")

