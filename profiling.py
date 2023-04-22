
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

from args import args, NOVELTIES
import cProfile

verbose = False


# novelty
novelty_name = args.novelty
novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single.json"]
config_file_paths.append(novelty_path)


env = SAPolycraftRL(
    config_file_paths=config_file_paths,
    agent_name="agent_0",
    task_name="main",
    show_action_log=False,
    enable_render=False
)

def run():
    for episode in range(5):
        obs, info = env.reset()
        print()
        print("++++++++++++++ Running episode", episode, "+++++++++++++++")

        agent = env.env.agent_manager.agents["agent_0"]

        for step in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if verbose:
                env.render()
            if terminated or truncated:
                print("terminated at step", step)
                break

    print("Done!")

cProfile.run('run()', 'results/profile.txt')

