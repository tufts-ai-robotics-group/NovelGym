
import os
import argparse
from envs.planning_until_failure import SingleAgentEnv
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from obs_convertion import LidarAll, OnlyFacingObs
import numpy as np
from datetime import datetime

from args import args, NOVELTIES
import cProfile

verbose = False


# novelty
novelty_name = args.novelty
novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single."]
config_file_paths.append(novelty_path)


env = SingleAgentEnv(
    config_file_paths=config_file_paths,
    agent_name="agent_0",
    task_name="main",
    show_action_log=False,
    enable_render=False
)

space = env.action_space.n

def run():
    for episode in range(10000000):
        obs, info = env.reset()
        print()
        print("++++++++++++++ Running episode", episode, "+++++++++++++++")
        begin_time = datetime.now()

        agent = env.env.agent_manager.agents["agent_0"]

        for step in range(1000):
            action = np.random.randint(space - 2)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if verbose:
                env.render()
            if terminated or truncated:
                print("terminated at step", step, "with reward", reward)
                break

        print("Done at", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
        print("Time Used:", str(datetime.now() - begin_time))

# cProfile.run('run()', 'results/profile.txt')
run()
