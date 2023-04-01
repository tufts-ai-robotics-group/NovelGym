
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


OBS_TYPES = {
    "lidar_all": LidarAll,
    "only_facing": OnlyFacingObs,
}

NOVELTIES = {
    "mi": "novelties/evaluation1/multi_interact/multi_interact.json",
    "kibt": "novelties/evaluation1/key_inventory_break_tree/key_inventory_break_tree.json",
    "rdb": "novelties/evaluation1/random_drop_break/random_drop_break.json",
    "space_ar": "novelties/evaluation1/space_around_crafting_table/space_around_crafting_table.json",
}

RL_ALGOS = {
    "dqn": ts.policy.DQNPolicy,
}


parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
# parser.add_argument("filename", type=str, nargs='+', help="The path of the config file.", default="polycraft_gym_main.json")
parser.add_argument(
    "--novelty",
    type=str, 
    help="The name of the novelty.", 
    required=False,
    default="mi",
    choices=NOVELTIES.keys()
)
# parser.add_argument(
#     '--rendering',
#     type=str,
#     help="The rendering mode.",
#     required=False,
#     default="human"
# )
parser.add_argument(
    '--seed',
    type=str,
    help="The seed.",
    required=False,
    default=None
)
parser.add_argument(
    '--num_threads',
    type=int,
    help="Number of sub threads used to run the env.",
    required=False,
    default=4
)
parser.add_argument(
    '--logdir',
    type=str,
    help="The directory to save the logs.",
    required=False,
    default="results"
)
parser.add_argument(
    '--obs_type',
    type=str,
    help="Type of observation.",
    required=False,
    default="lidar_all",
    choices=OBS_TYPES.keys()
)
parser.add_argument(
    '--rl_algo',
    type=str,
    help="The algorithm for RL.",
    required=False,
    default="dqn",
    choices=RL_ALGOS.keys()
)


verbose = True

args = parser.parse_args()

# novelty
novelty_name = args.novelty
novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single.json"]
config_file_paths.append(novelty_path)



env = SAPolycraftRL(
    config_file_paths=config_file_paths,
    agent_name="agent_0",
    task_name="main",
    show_action_log=True,
    enable_render=True
)


for episode in range(1000):
    obs, info = env.reset()
    env.render()
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")

    agent = env.env.agent_manager.agents["agent_0"]
    policy = agent.agent.policy

    for step in range(1000):
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

