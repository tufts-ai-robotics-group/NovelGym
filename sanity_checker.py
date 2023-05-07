import numpy as np
from envs.polycraft_simplified import SAPolycraftRL
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from obs_convertion import LidarAll, OnlyFacingObs
from args import parser, NOVELTIES

from policy_utils import create_policy
from utils.pddl_utils import get_all_actions

parser.add_argument(
    "--model_path", '-m',
    type=str,
    help="The path of the saved model to check.",
)

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
# get create policy
all_actions = get_all_actions(config_file_paths)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
policy = create_policy(args.rl_algo, state_shape, action_shape, all_actions)
policy.load_state_dict(torch.load(args.model_path))

for episode in range(1000):
    obs, info = env.reset()
    env.render()
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")

    agent = env.env.agent_manager.agents["agent_0"]

    for step in range(1000):
        action = policy(ts.data.Batch(obs=np.array([obs]), info=info)).act
        action = policy.map_action(action)
        print("action: ", agent.action_set.actions[action][0])
        input("Press Enter to continue...")
        obs, reward, terminated, truncated, info = env.step(action)
        print("reward: ", reward)
        
        if verbose:
            env.render()
        if terminated or truncated:
            break

print("Done!")

