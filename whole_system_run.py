import numpy as np
import os
from envs.polycraft_simplified import SAPolycraftRL
import tianshou as ts
from tqdm import tqdm
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
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help="The path of the saved model to check.",
)

args = parser.parse_args()
seed = args.seed or np.random.randint(0, 1000)
SEEDS = [seed + i for i in range(10)]
num_episodes = args.num_episodes

verbose = False

novelty_name = args.novelty
exp_name = args.exp_name or "default_exp"
dir_name = "results" + os.sep + exp_name
result_file = f"{dir_name}{os.sep}{novelty_name}_full_result.csv"


def log_info(seed_no, success_rate):
    with open(result_file, "a") as f:
        f.write(f"{seed_no},{success_rate}\n")

# novelty
novelty_name = args.novelty
novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single.json"]
config_file_paths.append(novelty_path)

seed = args.seed


if __name__ == "__main__":
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not os.path.exists(result_file):
        with open(result_file, "w") as f:
            f.write("seed_no,success_rate\n")
    env = SAPolycraftRL(
        config_file_paths=config_file_paths,
        agent_name="agent_0",
        task_name="main",
        show_action_log=False,
        enable_render=False,
        skip_epi_when_rl_done=False
    )
    # get create policy
    all_actions = get_all_actions(config_file_paths)
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    policy = create_policy(args.rl_algo, state_shape, action_shape, all_actions)
    policy.load_state_dict(torch.load(args.model_path))

    for seed in tqdm(SEEDS):
        success_count = 0
        env.reset(seed=seed)
        for episode in tqdm(range(num_episodes)):
            obs, info = env.reset()

            agent = env.env.agent_manager.agents["agent_0"]

            success = False
            for step in range(1000):
                action = policy(ts.data.Batch(obs=np.array([obs]), info=info)).act
                action = policy.map_action(action)
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated:
                    success = True
                    success_count += 1
                    break
                elif truncated:
                    break
            # print("Episode", episode, ": success" if success else ": fail")

        print()
        print("Seed: ", seed)
        print("Success Rate:", success_count / num_episodes)
        log_info(seed, success_count / num_episodes)

