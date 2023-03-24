
import os
import argparse
from envs.polycraft_simplified import SAPolycraftRL
from shutil import rmtree
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet
import torch

parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
# parser.add_argument("filename", type=str, nargs='+', help="The path of the config file.", default="polycraft_gym_main.json")
parser.add_argument(
    "-n",
    "--episodes",
    type=int,
    help="The number of episodes.",
    required=False,
    default=1000
)
parser.add_argument(
    "--exp_name",
    type=str, 
    help="The name of the experiment.", 
    required=False
)
parser.add_argument(
    '--rendering',
    type=str,
    help="The rendering mode.",
    required=False,
    default="human"
)
parser.add_argument(
    '--seed',
    type=str,
    help="The seed.",
    required=False,
    default=None
)
parser.add_argument(
    '--reset_rl',
    action=argparse.BooleanOptionalAction,
    help="Whether to reset the RL agent and remove the existing models.",
    required=False,
    default=False
)
parser.add_argument(
    '--agent',
    type=str,
    help="The agent module of the first agent.",
    required=False
)

verbose = False

args = parser.parse_args()
num_episodes = args.episodes

exp_name = args.exp_name
agent = args.agent
seed = args.seed
reset_rl = args.reset_rl


if reset_rl:
    try:
        rmtree(os.path.join(os.path.dirname(__file__), "agents", "rl_subagents", "rapid_learn_utils", "policies"))
    except:
        print("No existing RL policies to reset.")

# change agent
# if agent is not None:
#     config_content["entities"]["main_1"]["agent"] = agent


# env = SAPolycraftRL(
#     config_file_paths=config_file_paths,
#     agent_name="agent_0",
#     task_name="main",
#     show_action_log=True
# )

config_file_paths = ["config/polycraft_gym_rl_single.json"]
config_file_paths.append("novelties/evaluation1/multi_interact/multi_interact.json")

env = gym.make(
    "NG2-PolycraftMultiInteract-v0",
    config_file_paths=config_file_paths,
    agent_name="agent_0",
    task_name="main",
    show_action_log=False
)

# tisnhou env
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = BasicNet(state_shape, action_shape)
optim = torch.optim.Adam(net.parameters(), lr=1e-3)
policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=320)
train_collector = ts.data.Collector(policy, env, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
test_collector = ts.data.Collector(policy, env, exploration_noise=True)


train_collector.collect(n_step=5000, random=True)
policy.set_eps(0.1)


for i in range(int(1e6)):
    collect_result = train_collector.collect(n_step=10)

    # once if the collected episodes' mean returns reach the threshold,
    # or every 1000 steps, we test it on test_collector
    if collect_result['rews'].mean() >= env.spec.reward_threshold or i % 1000 == 0:
        policy.set_eps(0.05)
        result = test_collector.collect(n_episode=100)
        if result['rews'].mean() >= env.spec.reward_threshold:
            print(f'Finished training! Test mean returns: {result["rews"].mean()}')
            break
        else:
            # back to training eps
            policy.set_eps(0.1)

    # train policy with a sampled batch data from buffer
    losses = policy.update(64, train_collector.buffer)

print("Done!")

