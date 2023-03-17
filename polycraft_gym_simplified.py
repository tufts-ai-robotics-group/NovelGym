
import os
import argparse
from envs.polycraft_simplified import SAPolycraftRL
from shutil import rmtree

parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
parser.add_argument("filename", type=str, nargs='+', help="The path of the config file.", default="polycraft_gym_main.json")
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
config_file_paths = [
    os.path.join(os.path.dirname(__file__), file_name) 
        for file_name in args.filename
]

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

env = SAPolycraftRL(
    config_file_paths=config_file_paths,
    agent_name="agent_0",
    task_name="main"
)

agent = env.agent_manager.agents["agent_0"]
policy = agent.agent.policy

for episode in range(num_episodes):
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")
    obs = env.reset()

    for step in range(1000):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if verbose:
            env.render()
        if terminated:
            break

print("Done!")

