import os
import argparse
from gym_novel_gridworlds2.envs.sequential import NovelGridWorldSequentialEnv
from gym_novel_gridworlds2.utils.game_report import report_game_result
from gym_novel_gridworlds2.utils.json_parser import ConfigParser, load_json
from gym_novel_gridworlds2.utils.game_report import get_game_time_str
from utils.pddl_utils import generate_obj_types
from shutil import rmtree

parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
parser.add_argument("filename", type=str, nargs='+', help="The path of the config file.", default="polycraft_gym_main.json")
parser.add_argument(
    "-n",
    "--episodes",
    type=int,
    help="The number of episodes.",
    required=False,
    default=100
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

json_parser = ConfigParser()
config_content = load_json(config_json={"extends": config_file_paths})

if reset_rl:
    try:
        rmtree(os.path.join(os.path.dirname(__file__), "agents", "rl_subagents", "rapid_learn_utils", "policies"))
    except:
        print("No existing RL policies to reset.")

# change agent
if agent is not None:
    config_content["entities"]["main_1"]["agent"] = agent

env = NovelGridWorldSequentialEnv(
    config_dict=config_content, 
    max_time_step=1000, 
    time_limit=900, 
    run_name=exp_name,
    # logged_agents=['main_1'],
)

for episode in range(num_episodes):
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")
    print()
    env.reset(return_info=True, options={"episode": episode})
    # add an object map to the dynamics so that the observation json of rapid_learn
    # can be generated.
    env.dynamic.all_objects = generate_obj_types(config_content)
    env.render()

    for agent in env.agent_iter():
        action = None
        while (
            action is None
            or env.agent_manager.get_agent(agent).action_set.actions[action][1].allow_additional_action
        ):
            ## while action is valid, do action.
            if agent not in env.dones or env.dones[agent]:
                # skips the process if agent is done.
                env.step(0, {})
                break

            observation, reward, done, info = env.last()
            result = env.agent_manager.agents[agent].agent.policy(observation)

            # getting the actions
            extra_params = {}
            if type(result) == tuple:
                # symbolic agent sending extra params
                action, extra_params = result
            else:
                # rl agent / actions with no extra params
                action = result

            env.step(action, extra_params)

            env.render()
