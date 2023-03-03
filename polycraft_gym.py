import os
import argparse
from gym_novel_gridworlds2.envs.sequential import NovelGridWorldSequentialEnv
from gym_novel_gridworlds2.utils.game_report import report_game_result
from gym_novel_gridworlds2.utils.json_parser import ConfigParser, load_json
from gym_novel_gridworlds2.utils.game_report import get_game_time_str

parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
parser.add_argument("filename", type=str, nargs=1, help="The path of the config file.", default="polycraft_gym_main.json")
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

args = parser.parse_args()
file_name = args.filename[0]
num_episodes = args.episodes
exp_name = args.exp_name
seed = args.seed


json_parser = ConfigParser()
config_file_path = os.path.join(os.path.dirname(__file__), file_name)
config_content = load_json(config_file_path)

env = NovelGridWorldSequentialEnv(
    config_dict=config_content, 
    max_time_step=4000, 
    time_limit=900, 
    run_name=exp_name
)

print("hi")


for episode in range(num_episodes):
    print()
    print("++++++++++++++ Running episode", episode, "+++++++++++++++")
    print()
    env.reset(return_info=True, options={"episode": episode})
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
