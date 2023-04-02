import argparse

from config import NOVELTIES, OBS_TYPES, HINTS, RL_ALGOS


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


verbose = False

args = parser.parse_args()
