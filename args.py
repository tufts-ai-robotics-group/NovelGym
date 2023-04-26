import argparse

from config import NOVELTIES, OBS_TYPES, HINTS, POLICIES, POLICY_PROPS, NOVEL_ACTIONS, OBS_GEN_ARGS


parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
# parser.add_argument("filename", type=str, nargs='+', help="The path of the config file.", default="polycraft_gym_main.json")
parser.add_argument(
    "--novelty", '-n',
    type=str, 
    help="The name of the novelty.", 
    required=False,
    default="mi_cantplan",
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
    '--seed', '-s',
    type=str,
    help="The seed.",
    required=False,
    default=None
)
parser.add_argument(
    '--num_threads', '-j',
    type=int,
    help="Number of sub threads used to run the env.",
    required=False,
    default=4
)
parser.add_argument(
    '--logdir', '-o',
    type=str,
    help="The directory to save the logs.",
    required=False,
    default="results"
)
parser.add_argument(
    '--obs_type', '-b',
    type=str,
    help="Type of observation.",
    required=False,
    default="lidar_all",
    choices=OBS_TYPES.keys()
)
parser.add_argument(
    '--rl_algo', '-a',
    type=str,
    help="The algorithm for RL.",
    required=False,
    default="dqn",
    choices=POLICIES.keys()
)
parser.add_argument(
    '--metadata',
    help="Print metadata about the training and quit.",
    default=False,
    action='store_true'
)


verbose = False

args = parser.parse_args()
