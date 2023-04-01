from obs_convertion import LidarAll, OnlyFacingObs, OnlyHinted
import tianshou as ts
import argparse

OBS_TYPES = {
    "lidar_all": LidarAll,
    "only_facing": OnlyFacingObs,
    "only_hinted": OnlyHinted,
}

NOVELTIES = {
    "mi": "novelties/evaluation1/multi_interact/multi_interact.json",
    "kibt": "novelties/evaluation1/key_inventory_break_tree/key_inventory_break_tree.json",
    "rdb": "novelties/evaluation1/random_drop_break/random_drop_break.json",
    "space_ar": "novelties/evaluation1/space_around_crafting_table/space_around_crafting_table.json",
}


HINTS = {
    "mi": "",
    "kibt": str([
        "Sorry, you need a key to trade with me.",
        "(trade_block_of_titanium_1)"
    ]),
    "rdb": "",
    "space_ar": "",
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


verbose = False

args = parser.parse_args()
