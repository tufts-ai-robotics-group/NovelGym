import argparse
import torch
from config import NOVELTIES, OBS_TYPES, HINTS, POLICIES, POLICY_PROPS, NOVEL_ACTIONS, OBS_GEN_ARGS, AVAILABLE_ENVS


parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
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
    type=int,
    help="The seed.",
    required=False,
    default=None
)
parser.add_argument(
    '--num_threads', '-j',
    type=int,
    help="Number of sub threads used to run the env.",
    required=False,
    default=None
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
    default="ppo",
    choices=POLICIES.keys()
)
parser.add_argument(
    '--metadata',
    help="Print metadata about the training and quit.",
    default=False,
    action='store_true'
)
parser.add_argument(
    '--exp_name',
    help="The name of the experiment, used to save results.",
    default="default_exp"
)
parser.add_argument(
    '--env',
    help="The type of environment.",
    default="pf",
    choices=AVAILABLE_ENVS.keys()
)
parser.add_argument(
    '--resume', '-r',
    help="whether to resume training from a saved checkpoint.",
    action='store_true'
)
parser.add_argument(
    '--checkpoint', '--ckpt',
    help="The path to the checkpoint to load the model. This is used to fine tune a model. To resume training, use --resume instead.",
    default=None
)
parser.add_argument(
    '--lr', 
    help="Learning Rate",
    default=None
)
parser.add_argument(
    '--hidden_sizes', 
    help="Size of the hidden layer, separated by comma.",
    default=None
)
parser.add_argument(
    '--device', '-d',
    help="device to be run on",
    default='cuda' if torch.cuda.is_available() else 'cpu'
)

verbose = False
