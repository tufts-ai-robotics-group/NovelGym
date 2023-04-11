import os
import argparse
import tianshou as ts
import gymnasium as gym
import socket
import json
from typing import Type

from net.basic import BasicNet
from config import NOVELTIES, OBS_TYPES, HINTS, RL_ALGOS, NETS
from diarc.utils import save_pddl
from diarc.utils import recv_socket_data, send_socket_data, error_recovery

from envs.diarc_env import DiarcRapidLearn
from obs_convertion.base import ObservationGenerator
from utils.executor_discovery import ExecutorDiscoverer

parser = argparse.ArgumentParser(description="Rapid-Learn Diarc Socket Interface")
parser.add_argument('-p', '--port', type=int, default=int(os.environ.get("RL_PORT") or 5000), help='port to listen on')
parser.add_argument('-h', '--host', type=str, default=os.environ.get("RL_HOST") or 'localhost', help='host to listen on')
parser.add_argument('-o', '--obs_type', type=str, default=os.environ.get("OBS_TYPE") or 'lidar_all', help='observation type', choices=OBS_TYPES.keys())
parser.add_argument('--model_dir', type=str, default='models', help='model directory')
parser.add_argument('--rl_algo', type=str, help="The algorithm for RL.", required=False, default="dqn", choices=RL_ALGOS.keys())
parser.add_argument('--nn_model', type=str, help="The neural network model.", required=False, default="basic", choices=NETS.keys())

args = parser.parse_args()

DEBUG_DIARC = False

@error_recovery
def handle_rl_task(
    conn: socket.socket, 
    executor_discoverer: ExecutorDiscoverer,
    RepGenerator: Type[ObservationGenerator]
):
    """
    Handle a single RL task.
    """
    state_dict: dict = None
    data_dict: dict = None
    while state_dict is None:
        # in case extra state update json is being sent between 
        # the RESET and the new initialization json,
        # or if extra reset / replan got sent,
        # we discard everything until we find an initialization json.
        data = recv_socket_data(conn)
        data_decoded = data.decode('unicode_escape')
        if "RESET" not in data_decoded.upper() and "REPLAN" not in data_decoded.upper(): # we have separate handlers for replan and reset results
            data_dict = json.loads(data_decoded, strict=False)
            state_dict = data_dict.get("state")

    print("New RL task initiated.")
    failed_action = state_dict["failedAction"]
    mode = DiarcRapidLearn.RLMode.CANT_PLAN if failed_action == "cannotplan" \
        else DiarcRapidLearn.RLMode.FAILED_ACTION
    
    # print pddl if debug mode is on
    if DEBUG_DIARC:
        try:
            save_pddl(data_dict.get("domain"), failed_action)
        except Exception:
            pass
    
    failed_action = state_dict['state']['action'][1:-1].replace(" ", "_")
    env = DiarcRapidLearn( 
        mode=mode,
        conn=conn,
        RepGenerator=RepGenerator,
        rep_gen_args={},
        init_json=data_dict,
        episode=0
    )
    policy = executor_discoverer.get_executor(
        failed_action=env.rep_gen.failed_action,
        action_set=env.rep_gen.action_set,
        observation_space=env.observation_space
    )
    
    

if __name__ == "__main__":
    # observation generator
    RepGenerator = OBS_TYPES[args.obs_type]


