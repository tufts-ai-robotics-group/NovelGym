from enum import Enum
import time
import socket
import json
from typing import Tuple
import params

# import local classes
from env_utils import Polycraftv2Env, SimpleItemEncoder
from discover_executor import DiscoverExecutor

import os
from utils import has_substr_in_buffer_blocking, has_substr_in_buffer, save_pddl
from utils import recv_socket_data, error_recovery, socket_buf_is_not_empty

DEBUG_DIARC = os.environ.get("DEBUG_DIARC") == "true"

class RAPidLearn(object):
    class RLMode(Enum):
        FAILED_ACTION = 1
        CANT_PLAN = 2
    
    def __init__(self, mode: RLMode = RLMode.FAILED_ACTION) -> None:
        self.mode = mode
        pass

    def parse_receive_replan_result(self, conn) -> Tuple[int, bool, Tuple[bool, str]]:
        """
        Given a connection, receives, parses and 
        returns the reward, whether replan is successful or not, 
        and whether we got reset in the middle or not.
        (reward, replan_success, (got_reset, reset_result_str))
        """
        replan_success = False
        response_is_json = False

        # loop until the next item in the buffer is a json or replan is successful.
        while not replan_success and not has_substr_in_buffer_blocking(conn, "{", num_bytes=2):
            data = recv_socket_data(conn)
            data_decoded = data.decode('unicode_escape')

            # parse the replan result
            replan_success = "replan" in data_decoded.lower() and data_decoded.split(" ")[1].lower() == "true" # replan true received

            # if we got reset in the middle, just return
            if "reset" in data_decoded.lower():
                return 0, False, (True, data_decoded)
            else:
                # send the message so that diarc knows we received the msg.
                conn.send(b"received.\r\n")

        if replan_success:
            # if success, assign a big positive reward.
            reward = 1000
            done = True
        # TODO implement this situation
        # elif replan_result['Replan_Result'] == 'Failed_same_action':
        #     reward = 250
        #     done = True
        elif not replan_success:
            # If replan failed, i.e. entering a catastrophic situation,
            # assign reward based on the mode.
            if self.mode == self.RLMode.CANT_PLAN:
                # if the reason for starting RL is that it cannot plan anyway,
                # don't assign negative reward since it's not RL's fault.
                # -1 is already assigned in the get_action function.
                reward = 0
            else:
                # otherwise, penalize RL for making it unplannable.
                reward = -250
            done = False


        return reward, done, (False, None)
    

    def send_action(self, conn: socket.socket, data_to_send: str, should_chop=False):
        """
        If the buffer is empty, send the data and return True (meaning that the data was sent).
        If the buffer is not empty, do not send the data and return False (meaning that the data was not sent).
        If got reset, it will automatically end the episode.
        """
        buffer_has_data, data = socket_buf_is_not_empty(conn)
        data_decoded = data.decode("unicode_escape")
        if not buffer_has_data or not self.check_handle_reset(data_decoded, should_chop=should_chop):
            # nothing in buffer or not reset in buffer, ignore.
            conn.sendall((data_to_send + '\r\n').encode())
            return True
        else: 
            # reset in buffer, quit the process
            return False


    def check_handle_reset(self, data_decoded, should_chop=False): # reset_handler is a better name
        """
        Checks if the episode is over and returns the reward.
        How to check if the previous action was executed????????
        if the previous action was executed in DIARC, then we add the reward, dont modify the NW.
        If the previous acttion was not executed in DIARC, then we modify the reward and modify the NW.
        """
        if "reset false" in data_decoded.lower(): 
            # if we got reset from DIARC and we did not achieve the main goal then
            # end the episode and save the model
            reward, success = -100, False
            self.disc_exec.end_episode(reward, success, should_chop=should_chop)
            return True
        elif "reset true" in data_decoded.lower():
            # if we got reset from DIARC and we achieved the main goal then
            # end the episode and save
            reward = params.POSITIVE_REINFORCEMENT
            success = True
            self.disc_exec.end_episode(reward, success, should_chop=should_chop)
            return True
        return False
    
    
    def end_episode(self, conn: socket.socket, reward, success, should_chop=False):
        self.disc_exec.end_episode(reward, success, should_chop)
        self.send_action(conn, "DONE\r\n")

    
    def solve_task(self, conn: socket.socket, input):
        """
        Runs one episode of the task.
        """
        try:
            self.env = Polycraftv2Env(input)
        except Exception as e:
            # when there's something wrong with parsing the init json, send done
            # and quit.
            # since there might be failed actions that are not in the action
            # list, but diarc is still sending us that.
            print("Error parsing init json due to unknown action name or wrong json format. Sending Done and giving up.")
            print(e)
            self.send_action(conn, "DONE\r\n")
            return

        # this gets the data required to initialize the DE class
        data = self.env.init_info()
        data_decoded = ""

        # this instantiates the discover executor class which in turn instantiates the PG class
        self.disc_exec = DiscoverExecutor(**data)

        time_step = 0
        while True:
            if time_step == 0:
                # if it's the initialization phase, directly reuse the state
                # field in the initialization json
                if 'state' not in input:
                    print("expected initialization json but got:", data)
                    break
                data_dict = input['state']
            else:
                # when it's not the initialization time step,
                # receive new state update 
                data = recv_socket_data(conn)
                data_decoded = data.decode('unicode_escape')

                # if the diarc episode is over, end the rl episode
                if self.check_handle_reset(data_decoded):
                    break
                data_dict = json.loads(data_decoded, strict=False)

                if self.mode == self.RLMode.CANT_PLAN:
                    # if it's can't plan mode, we expect a new init json each time.
                    data_dict = data_dict.get("state") or data_dict
                elif "state" in data_dict:
                    # if it's in action failed mode, and received a new init json
                    # it means nested novelties:
                    # a new novelty appears when trying to solve an old
                    # novelty. We don't handle that and just give up
                    print("Give Up")
                    msg_sent = self.send_action(conn, "giveup")
                    if not msg_sent:
                        break
                    return
            
            # runs the rl policy to get the action, continue episode 
            # assigning -1 reward automatically
            try:
                action: str = self.get_action(data_dict)
            except SimpleItemEncoder.TooManyItemTypes:
                # we got initialized in the secondary 
                msg_sent = self.send_action(conn, "giveup")
                if not msg_sent:
                    break

            if action.lower() == "replan":
                # In any of the mode,
                # if we asked for a replan, send the replan, and handle the result.
                msg_sent = self.send_action(conn, "replan")
                if not msg_sent:
                    break

                # get reward
                reward, success, (msg_sent, reset_result) = self.parse_receive_replan_result(conn)
                self.end_episode(conn, reward, success, should_chop=False)
                break
            elif self.mode == self.RLMode.CANT_PLAN:
                # if we're in can't plan mode
                # send the action and record the result
                # in this mode we will always send replan after every action anyway.
                msg_sent = self.send_action(conn, action, should_chop=True)
                if not msg_sent:
                    break
                
                data = recv_socket_data(conn)
                if self.check_handle_reset(data.decode('unicode_escape')):
                    break

                # immediately send replan afterwards
                msg_sent = self.send_action(conn, "replan")
                if not msg_sent:
                    break
                reward, success, (got_reset, reset_result) = self.parse_receive_replan_result(conn)

                # if for the current episode we got replan success, that means
                # RL finished its task and we end the episode and assign the reward.
                if got_reset and reset_result is not None:
                    self.check_handle_reset(reset_result)
                    break
                elif success:
                    self.end_episode(conn, reward, success, should_chop=False)
                    break
            else:
                # if we aren't in the can't plan mode,
                # i.e. in the failed action mode
                # just send the action and do not replan.
                msg_sent = self.send_action(conn, action, should_chop=True)
                if not msg_sent:
                    break
            time_step += 1
            self.disc_exec.end_step(reward=-1)

    def get_action(self, data):
        '''
        one time step of the task
        Gets the action from the agent. 
        It either returns "replan" or an "action" after doing forward pass.
        '''
        obs = self.env.generate_observation(data)
        effects_met = self.env.check_if_effects_met(data)
        # print("----------------------")
        # print("New Observation:", obs)
        # print("Effects met:", effects_met)
        if effects_met[0] == True or effects_met[1] == True:
            # effect met, end the episode.
            print("Effects met. Replan.")
            return "Replan"
        else:
            # otherwise, continue the episode.
            # reward = -1
            self.action = self.disc_exec.step_episode(obs) 
            action_name = self.env.get_action_name(self.action)
            print("Action Selected:", action_name)
            return action_name

@error_recovery
def handle_rl_task(conn: socket.socket):
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
    mode = RAPidLearn.RLMode.CANT_PLAN if failed_action == "cannotplan" \
        else RAPidLearn.RLMode.FAILED_ACTION
    
    # print pddl if debug mode is on
    if DEBUG_DIARC:
        try:
            save_pddl(data_dict.get("domain"), failed_action)
        except Exception:
            pass

    rl = RAPidLearn(mode=mode)
    rl.solve_task(conn, data_dict)


HOST = os.environ.get("RL_HOST") or '127.0.0.1'
PORT = os.environ.get("RL_PORT") or 6013

if __name__ == "__main__":
    if DEBUG_DIARC:
        print("Diarc debug mode is on. PDDL will be saved.")
    # start the connection
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, int(PORT)))
        s.listen()
        print(f"Listening at {HOST}:{PORT}")
        prev_content = None
        while True:
            # waiting for agent to connect
            conn, addr = s.accept()
            try:
                with conn:
                    conn.setblocking(False)
                    print('Connected by', addr)
                    while True:
                        handle_rl_task(conn)
            except socket.timeout as e:
                prev_content = None
                print("Socket connection with", addr, "closed.")
                continue
