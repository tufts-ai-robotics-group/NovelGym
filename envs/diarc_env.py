from enum import Enum
from typing import Tuple, Type
import socket
import json
import gymnasium as gym
from diarc.utils import has_substr_in_buffer_blocking, has_substr_in_buffer, recv_socket_data, socket_buf_is_not_empty
from utils.item_encoder import SimpleItemEncoder
from obs_convertion.base import ObservationGenerator
from config import REWARDS
import numpy as np

class DiarcRapidLearn(gym.Env):
    class RLMode(Enum):
        FAILED_ACTION = 1
        CANT_PLAN = 2
    
    def __init__(
            self, 
            mode: "RLMode", 
            conn: socket.socket, 
            RepGenerator: Type[ObservationGenerator], 
            rep_gen_args: dict, 
            init_json: dict, 
            episode: int=0
        ):
        self.mode = mode
        self.RepGeneratorModule = RepGenerator
        self.rep_gen_args = rep_gen_args
        self.episode = episode
        self.rep_gen = RepGenerator(json_input=init_json, **rep_gen_args)
        self.conn = conn

        self.observation_space = self.rep_gen.observation_space
        self.action_space = gym.spaces.Discrete(len(self.rep_gen.action_set))
        self.last_obs = self.rep_gen.generate_observation(init_json['state'])


    def step(self, action):
        action_name = self.rep_gen.action_set[action]
        send_success, reset_results = self._send_action(action_name)

        if not send_success:
            return self.last_obs, *reset_results

        obs, reward, done, truncated, info = self._parse_receive_replan_result(self.conn)

    # TODO
    def _parse_receive_replan_result(self, conn) -> Tuple[int, bool, Tuple[bool, str]]:
        """
        Given a connection, receives, parses and 
        returns the reward, whether replan is successful or not, 
        and whether we got reset in the middle or not.
        (reward, replan_success, (got_reset, reset_result_str))
        """
        replan_success = False

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

        return replan_success
    

    def _send_action(self, conn: socket.socket, data_to_send: str, should_chop=False) -> bool:
        """
        If the buffer is empty, send the data and return True (meaning that the data was sent).
        If the buffer is not empty, do not send the data and return False (meaning that the data was not sent).
        If got reset, it will automatically end the episode.
        """
        buffer_has_data, data = socket_buf_is_not_empty(conn)
        data_decoded = data.decode("unicode_escape")
        if buffer_has_data:
            has_reset_in_buffer, results = self._check_handle_reset(data_decoded, should_chop=should_chop)
        else:
            has_reset_in_buffer, results = False, ()
        
        # buffer has data
        if buffer_has_data and has_reset_in_buffer:
            return False, results
        else:
            # nothing in buffer or not reset in buffer, ignore.
            conn.sendall((data_to_send + '\r\n').encode())
            return True, ()


    def _check_handle_reset(self, data_decoded): 
        """
        checks and handles reset in the buffer, given the data.
        returns: is_reset, success or not
        """
        if "reset false" in data_decoded.lower(): 
            return True, False
        elif "reset true" in data_decoded.lower():
            # if we got reset from DIARC and we achieved the main goal then
            # end the episode and save
            return True, True
        return False, ()

    def _receive_json(self, conn: socket.socket) -> Tuple[dict, bool]:
        # wait for the update state
        while not has_substr_in_buffer_blocking(conn, "{", num_bytes=2):
            data = recv_socket_data(conn)
            data_decoded = data.decode('unicode_escape')

            # if we got reset in the middle, just return
            if "reset true" in data_decoded.lower():
                return None, True
            elif "reset false" in data_decoded.lower():
                return None, False
            else:
                # send the message so that diarc knows we received the msg.
                conn.send(b"received.\r\n")

        # parse the update state
        state_str = recv_socket_data(conn).decode('unicode_escape')
        state_dict = json.loads(state_str)
        return state_dict, False
    

    def _receive_process_state(self, conn: socket.socket) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Receives the update state from the server.
        Updates the internal representation of the state.

        returns the observation.
        TODO: reset
        """
        state_json, reset_result = self._receive_json(conn)
        # in case of reset, return the last observation
        if state_json is None:
            if reset_result:
                return self.last_obs, REWARDS['positive'], True, False, {}
            else:
                return self.last_obs, REWARDS['step'], False, True, {}
        
        # process the state
        if self.mode == self.RLMode.CANT_PLAN:
            # if the reason for starting RL is that it cannot plan anyway,
            # then we expect diarc to send us full init json.
            state_dict = state_dict.get("state") or state_dict
        else:
            # if it's in action failed mode, and received a new init json
            # it means nested novelties:
            # a new novelty appears when trying to solve an old
            # novelty. We don't handle that and just give up
            print("Give Up")
            msg_sent = self._send_action(conn, "giveup")
            return self.last_obs, REWARDS['step'], False, True, {}
        
        # get the state representation
        obs = self.rep_gen.generate_observation(state_dict)
        self.last_obs = obs

        # is the episode done?
        if self.mode == self.RLMode.CANT_PLAN:
            self._send_action(conn, "replan")
            is_plannable = self._parse_receive_replan_result(conn)
        else:
            effect_met, future_effect_met = self.rep_gen.check_if_effects_met(state_dict)
            is_plannable = effect_met or future_effect_met
        
        # assign reward based on plannable or not
        if not is_plannable:
            reward = REWARDS['step']
            is_done, is_truncated = False, False
        else:
            if self.mode == self.RLMode.CANT_PLAN:
                # if RL gets out of an unplannable situation, then it's a success
                reward = REWARDS['positive']
                is_done, is_truncated = True, False
            else:
                # need to check if it's actually plannable or not in the symbolic
                # side.
                self._send_action(conn, "replan")
                planner_unstuck = self._parse_receive_replan_result(conn)
                if planner_unstuck:
                    reward = REWARDS['positive']
                    is_done, is_truncated = True, False
                else:
                    reward = REWARDS['negative']
                    is_done, is_truncated = True, False
        return obs, reward, is_done, is_truncated, {}

    def reset(self, seed=None, options={}):
        self.episode += 1
        return self.last_obs, {}


    
    # def end_episode(self, conn: socket.socket, reward, success, should_chop=False):
    #     self.disc_exec.end_episode(reward, success, should_chop)
    #     self._send_action(conn, "DONE\r\n")
    
    # def solve_task(self, conn: socket.socket, input):
    #     """
    #     Runs one episode of the task.
    #     """
    #     try:
    #         self.env = Polycraftv2Env(input)
    #     except Exception as e:
    #         # when there's something wrong with parsing the init json, send done
    #         # and quit.
    #         # since there might be failed actions that are not in the action
    #         # list, but diarc is still sending us that.
    #         print("Error parsing init json due to unknown action name or wrong json format. Sending Done and giving up.")
    #         print(e)
    #         self._send_action(conn, "DONE\r\n")
    #         return

    #     # this gets the data required to initialize the DE class
    #     data = self.env.init_info()
    #     data_decoded = ""

    #     # this instantiates the discover executor class which in turn instantiates the PG class
    #     self.disc_exec = DiscoverExecutor(**data)

    #     time_step = 0
    #     while True:
    #         if time_step == 0:
    #             # if it's the initialization phase, directly reuse the state
    #             # field in the initialization json
    #             if 'state' not in input:
    #                 print("expected initialization json but got:", data)
    #                 break
    #             data_dict = input['state']
    #         else:
    #             # when it's not the initialization time step,
    #             # receive new state update 
    #             data = recv_socket_data(conn)
    #             data_decoded = data.decode('unicode_escape')

    #             # if the diarc episode is over, end the rl episode
    #             if self._check_handle_reset(data_decoded):
    #                 break
    #             data_dict = json.loads(data_decoded, strict=False)

    #             if self.mode == self.RLMode.CANT_PLAN:
    #                 # if it's can't plan mode, we expect a new init json each time.
    #                 data_dict = data_dict.get("state") or data_dict
    #             elif "state" in data_dict:
    #                 # if it's in action failed mode, and received a new init json
    #                 # it means nested novelties:
    #                 # a new novelty appears when trying to solve an old
    #                 # novelty. We don't handle that and just give up
    #                 print("Give Up")
    #                 msg_sent = self._send_action(conn, "giveup")
    #                 if not msg_sent:
    #                     break
    #                 return
            
    #         # runs the rl policy to get the action, continue episode 
    #         # assigning -1 reward automatically
    #         try:
    #             action: str = self.get_action(data_dict)
    #         except SimpleItemEncoder.TooManyItemTypes:
    #             # we got initialized in the secondary 
    #             msg_sent = self._send_action(conn, "giveup")
    #             if not msg_sent:
    #                 break

    #         if action.lower() == "replan":
    #             # In any of the mode,
    #             # if we asked for a replan, send the replan, and handle the result.
    #             msg_sent = self._send_action(conn, "replan")
    #             if not msg_sent:
    #                 break

    #             # get reward
    #             reward, success, (msg_sent, reset_result) = self._parse_receive_replan_result(conn)
    #             self.end_episode(conn, reward, success, should_chop=False)
    #             break
    #         elif self.mode == self.RLMode.CANT_PLAN:
    #             # if we're in can't plan mode
    #             # send the action and record the result
    #             # in this mode we will always send replan after every action anyway.
    #             msg_sent = self._send_action(conn, action, should_chop=True)
    #             if not msg_sent:
    #                 break
                
    #             data = recv_socket_data(conn)
    #             if self._check_handle_reset(data.decode('unicode_escape')):
    #                 break

    #             # immediately send replan afterwards
    #             msg_sent = self._send_action(conn, "replan")
    #             if not msg_sent:
    #                 break
    #             reward, success, (got_reset, reset_result) = self._parse_receive_replan_result(conn)

    #             # if for the current episode we got replan success, that means
    #             # RL finished its task and we end the episode and assign the reward.
    #             if got_reset and reset_result is not None:
    #                 self._check_handle_reset(reset_result)
    #                 break
    #             elif success:
    #                 self.end_episode(conn, reward, success, should_chop=False)
    #                 break
    #         else:
    #             # if we aren't in the can't plan mode,
    #             # i.e. in the failed action mode
    #             # just send the action and do not replan.
    #             msg_sent = self._send_action(conn, action, should_chop=True)
    #             if not msg_sent:
    #                 break
    #         time_step += 1
    #         self.disc_exec.end_step(reward=-1)


    # def get_action(self, data):
    #     '''
    #     one time step of the task
    #     Gets the action from the agent. 
    #     It either returns "replan" or an "action" after doing forward pass.
    #     '''
    #     obs = self.env.generate_observation(data)
    #     effects_met = self.env.check_if_effects_met(data)
    #     # print("----------------------")
    #     # print("New Observation:", obs)
    #     # print("Effects met:", effects_met)
    #     if effects_met[0] == True or effects_met[1] == True:
    #         # effect met, end the episode.
    #         print("Effects met. Replan.")
    #         return "Replan"
    #     else:
    #         # otherwise, continue the episode.
    #         # reward = -1
    #         self.action = self.disc_exec.step_episode(obs) 
    #         action_name = self.env.get_action_name(self.action)
    #         print("Action Selected:", action_name)
    #         return action_name
