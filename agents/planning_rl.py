from functools import lru_cache
from gym_novel_gridworlds2.agents.agent import Agent
from gym_novel_gridworlds2.state import State
from gym_novel_gridworlds2.state.dynamic import Dynamic
from gym.spaces import Discrete
import json

from copy import deepcopy

from utils.pddl_utils import generate_pddl
from utils.plan_utils import call_planner
from agents.base_planning import BasePlanningAgent, get_config_json
from agents.base_rl import BaseRLAgent
from gym_novel_gridworlds2.agents import RandomAgent
from gym_novel_gridworlds2.utils.json_parser import import_module

JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
PDDL_DOMAIN = "pddl_domain.pddl"
PDDL_PROBLEM = "pddl_problem.pddl"

class PlanningRLAgent(BasePlanningAgent):
    def __init__(self, rl_module=None, rl_module_params={}, **kwargs):
        super().__init__(**kwargs)
        self.action_buffer = []
        self.done = False
        self.last_action = None
        
        # RL Mode
        self.rl = False

        # RL Agent: an agent inside an agent.
        if rl_module is not None:
            RLModule = import_module(rl_module)
            self.rl_agent = RLModule(**rl_module_params)
        else:
            self.rl_agent = RandomAgent()
        
    
    def plan(self):
        with open(JSON_CONFIG_PATH, "w") as f:
            f.write(self.pddl_domain)
        with open(PDDL_PROBLEM, "w") as f:
            f.write(self.pddl_problem)
        plan, translated = call_planner(JSON_CONFIG_PATH, PDDL_PROBLEM)
        if translated is not None:
            self.action_buffer = deepcopy(translated)
            self.action_buffer.reverse()
            print("Found Plan:")
            for item in plan:
                print("    ", item)
        else:
            print("No Plan Found. Will run RL to rescue.")
            self.rl = True
    

    def update_metadata(self, metadata: dict):
        if not self.rl and metadata["result"] != "SUCCESS":
            print("Failed Action", self.last_action, "in the plan.")
            print("Entering RL Mode")
            self.rl = True
            self.rl_agent
        self.rl_agent.update_metadata(metadata)
    

    def get_observation_space(self, map_size: tuple, other_size: int):
        return self.rl_agent.get_observation_space(map_size, other_size)


    def get_observation(self, state, dynamic):
        """
        For the planning agent, we don't have observation space.
        So the observation is just a wrapper around the RL observation func.
        """
        if self.action_buffer == []:
            pddl_domain, pddl_problem = generate_pddl(get_config_json(), state, dynamic)
            self.pddl_domain = pddl_domain
            self.pddl_problem = pddl_problem

        return self.rl_agent.get_observation(state, dynamic)


    def _run_rl(self, observation):
        return self.rl_agent.policy(observation)


    def policy(self, observation):
        if self.rl:
            return self._run_rl(observation)
        
        if self.done:
            self.last_action = "nop"
            return self.action_set.action_index["nop"]
        elif self.action_buffer == []:
            self.plan()
        
        # if the plan exists, execute the first action
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop()
            # if len(self.action_buffer) == 0:
            #     self.done = True
            self.last_action = action
            return self.action_set.action_index[action]

        self.last_action = "nop"
        return self.action_set.action_index["nop"]
