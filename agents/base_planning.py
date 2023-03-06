from functools import lru_cache
from gym_novel_gridworlds2.agents.agent import Agent
from gym_novel_gridworlds2.state import State
from gym_novel_gridworlds2.state.dynamic import Dynamic
from gym.spaces import Discrete
import json

from copy import deepcopy

from utils.pddl_utils import generate_pddl
from utils.plan_utils import call_planner

JSON_CONFIG_PATH = "config/polycraft_gym_main.json"


@lru_cache(maxsize=32)
def get_config_json():
    with open(JSON_CONFIG_PATH) as f:
        config_json = json.load(f)
    return config_json

class BasePlanningAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_buffer = []
        self.done = False
    
    def get_observation_space(self, map_size: tuple, other_size: int):
        return Discrete(1)
    
    def get_observation(self, state, dynamic):
        # raise NotImplementedError("Get observation for " + self.name + " is not implemented.")
        pddl_domain, pddl_problem = generate_pddl(get_config_json(), state, dynamic)
        self.pddl_domain = pddl_domain
        self.pddl_problem = pddl_problem
        return [0]

    def plan(self):
        with open("pddl_domain.pddl", "w") as f:
            f.write(self.pddl_domain)
        with open("pddl_problem.pddl", "w") as f:
            f.write(self.pddl_problem)
        plan, translated = call_planner("pddl_domain.pddl", "pddl_problem.pddl")
        if translated is not None:
            self.action_buffer = deepcopy(translated)
            self.action_buffer.reverse()
            print("Found Plan:")
            for item in plan:
                print("    ", item)
        else:
            print("No Plan Found. Sleeping.")
            self.done = True

    def policy(self, observation):
        if self.done:
            return self.action_set.action_index["nop"]
        elif self.action_buffer == []:
            self.plan()
        
        # if the plan exists, execute the first action
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop()
            if len(self.action_buffer) == 0:
                self.done = True
            return self.action_set.action_index[action]

        return self.action_set.action_index("nop")
