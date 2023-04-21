from functools import lru_cache
from typing import List, Tuple
from gym_novel_gridworlds2.agents.agent import Agent
from gym_novel_gridworlds2.state import State
from gym_novel_gridworlds2.state.dynamic import Dynamic
from gym.spaces import Discrete
import json

from typing import Optional

from copy import deepcopy

from utils.pddl_utils import KnowledgeBase
from utils.plan_utils import call_planner

import tempfile
import os

JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
PDDL_DOMAIN = "pddl_domain.pddl"
PDDL_PROBLEM = "pddl_problem.pddl"

@lru_cache(maxsize=32)
def get_base_config_json():
    with open(JSON_CONFIG_PATH) as f:
        config_json = json.load(f)
    return config_json

class BasePlanningAgent(Agent):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self._reset()
        self.kb = None


    def _reset(self):
        self.action_buffer: List[tuple] = []
        self.done = False
        self.last_action: Optional[Tuple[str, tuple]] = None
        self.failed_action: Optional[Tuple[str, tuple]] = None
        self.pddl_plan: List[tuple] = []
        
        # rl mode
        self.stuck = False
    
    def get_observation_space(self, map_size: tuple, other_size: int):
        return Discrete(1)
    
    def get_observation(self, state, dynamic):
        # raise NotImplementedError("Get observation for " + self.name + " is not implemented.")
        self.kb = KnowledgeBase(get_base_config_json())
        pddl_domain, pddl_problem = self.kb.generate_pddl(state, dynamic)
        self.state = state
        self.dynamic = dynamic
        self.pddl_domain = pddl_domain
        self.pddl_problem = pddl_problem
        return [0]

    def plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.pddl_domain, self.pddl_problem = self.kb.generate_pddl(self.state, self.dynamic)
            domain_path = os.path.join(tmpdir, PDDL_DOMAIN)
            problem_path = os.path.join(tmpdir, PDDL_PROBLEM)
            with open(domain_path, "w") as f:
                f.write(self.pddl_domain)
            with open(problem_path, "w") as f:
                f.write(self.pddl_problem)
            plan, translated = call_planner(domain_path, problem_path, verbose=self.verbose)
        if translated is not None:
            self.pddl_plan = "\n".join(["(" + " ".join(operator) + ")" for operator in plan])
            self.action_buffer = list(zip(translated, plan))
            self.action_buffer.reverse()
            if self.verbose:
                print("Found Plan:")
                for item in plan:
                    print("    ", item)
            return True
        else:
            if self.verbose:
                print("No Plan Found. Will run RL to rescue.")
            return False
    
    def update_metadata(self, metadata: dict):
        if metadata["gameOver"]:
            self._reset()
        elif not self.stuck and metadata["command_result"]["result"] != "SUCCESS":
            # init rl here
            self.stuck = True
            self.failed_action = self.last_action


    def policy(self, observation):
        if self.stuck:
            # current agent does not include rl module, just do nop.
            return self.action_set.action_index["nop"]
        elif self.done:
            self.last_action = ("nop", ("nop",))
            return self.action_set.action_index["nop"]
        elif self.action_buffer == []:
            plan_result = self.plan()
            if not plan_result:
                # if cannot plan, run RL
                self.pddl_plan = []
                self.stuck = True
                self.failed_action = "cannotplan"
                return self.action_set.action_index["nop"]
        
        # if the plan exists, execute the first action
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop()
            self.last_action = action
            return self.action_set.action_index[action[0]]

        return self.action_set.action_index["nop"]
