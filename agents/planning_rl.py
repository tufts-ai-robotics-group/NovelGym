from functools import lru_cache
from gym_novel_gridworlds2.agents.agent import Agent
from gym_novel_gridworlds2.state import State
from gym_novel_gridworlds2.state.dynamic import Dynamic
from gym.spaces import Discrete
import yaml

from copy import deepcopy

from utils.pddl_utils import generate_pddl
from utils.plan_utils import call_planner
from agents.base_planning import BasePlanningAgent, get_base_config
from agents.rl_subagents.random import RLRandom
from agents.rl_subagents.base import BaseRLAgent
from gym_novel_gridworlds2.utils.json_parser import import_module

CONFIG_PATH = "config/polycraft_gym_main.yaml"
PDDL_DOMAIN = "pddl_domain.pddl"
PDDL_PROBLEM = "pddl_problem.pddl"

class PlanningRLAgent(BasePlanningAgent):
    def __init__(self, rl_module=None, rl_module_params={}, **kwargs):
        super().__init__(**kwargs)
        self._reset()

        # RL Agent: an agent inside an agent.
        rl_actionset = self.action_set.create_sub_actionset(excluded_actions=["nop", "give_up"])
        if rl_module is not None:
            RLModule = import_module(rl_module)
            self.rl_agent: BaseRLAgent = RLModule(
                id=self.id, 
                action_set=rl_actionset, 
                state=self.state, 
                entity_data=self.entity_data, 
                **rl_module_params
            )
        else:
            self.rl_agent: BaseRLAgent = RLRandom(
                id=self.id, 
                action_set=rl_actionset, 
                state=self.state, 
                entity_data=self.entity_data
            )
    
    def _reset(self):
        self.action_buffer = []
        self.done = False
        self.last_action = None
        self.pddl_plan = []
        
        # rl mode
        self.rl = False
    
    def plan(self):
        self.pddl_domain, self.pddl_problem = generate_pddl(get_base_config(), self.state, self.dynamic)
        with open(PDDL_DOMAIN, "w") as f:
            f.write(self.pddl_domain)
        with open(PDDL_PROBLEM, "w") as f:
            f.write(self.pddl_problem)
        plan, translated = call_planner(PDDL_DOMAIN, PDDL_PROBLEM, verbose=self.verbose)
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
        elif not self.rl and metadata["command_result"]["result"] != "SUCCESS":
            # init rl here
            self._init_rl(self.last_action, self.pddl_domain, self.pddl_plan)
        self.rl_agent.update_metadata(metadata)


    def get_observation_space(self, map_size: tuple, other_size: int):
        return self.rl_agent.get_observation_space(map_size, other_size)


    def get_observation(self, state, dynamic):
        """
        For the planning agent, we don't have observation space.
        So the observation is just a wrapper around the RL observation func.
        """
        self.state = state
        self.dynamic = dynamic
        return self.rl_agent.get_observation(state, dynamic)


    def _run_rl(self, observation):
        while True:
            action = self.rl_agent.policy(observation)
            if action == 0:
                # replan
                plan_result = self.plan()
                self.rl_agent.update_metadata({
                    "command_result": {
                        "command": "replan",
                        "result": "SUCCESS" if plan_result else "FAILED"
                    },
                    "gameOver": False,
                    "goal": {"goalAchieved": False}
                })
                if plan_result:
                    # if replan success, execute the plan.
                    self.rl = False
                    self.last_action = self.action_buffer.pop()
                    return self.action_set.action_index[self.last_action[0]]
                elif self.failed_action != "cannotplan":
                    # if we failed an operator, and the rl policy tells us to replan:
                    # we do want to make sure 
                    self.rl = False
                    return self.action_set.action_index["give_up"]
                else:
                    # if replan failed, continue to run RL if it failed to plan 
                    # because it doesn't know when it's plannable
                    action = None

            else:
                # if rl gives an action, run this action.
                return action - 1


    def _init_rl(self, failed_action, pddl_domain, pddl_plan):
        if self.verbose:
            print("Failed Action", self.last_action[0], "in the plan.")
            print("Entering RL Mode")
        self.rl = True
        self.failed_action = failed_action
        self.rl_agent.init_rl("(" + " ".join(failed_action[1]) + ")", pddl_domain, pddl_plan)


    def policy(self, observation):
        # TODO check for replan
        if self.rl:
            return self._run_rl(observation)
        
        if self.done:
            self.last_action = ("nop", ("nop",))
            return self.action_set.action_index["nop"]
        elif self.action_buffer == []:
            plan_result = self.plan()
            if not plan_result:
                # if cannot plan, run RL
                self.pddl_plan = []
                self._init_rl("cannotplan", self.pddl_domain, self.pddl_plan)
                return self._run_rl(observation)
        
        # if the plan exists, execute the first action
        if len(self.action_buffer) > 0:
            action = self.action_buffer.pop()
            # if len(self.action_buffer) == 0:
            #     self.done = True
            self.last_action = action
            return self.action_set.action_index[action[0]]

        self.last_action = "nop"
        return self.action_set.action_index["nop"]
