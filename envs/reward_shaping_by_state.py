from copy import deepcopy
from typing import Tuple
import gymnasium as gym
from agents import BasePlanningAgent


from gym_novel_gridworlds2.actions import ActionSet
from gym_novel_gridworlds2.envs import NovelGridWorldSequentialEnv

import re

REWARDS = {
    "step": -1,
    "plan_fit": 5
}

REHIT_SUBGOAL_DECAY_FACTOR = 0.5

INVENTORY_CHANGES = {
    "craft_planks": {
    },
    "craft_stick": {
        "stick": 4
    },
    "collect_from_tree_tap": {
        "rubber": 1
    },
    "break_diamond_ore": {
        "diamond": 9
    },
    "craft_pogo_stick": {
        "pogo_stick": 1
    },
    "break_block_of_platinum": {
        "block_of_platinum": 1
    },
    "trade_block_of_titanium_1": {
        "block_of_titanium": 1
    },
    "break_oak_log": {
    }
}

def _parse_add_sub_goals(pddl_plan):
    pddl_plan = pddl_plan.split("\n")
    sub_goals = []
    for action in reversed(pddl_plan):
        tokens = action.lstrip("(").rstrip(")").split(" ")
        if tokens[0] == "break_diamond_ore":
            sub_goals.append(INVENTORY_CHANGES["break_diamond_ore"])
        elif "break" in tokens[0]:
            sub_goals.append(INVENTORY_CHANGES["break_" + tokens[1]])
        elif "craft" in tokens[0] or "trade" in tokens[0] or "collect" in tokens[0]:
            sub_goals.append(INVENTORY_CHANGES[tokens[0]])
    return sub_goals


def _inventory_goal_met(old_inventory, new_inventory, subgoal):
    for item, increment in subgoal.items():
        if new_inventory.get(item, 0) - old_inventory.get(item, 0) < increment:
            return False
    return True


class RSPreplannedStateSubgoal(gym.Wrapper):
    """
    An environment that gives rewards given if the action picked matches
    the first action in the plan.
    """
    def __init__(self, env: gym.Env):
        self.env = env
        self.last_reward = None
        # excludes actions in which we already give extra reward
        # to avoid local minima 
        self.rs_exclude_list = set()
        self.last_inventory = None

        # temporatily store the last subgoal for repeated reward
        self.last_subgoal = None 

        # track and compare with initial inventory. If the algorithm skipped ahead, 
        # then we can give skip some subgoals.
        self.init_inventory = {} 
        self.rehit_subgoal_decay = 1
    
    
    def _get_copied_agent_inventory(self):
        base_env: NovelGridWorldSequentialEnv = self.unwrapped
        agent_rep = base_env.agent_manager.agents[self.get_wrapper_attr("agent_name")]
        return deepcopy(agent_rep.entity.inventory)


    def step(self, action):
        self.last_inventory = self._get_copied_agent_inventory()
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._update_reward(action, terminated, truncated, info, reward)

        return obs, reward, terminated, truncated, info


    def reset(self, seed=None, options={}):
        result = self.env.reset(seed, options)

        agent: BasePlanningAgent = self.unwrapped.agent_manager.agents[self.get_wrapper_attr("agent_name")].agent
        plan_success = agent.plan()
        pddl_plan = agent.pddl_plan
        if plan_success and "nop" not in pddl_plan[0]:
            try:
                self.subgoals = _parse_add_sub_goals(pddl_plan)
            except KeyError as e:
                raise Exception("Unable to add subgoal for Reward Shaping") from e
        else:
            self.subgoals = []
        if self.unwrapped.render_mode == "human":
            agent.verbose = True
            print("sub goals:")
            for goal in reversed(self.subgoals):
                print("   inventory increase: ", goal)
            print()
        self.last_inventory = None
        self.init_inventory = self._get_copied_agent_inventory()
        self.last_subgoal = None # temporatily store the last subgoal for repeated reward
        self.rehit_subgoal_decay = 1
        return result
    
    def convert_action_to_name(self, action: int):
        agent_name = self.get_wrapper_attr("agent_name")
        action_set: ActionSet = self.unwrapped.agent_manager.agents[agent_name].action_set
        return action_set.actions[action][0]

    def _skip_subgoal_if_done(self):
        """
        When a subgoal is completed, skip subsequent subgoals if they have already been completed
        somehow in previous time steps.
        """
        while len(self.subgoals) > 0:
            if not _inventory_goal_met(self.init_inventory, self.last_inventory, self.subgoals[-1]):
                # if the agent did not skip ahead, we do not update the subgoal.
                break
            self.last_subgoal = self.subgoals[-1]
            self.rehit_subgoal_decay = 1
            self.subgoals.pop()
            if self.unwrapped.render_mode == "human":
                print(f"Already completed subgoal {self.last_subgoal}. Next goal:", self.subgoals[-1])
    
    def check_goal_state(self):
        print()
        print("subgoals", self.subgoals)
        print("last goal", self.last_subgoal)
        print("decay", self.rehit_subgoal_decay)
        print()


    def _update_reward(self, action, terminated, truncated, info: dict, reward):
        # if episode finished, just assign reward
        if terminated or truncated:
            return reward
        
        # replan and assign rewards based on planner result
        action_name = self.convert_action_to_name(action)

        if info.get("success", False) and len(self.subgoals) > 0: # action success and have sub goals
            last_inventory = self.last_inventory
            new_inventory = self._get_copied_agent_inventory()
            
            if _inventory_goal_met(last_inventory, new_inventory, self.subgoals[-1]): # check inventory
                self.last_subgoal = self.subgoals[-1]
                self.rehit_subgoal_decay = 1
                self.subgoals.pop()
                if self.unwrapped.render_mode == "human":
                    print(f"hit {action_name}. got plan fit reward. Next goal:", self.subgoals[-1])
                self._skip_subgoal_if_done()
                return REWARDS["plan_fit"]
            elif self.last_subgoal is not None and _inventory_goal_met(last_inventory, new_inventory, self.last_subgoal):
                if self.unwrapped.render_mode == "human":
                    print(f"hit {action_name} again. decay now: {self.rehit_subgoal_decay}")
                self.rehit_subgoal_decay *= REHIT_SUBGOAL_DECAY_FACTOR
                return REWARDS["plan_fit"] * self.rehit_subgoal_decay
        return REWARDS["step"]

