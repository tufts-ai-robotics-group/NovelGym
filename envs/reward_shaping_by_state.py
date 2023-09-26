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

INVENTORY_CHANGES = {
    "craft_planks": {
        "planks": 4
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
        "oak_log": 1
    }
}

def _parse_add_sub_goals(pddl_plan):
    pddl_plan = pddl_plan.split("\n")
    sub_goals = []
    for action in reversed(pddl_plan):
        print(action)
        tokens = action.lstrip("(").rstrip(")").split(" ")
        if tokens[0] == "break_diamond_ore":
            sub_goals.append(INVENTORY_CHANGES["break_diamond_ore"])
        elif "break" in tokens[0]:
            sub_goals.append(INVENTORY_CHANGES["break_" + tokens[1]])
        elif "craft" in tokens[0] or "trade" in tokens[0] or "collect" in tokens[0]:
            sub_goals.append(INVENTORY_CHANGES[tokens[0]])
    return sub_goals


def _inventory_goal_met(old_inventory, new_inventory, subgoal):
    print(old_inventory, new_inventory, subgoal)
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
                print(goal)
            print()
        self.last_inventory = None
        return result
    
    def convert_action_to_name(self, action: int):
        agent_name = self.get_wrapper_attr("agent_name")
        action_set: ActionSet = self.unwrapped.agent_manager.agents[agent_name].action_set
        return action_set.actions[action][0]

    def _update_reward(self, action, terminated, truncated, info: dict, reward):
        # if episode finished, just assign reward
        if terminated or truncated:
            return reward
        
        # replan and assign rewards based on planner result
        action_name = self.convert_action_to_name(action)

        if info.get("success", False): # action success
            last_inventory = self.last_inventory
            new_inventory = self._get_copied_agent_inventory()
            
            if _inventory_goal_met(last_inventory, new_inventory, self.subgoals[-1]): # check inventory
                self.subgoals.pop()
                if self.unwrapped.render_mode == "human":
                    print(f"hit {action_name}. got plan fit reward. Next goal:", self.subgoals[-1])
                return REWARDS["plan_fit"]
        return REWARDS["step"]

