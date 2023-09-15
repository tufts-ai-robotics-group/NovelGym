from typing import Tuple
import gymnasium as gym
from agents import BasePlanningAgent

from .single_agent import SingleAgentEnv

from gym_novel_gridworlds2.actions import ActionSet

import queue

REWARDS = {
    "step": -1,
    "plan_fit": 5
}

def is_good_goal(goal: tuple):
    return True


class RSPreplannedSubgoal(gym.Wrapper):
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
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._update_reward(action, terminated, truncated, reward)
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options={}):
        result = self.env.reset(seed, options)

        agent: BasePlanningAgent = self.unwrapped.agent_manager.agents[self.get_wrapper_attr("agent_name")].agent
        plan_success = agent.plan()
        action_buffer = agent.action_buffer # reversed action buffer
        if plan_success and "nop" not in action_buffer[0]:
            self.subgoals = [
                action[0] for action in action_buffer if is_good_goal(action[1][0])
            ]
        else:
            self.subgoals = []
        if self.unwrapped.render_mode == "human":
            print("sub goals:")
            for goal in reversed(self.subgoals):
                num = agent.action_set.action_index[goal]
                print("{:<4}{}".format(num, goal))
            print()
        return result
    
    def convert_action_to_name(self, action: int):
        agent_name = self.get_wrapper_attr("agent_name")
        action_set: ActionSet = self.unwrapped.agent_manager.agents[agent_name].action_set
        return action_set.actions[action][0]

    def _update_reward(self, action, terminated, truncated, reward):
        # if episode finished, just assign reward
        if terminated or truncated:
            return reward
        
        # replan and assign rewards based on planner result
        action_name = self.convert_action_to_name(action)

        if len(self.subgoals) > 0 and action_name == self.subgoals[-1]:
            self.subgoals.pop()
            if self.unwrapped.render_mode == "human":
                print(f"hit {action_name}. got plan fit reward")
            return REWARDS["plan_fit"]
        else:
            return REWARDS["step"]

