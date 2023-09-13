from typing import Tuple
import gymnasium as gym
from agents import BasePlanningAgent

from .single_agent import SingleAgentEnv

REWARDS = {
    "step": -1,
    "plan_fit": 1
}


class RewardShapingWrapper(gym.Wrapper):
    """
    An environment that gives rewards given if the action picked matches
    the first action in the plan.
    """
    def __init__(self, env: gym.Env):
        self.env = env
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self._update_reward(terminated, truncated, reward)
        return obs, reward, terminated, truncated, info


    def _update_reward(self, terminated, truncated, reward):
        # if episode finished, just assign reward
        if terminated or truncated:
            return reward
        
        # replan and assign rewards based on planner result
        agent: BasePlanningAgent = self.unwrapped.agent_manager.agents[self.get_wrapper_attr("agent_name")].agent
        old_plan_len = agent.pddl_plan.count('\n') + 1
        agent.plan()
        new_plan_len = agent.pddl_plan.count('\n') + 1

        if new_plan_len < old_plan_len and "nop" not in agent.pddl_plan:
            return REWARDS["plan_fit"]
        else:
            return REWARDS["step"]

