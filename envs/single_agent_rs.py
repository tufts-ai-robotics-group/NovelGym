from typing import Tuple
import gymnasium as gym
from agents import BasePlanningAgent

from .single_agent import SingleAgentEnv

REWARDS = {
    "positive": 1000,
    "negative": -250,
    "step": -1,
    "plan_fit": 5
}


class SingleAgentRSShorterPlanEnv(SingleAgentEnv):
    """
    An environment that gives rewards given if the action picked matches
    the first action in the plan.
    """
    def _gen_reward(self):
        # if episode finished, just assign reward
        if self.env.internal_state._goal_achieved:
            return True, False, REWARDS["positive"]
        elif self.env.internal_state._given_up:
            return True, False, REWARDS["negative"]
        elif self.env.dones["agent_0"]:
            return False, True, REWARDS["step"]
        
        # replan and assign rewards based on planner result
        agent: BasePlanningAgent = self.env.agent_manager.agents["agent_0"].agent
        old_plan_len = agent.pddl_plan.count('\n') + 1
        agent.plan()
        new_plan_len = agent.pddl_plan.count('\n') + 1

        if new_plan_len < old_plan_len:
            return False, False, REWARDS["plan_fit"]
        else:
            return False, False, REWARDS["step"]

