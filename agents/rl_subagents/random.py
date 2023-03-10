from gym_novel_gridworlds2.agents import RandomAgent
from agents.rl_subagents.base import BaseRLAgent


class RLRandom(RandomAgent, BaseRLAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
