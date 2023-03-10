from gym_novel_gridworlds2.agents.agent import Agent
from gym.spaces import Discrete


class BaseRLAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_action = None
    
    def get_observation_space(self, map_size: tuple, other_size: int):
        return Discrete(1)
    
    def get_observation(self, state, dynamic):
        return [0]

    def init_rl(self, failed_action, pddl_domain):
        self.failed_action = failed_action
        self.pddl_domain = pddl_domain

    def policy(self, observation):
        return 0
