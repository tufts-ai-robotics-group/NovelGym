from gym_novel_gridworlds2.agents.agent import Agent
from gym.spaces import Discrete


class BaseRLAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.failed_action = None
    
    def get_observation_space(self, map_size: tuple, other_size: int):
        return Discrete(1)
    
    def get_observation(self, state, dynamic):
        # raise NotImplementedError("Get observation for " + self.name + " is not implemented.")
        return [0]

    def report_failed_action(self, action: str):
        self.failed_action = action

    def policy(self, observation):
        print(observation)
        print(f">>>>>>>>> keyboard agent: Agent {self.id} can do these actions:")
        action_names = self.action_set.get_action_names()
        print(
            ">>>>>>>>>> ",
            ", ".join(
                [f"{index}: {name}" for (index, name) in enumerate(action_names)]
            ),
        )
        action = input(">>>>>>>>>> Enter your action (in number): ")
        return int(action)
