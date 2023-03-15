from gym_novel_gridworlds2.agents.agent import Agent
from gym.spaces import Discrete
from .base import BaseRLAgent
from utils.diarc_json_utils import generate_diarc_json_from_state
from .rapid_learn_utils.env_utils import Polycraftv2Env
from .rapid_learn_utils.discover_executor import DiscoverExecutor
from .rapid_learn import RapidLearnAgent

class RLTest(RapidLearnAgent):
    def policy(self, observation):
        """
        Returns 0 for replan,
        1-n for actions. (shifted by 1)
        """
        # TODO let 0 be the action to replan
        if self.executor is None:
            self.executor = DiscoverExecutor()
        if self.effects_met[0] or self.effects_met[1]:
            print("Effects met. Replan.")
            res = 0
        else:
            action = self.executor.step_episode(observation)
            reward = self.get_reward(reset=False, success=False)
            self.executor.end_step(reward)
            res = action + 1
        print(f">>>>>>>>> keyboard agent: Agent {self.id} can do these actions:")
        action_names = self.action_set.get_action_names()
        print(
            ">>>>>>>>>> ",
            "0: replan, ",
            ", ".join(
                [f"{index + 1}: {name}" for (index, name) in enumerate(action_names)]
            ),
        )
        print(">>>>>>>>>> machine selected:", res)
        action = input(">>>>>>>>>> Enter your action (in number): ")
        return int(action)
