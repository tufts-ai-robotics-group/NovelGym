from gym_novel_gridworlds2.agents.agent import Agent
from gym.spaces import Discrete
from .base import BaseRLAgent
from utils.diarc_json_utils import generate_diarc_json_from_state
from .rapid_learn_utils.env_utils import Polycraftv2Env
from .rapid_learn_utils.discover_executor import DiscoverExecutor

class RapidLearnAgent(BaseRLAgent):
    def __init__(self, reward_dict=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed_action = None
        self.new_episode = False
        self.env = None
        self.executor = None
        self.reward_dict = reward_dict or {
            "positive": 1000,
            "step": 1,
            "negative": -250
        }
    
    def get_observation_space(self, map_size: tuple, other_size: int):
        if self.env is None:
            return Discrete(1)
        else:
            return self.env.observation_space
    
    def get_observation(self, state, dynamic):
        self.dynamic = dynamic
        self.state = state
        if self.env is None:
            return [0]
        else:
            obs_json = generate_diarc_json_from_state(self.id, state, self.dynamic, self.failed_action, False)
            self.effects_met = self.env.check_if_effects_met(obs_json)
            return self.env.generate_observation(obs_json)
    
    def update_metadata(self, metadata: dict):
        gameover = metadata["gameOver"]
        goal_achieved = metadata["goal"]["goalAchieved"]
        if gameover:
            self.end_episode(success=goal_achieved)
        elif metadata["command_result"]["command"] == "replan":
            self.end_episode(success=metadata["command_result"]["result"].upper() == "SUCCESS")


    def get_reward(self, reset, success):
        if reset:
            if success:
                return self.reward_dict["positive"]
            else:
                return self.reward_dict["negative"]
        return self.reward_dict["step"]


    def end_episode(self, success):
        if self.executor is not None:
            self.executor.end_episode(self.get_reward(reset=True, success=success), success=success)
            self.executor = None
            self.effects_met = (False, False)
            self.env = None


    def init_rl(self, failed_action, pddl_domain, pddl_plan):
        self.failed_action = failed_action
        self.pddl_domain = pddl_domain
        init_dict = {
            "state": generate_diarc_json_from_state(self.id, self.state, self.dynamic, self.failed_action, True),
            "domain": self.pddl_domain,
            "plan": pddl_plan,
            "novelActions": [],
            "actionSet": [action[0] for action in self.action_set.actions],
        }
        self.env = Polycraftv2Env(init_dict, RL_test=True)
        self.executor = DiscoverExecutor(**self.env.init_info())
        # self.executor.reward_generator.RL_test = True


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
            return 0
        else:
            action = self.executor.step_episode(observation)
            reward = self.get_reward(reset=False, success=False)
            self.executor.end_step(reward)
            return action + 1
