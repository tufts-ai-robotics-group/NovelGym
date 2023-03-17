import gymnasium as gym

from gym_novel_gridworlds2.envs.sequential import NovelGridWorldSequentialEnv
from gym_novel_gridworlds2.utils.json_parser import ConfigParser, load_json
from utils.pddl_utils import generate_obj_types

class SAPolycraftRL(gym.Wrapper):
    def __init__(self, config_file_paths, agent_name, task_name=""):
        config_content = load_json(config_json={"extends": config_file_paths})
        self.config_content = config_content


        self.env = NovelGridWorldSequentialEnv(
            config_dict=config_content, 
            max_time_step=None, 
            time_limit=None, 
            run_name=task_name,
            logged_agents=[]
        )
        self.env.dynamic.all_objects = generate_obj_types(self.config_content)
        self.agent_name = agent_name

        self._observation_space = None
        self._action_space = None
        self.episode = 0
        self._fast_forward()


    
    def _fast_forward(self):
        # fast forward the environment until the agent in interest is reached.
        agent = self.env.agent_selection
        while agent != self.agent_name:
            if agent not in self.env.dones or self.env.dones[agent]:
                # skips the process if agent is done.
                self.env.step(0, {})
            else:
                obs, reward, done, info = self.env.last()
                action = self.env.agent_manager.agents[agent].agent.policy(obs)
                            # getting the actions
                extra_params = {}
                if type(action) == tuple:
                    # symbolic agent sending extra params
                    action, extra_params = action
                else:
                    # rl agent / actions with no extra params
                    action = action

                self.env.step(action, extra_params)
            agent = self.env.agent_selection


    def step(self, action):
        # run the agent in interest
        self.env.step(action, {})

        # run another step of other agents using the stored policy 
        # until the agent in interest is reached again.
        self._fast_forward()

        obs, reward, done, info = self.env.last()
        return obs, reward, done, False, info

    def reset(self):
        # reset the environment
        self.episode += 1
        self.env.reset(options={"episode": self.episode})
        self.env.dynamic.all_objects = generate_obj_types(self.config_content)
        
        # fast forward
        self._agent_iter = self.env.agent_iter()
        self._fast_forward()
        obs, reward, done, info = self.env.last()
        return obs

