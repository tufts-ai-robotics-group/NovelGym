import gymnasium as gym

from gym_novel_gridworlds2.envs.sequential import NovelGridWorldSequentialEnv
from gym_novel_gridworlds2.utils.json_parser import ConfigParser, load_json
from utils.diarc_json_utils import generate_diarc_json_from_state
from utils.pddl_utils import generate_obj_types, generate_pddl
from obs_convertion import LidarAll

class SAPolycraftRL(gym.Wrapper):
    def __init__(
            self, 
            config_file_paths, 
            agent_name, 
            task_name="", 
            show_action_log=False, 
            RepGenerator=LidarAll,
            rep_gen_args={},
        ):
        config_content = load_json(config_json={"extends": config_file_paths})
        self.config_content = config_content

        self.player_id = 0

        self.env = NovelGridWorldSequentialEnv(
            config_dict=config_content, 
            max_time_step=None, 
            time_limit=None, 
            run_name=task_name,
            logged_agents=['main_1'] if show_action_log else []
        )
        self.env.dynamic.all_objects = generate_obj_types(self.config_content)
        self.agent_name = agent_name

        self.RepGeneratorModule = RepGenerator
        self.rep_gen_args = rep_gen_args
        self.rep_gen = None
        self.items_lidar_disabled = []

        self.episode = 0

    
    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.RepGeneratorModule.get_observation_space(
                self.env.dynamic.all_objects,
                **self.rep_gen_args
            )
        else:
            return self._observation_space
    

    @property
    def action_space(self):
        return self.env.action_space("main_1")

    
    def _fast_forward(self):
        # fast forward the environment until the agent in interest is reached.
        agent = self.env.agent_selection
        while agent != self.agent_name or not getattr(self.env.agent_manager.agents[agent].agent, "stuck", False):
            if len(self.env.dones) == 0:
                # episode is done, restart a new episode.
                print("------Episode is complete without RL.------")
                return False
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
        
        # initialize the observation generator
        self._init_obs_gen()
        return True


    def _init_obs_gen(self):
        """
        Initialize the observation generator.
        """
        main_agent = self.env.agent_manager.agents["agent_0"].agent
        failed_action = main_agent.failed_action
        action_set = self.env.agent_manager.agents['agent_0'].action_set

        diarc_json = generate_diarc_json_from_state(
            player_id=self.player_id,
            state=self.env.internal_state,
            dynamic=self.env.dynamic,
            failed_action=failed_action,
            success=False,
        )
        pddl_domain, pddl_problem = generate_pddl(
            ng2_config=self.config_content,
            state=self.env.internal_state,
            dynamics=self.env.dynamic,
        )
        
        json_input = {
            "state": diarc_json,
            "domain": pddl_domain,
            "plan": main_agent.pddl_plan,
            "novelActions": [],
            "actionSet": [action[0] for action in action_set.actions if action not in ["nop", "give_up"]],
        }
        self.rep_gen = self.RepGeneratorModule(
            json_input=json_input, 
            items_lidar_disabled=self.items_lidar_disabled,
            **self.rep_gen_args
        )
    
    def _gen_obs(self):
        """
        Generate the observation.
        """
        main_agent = self.env.agent_manager.agents["agent_0"].agent
        failed_action = main_agent.failed_action
        diarc_json = generate_diarc_json_from_state(
            player_id=self.player_id,
            state=self.env.internal_state,
            dynamic=self.env.dynamic,
            failed_action=failed_action,
            success=False,
        )
        return self.rep_gen.generate_observation(diarc_json)


    def step(self, action):
        # run the agent in interest
        self.env.step(action, {})

        # run another step of other agents using the stored policy 
        # until the agent in interest is reached again.
        self._fast_forward()

        obs, reward, done, info = self.env.last()
        obs = self._gen_obs()
        return obs, reward, done, False, info

    def reset(self):
        # reset the environment
        needs_rl = False
        while not needs_rl:
            self.episode += 1
            self.env.reset(options={"episode": self.episode})
            self.env.dynamic.all_objects = generate_obj_types(self.config_content)
            
            # fast forward
            self._agent_iter = self.env.agent_iter()

            needs_rl = self._fast_forward()
        obs, reward, done, info = self.env.last()
        obs = self._gen_obs()
        return obs

