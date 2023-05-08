from typing import Tuple
import gymnasium as gym

from gym_novel_gridworlds2.envs.sequential import NovelGridWorldSequentialEnv
from gym_novel_gridworlds2.utils.json_parser import ConfigParser, load_json
from agents.base_planning import BasePlanningAgent
from utils.diarc_json_utils import generate_diarc_json_from_state
from utils.pddl_utils import generate_obj_types, get_entities
from obs_convertion import LidarAll

REWARDS = {
    "positive": 1000,
    "negative": -250,
    "step": -1,
}


class SAPolycraftRL(gym.Wrapper):
    metadata = {"render_modes": ["human"]}
    def __init__(
            self, 
            config_file_paths, 
            agent_name, 
            task_name="", 
            show_action_log=False, 
            RepGenerator=LidarAll,
            rep_gen_args={},
            enable_render=False,
            skip_epi_when_rl_done=True,
            seed=None
        ):
        config_content = load_json(config_json={"extends": config_file_paths}, verbose=False)
        self.config_content = config_content

        self.player_id = 0

        self.env = NovelGridWorldSequentialEnv(
            config_dict=config_content, 
            max_time_step=None, 
            time_limit=None,
            enable_render=enable_render,
            run_name=task_name,
            logged_agents=['main_1'] if show_action_log else []
        )
        self.show_action_log = show_action_log
        self.agent_name = agent_name

        self.RepGeneratorModule = RepGenerator
        self.rep_gen_args = rep_gen_args
        self.rep_gen = None
        self.items_lidar_disabled = []

        self.episode = -1

        if seed is not None:
            self.env.reset(seed=seed)
        self.env.dynamic.all_objects = generate_obj_types(self.config_content)
        self.env.dynamic.all_entities = get_entities(self.config_content)

        self._action_space = None
        self._observation_space = None

        self._skip_epi_when_rl_done = skip_epi_when_rl_done

    
    @property
    def observation_space(self):
        if self._observation_space is None:
            return self.RepGeneratorModule.get_observation_space(
                self.env.dynamic.all_objects,
                self.env.dynamic.all_entities,
                **self.rep_gen_args
            )
        else:
            return self._observation_space
    

    @property
    def action_space(self):
        if self._action_space is None:
            action_set = self.env.agent_manager.agents['agent_0'].action_set
            action_set_rl = [action for action, _ in action_set.actions if action not in ["nop", "give_up"]]
            self._action_space = gym.spaces.Discrete(len(action_set_rl))
        return self._action_space

    
    def _fast_forward(self):
        # fast forward the environment until the agent in interest is reached.
        agent = self.env.agent_selection
        while agent != self.agent_name or \
              not getattr(self.env.agent_manager.agents[agent].agent, "stuck", False):
            if len(self.env.dones) == 0 or (agent == self.agent_name and self.env.dones[agent]):
                # episode is done, restart a new episode.
                if self.env.enable_render:
                    print("------Episode is complete without RL.------")
                return False
            if agent not in self.env.dones or self.env.dones[agent]:
                # skips the process if agent is not the main agent and is done.
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
        
        return True


    def _init_obs_gen(self):
        """
        Initialize the observation generator.
        """
        main_agent: BasePlanningAgent = self.env.agent_manager.agents["agent_0"].agent
        if self.show_action_log:
            main_agent.verbose = True
        failed_action = main_agent.failed_action
        action_set = self.env.agent_manager.agents['agent_0'].action_set

        if type(failed_action) == tuple:
            failed_action = "(" + " ".join(failed_action[1]) + ")"
        
        diarc_json = generate_diarc_json_from_state(
            player_id=self.player_id,
            state=self.env.internal_state,
            dynamic=self.env.dynamic,
            failed_action=failed_action,
            success=False,
        )
        
        json_input = {
            "state": diarc_json,
            "domain": main_agent.pddl_domain,
            "plan": main_agent.pddl_plan,
            "novelActions": [],
            "actionSet": [action[0] for action in action_set.actions if action not in ["nop", "give_up"]],
        }
        self.rep_gen = self.RepGeneratorModule(
            json_input=json_input, 
            items_lidar_disabled=self.items_lidar_disabled,
            RL_test=True,
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


    def _gen_reward(self) -> Tuple[bool, bool, float]:
        """
        done, truncated, reward
        """
        # case 1: is done
        if self.env.internal_state._goal_achieved:
            return True, False, REWARDS["positive"]
        elif self.env.internal_state._given_up:
            return True, False, REWARDS["negative"]
        elif self.env.dones["agent_0"]:
            return False, True, REWARDS["step"]
        
        # not done, check if effects met
        main_agent: BasePlanningAgent = self.env.agent_manager.agents["agent_0"].agent
        failed_action = main_agent.failed_action

        # case 2: unplannable mode, replan straight away
        if failed_action == "cannotplan":
            plan_found = main_agent.plan()
            if plan_found:
                # case 2.1, plan found. give positive reward and quit
                return True, False, REWARDS['positive']
            else:
                return False, False, REWARDS['step']


        # case 3: failed action mode. firstly check if effects met, then replan and assign rewards
        diarc_json = generate_diarc_json_from_state(
            player_id=self.player_id,
            state=self.env.internal_state,
            dynamic=self.env.dynamic,
            failed_action=failed_action,
            success=False,
        )
        effects_met = self.rep_gen.check_if_effects_met(diarc_json)
        # case 3.1: effects not met, return step reward and continue
        if not (effects_met[0] or effects_met[1]):
            return False, False, REWARDS['step']
        else:
            plan_found = main_agent.plan()
            if plan_found:
                # case 3.2, effects met, plannable
                return True, False, REWARDS['positive']
            else:
                # case 3.3, effects met, unplannable
                return True, False, REWARDS['negative']

    def step(self, action):
        # run the agent in interest
        self.env.step(action, {})

        # run another step of other agents using the stored policy 
        # until the agent in interest is reached again.
        self._fast_forward()

        obs, reward, env_done, info = self.env.last()

        # check if effects met and give the rewards
        plannable_done, truncated, reward = self._gen_reward()

        # generate the observation
        obs = self._gen_obs()

        # if we want to skip the rest of the symbolic learning when RL reaches
        # the goal to speed up training, we set done to be true when RL is done
        if self._skip_epi_when_rl_done:
            done = env_done or plannable_done
        else:
            done = env_done
        return obs, reward, done, truncated, info

    def seed(self, seed=None):
        self.env.reset(seed=seed)
        self.env.dynamic.all_objects = generate_obj_types(self.config_content)
        self.env.dynamic.all_entities = get_entities(self.config_content)

    def reset(self, seed=None, options={}):
        # reset the environment
        needs_rl = False
        main_agent = self.env.agent_manager.agents["agent_0"].agent
        main_agent._reset()
        if self.show_action_log:
            main_agent.verbose = True


        while not needs_rl:
            self.episode += 1
            self.env.reset(seed=seed, options={"episode": self.episode})
            self.env.dynamic.all_objects = generate_obj_types(self.config_content)
            self.env.dynamic.all_entities = get_entities(self.config_content)
            
            # fast forward
            self._agent_iter = self.env.agent_iter()

            needs_rl = self._fast_forward()
        obs, reward, done, info = self.env.last()
        # info = {
        #     "pddl_domain": getattr(self, "pddl_domain", ""),
        #     "pddl_problem": getattr(self, "pddl_problem", ""),
        #     "pddl_plan": getattr(main_agent, "pddl_plan", ""),
        #     **info
        # }

        # initialize the observation generator
        self._init_obs_gen()

        # get the observation
        obs = self._gen_obs()
        return obs, {}

