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


class SingleAgentWrapper(gym.Wrapper):
    """
    An environment where given the pddl domains,
    it will execute the plan until an action failed, or until it's unable to plan.
    then the environment steps will start.
    """
    metadata = {"render_modes": ["human"]}
    def __init__(
            self, 
            base_env: NovelGridWorldSequentialEnv,
            agent_name, 
            task_name="", 
            show_action_log=False, 
            RepGenerator=LidarAll,
            rep_gen_args={},
            skip_epi_when_rl_done=True,
            seed=None
        ):
        self.player_id = 0

        self.env = base_env
        self.show_action_log = show_action_log
        self.agent_name = agent_name

        self.RepGeneratorModule = RepGenerator
        self.rep_gen_args = rep_gen_args
        self.rep_gen = None
        self.items_lidar_disabled = []

        self.episode = -1

        if seed is not None:
            self.env.reset(seed=seed)
        self.env.dynamic.all_objects = generate_obj_types(self.env.config_dict)
        self.env.dynamic.all_entities = get_entities(self.env.config_dict)

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

    
    def _run_env_agents(self):
        # fast forward the environment until the agent in interest is reached.
        agent = self.env.agent_selection
        while agent != self.agent_name:
            if agent not in self.env.terminations or \
                    (agent == self.agent_name and (self.env.terminations[agent] or 
                                                   self.env.truncations[agent])):
                # episode is done, restart a new episode.
                if self.env.render_mode == "human":
                    print("------Episode is finished internally.------")
                return False
            # TODO: remove extra params
            obs, reward, terminated, truncated, info = self.env.last()
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


    def _gen_reward(self):
        """
        env_done, reward
        """
        if self.env.internal_state._goal_achieved:
            return True, REWARDS["positive"]
        elif self.env.internal_state._given_up:
            return True, REWARDS["negative"]
        else:
            return False, REWARDS["step"]
        

    def step(self, action):
        # run the agent in interest
        self.env.step(action, {})

        # run another step of other agents using the stored policy 
        # until the agent in interest is reached again.
        needs_rl = self._run_env_agents()

        obs, reward, env_terminated, truncated, info = self.env.last()

        # check if effects met and give the rewards
        plannable_done, reward = self._gen_reward()

        # generate the observation
        obs = self._gen_obs()

        # if we want to skip the rest of the symbolic learning when RL reaches
        # the goal to speed up training, we set done to be true when RL is done
        if self._skip_epi_when_rl_done:
            terminated = env_terminated or plannable_done
        else:
            terminated = env_terminated
        return obs, reward, terminated, truncated, {"skipped_epi_count": 0, **info}

    def seed(self, seed=None):
        self.env.reset(seed=seed)
        self.env.dynamic.all_objects = generate_obj_types(self.env.config_dict)
        self.env.dynamic.all_entities = get_entities(self.env.config_dict)

    def reset(self, seed=None, options={}):
        if options is None:
            options = {}
        # print("reset")
        # reset the environment
        needs_rl = False
        main_agent = self.env.agent_manager.agents[self.agent_name].agent
        main_agent._reset()
        if self.show_action_log:
            main_agent.verbose = True


        skipped_epi_count = 0
        while not needs_rl:
            self.episode += 1
            self.env.reset(seed=seed, options={"episode": self.episode, **options})
            self.env.dynamic.all_objects = generate_obj_types(self.env.config_dict)
            self.env.dynamic.all_entities = get_entities(self.env.config_dict)
            
            # fast forward
            self._agent_iter = self.env.agent_iter()

            needs_rl = self._run_env_agents()
            if not needs_rl:
                skipped_epi_count += 1
        obs, reward, terminated, truncated, info = self.env.last()
        # plan the main agent so utils can be used
        main_agent.plan()

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
        return obs, {"skipped_epi_count": skipped_epi_count, **info}

