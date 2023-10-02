from typing import Tuple
import gymnasium as gym

from agents.base_planning import BasePlanningAgent
from utils.diarc_json_utils import generate_diarc_json_from_state


REWARDS = {
    "positive": 1000,
    "negative": -250,
    "step": -1,
}


class RapidLearnWrapper(gym.Wrapper):
    """
    An environment where given the pddl domains,
    it will execute the plan until an action failed, or until it's unable to plan.
    then the environment steps will start.

    Failure is denoted by the agent setting its stuck flag to True.

    TODO so far it re-initializes a reward generator since the effect met
    checker is bundled with the obs checker. in the future it should be
    separated out.
    """
    metadata = {"render_modes": ["human"]}
    
    def _execute_plan(self):
        """
        returns true if stuck in an episode.
        """
        # fast forward the environment until the agent is stuck.
        self.env._run_env_agents()
        agent = self.unwrapped.agent_selection
        # while not terminated and not stuck
        while agent in self.unwrapped.terminations and \
              not self.unwrapped.terminations[agent] and \
              not getattr(self.unwrapped.agent_manager.agents[agent].agent, "stuck", False):
            obs, reward, terminated, truncated, info = self.unwrapped.last()
            agent_obj = self.env.agent_manager.agents[agent].agent
            action = agent_obj.policy(obs)
                        # getting the actions
            if type(action) == tuple:
                # symbolic agent sending extra params
                action, _ = action
            else:
                # rl agent / actions with no extra params
                action = action

            # run the action
            obs, reward, terminated, truncated, info = self.env.step(action)
            if not info["success"]:
                agent_obj.set_stuck()
            # run the agents in the environment
            is_stuck = self.env._run_env_agents()
            if not is_stuck:
                if self.unwrapped.render_mode == "human":
                    print("------Episode is finished internally.------")
                return False
            agent = self.unwrapped.agent_selection
        is_stuck = agent in self.unwrapped.terminations and \
                    not self.unwrapped.terminations[agent]
        if not is_stuck:
            print("------Episode is finished internally.------")
        return is_stuck

    def step(self, action):
        # this step function does not call the wrapped step function.
        self.env.step(action)

        # run another step of other agents using the stored policy 
        # until the agent in interest is reached again.
        while True:
            # returns true if the agent gets stuck in the current episode.
            # returns false if the agent goes into the next episode.
            is_stuck = self._execute_plan()

            obs, reward, env_terminated, truncated, info = self.env.last()

            # check if effects met and give the rewards
            plannable_done, truncated, reward = self._gen_reward()

            if not is_stuck:
                # not stuck, in a new episode.
                # if the episode is done, we break the loop
                break
            elif plannable_done and self.skip_epi_when_rl_done:
                # skip whole episode if RL gets us back to the normal state
                # used in training
                break
            elif not plannable_done:
                # if not plannable done, we continue the learning
                break

        # generate the observation
        obs = self._gen_obs()

        # if we want to skip the rest of the symbolic learning when RL reaches
        # the goal to speed up training, we set done to be true when RL is done
        if self.skip_epi_when_rl_done:
            terminated = env_terminated or plannable_done
        else:
            terminated = env_terminated
        return obs, reward, terminated, truncated, {"skipped_epi_count": 0}

    def reset(self, seed=None, options={}):
        _, info = self.env.reset()
        # by this time the env agent have already planned. Now
        # We try to execute the plan until there is any failure.
        skipped_epi_count = int(info["skipped_epi_count"])

        needs_rl = False
        while not needs_rl:
            needs_rl = self._execute_plan()
            if not needs_rl:
                skipped_epi_count += 1
                self.env.reset()
        # get the observation
        self._init_obs_gen()
        obs = self._gen_obs()
        return obs, {"skipped_epi_count": skipped_epi_count}
    
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


    def _gen_reward(self) -> Tuple[bool, bool, float]:
        """
        done, truncated, reward
        """
        # case 1: is done
        if self.unwrapped.internal_state._goal_achieved:
            return True, False, REWARDS["positive"]
        elif self.unwrapped.terminations["agent_0"]:
            return True, False, REWARDS["negative"]
        elif self.unwrapped.truncations["agent_0"]:
            return False, True, REWARDS["step"]
        
        # not done, check if effects met
        main_agent: BasePlanningAgent = self.unwrapped.agent_manager.agents["agent_0"].agent
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
            player_id=self.env.player_id,
            state=self.unwrapped.internal_state,
            dynamic=self.unwrapped.dynamic,
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
