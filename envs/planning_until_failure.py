from typing import Tuple
import gymnasium as gym

from .single_agent import SingleAgentEnv

REWARDS = {
    "positive": 1000,
    "negative": -250,
    "step": -1,
}


class PlanningUntilFailureEnv(SingleAgentEnv):
    """
    An environment where given the pddl domains,
    it will execute the plan until an action failed, or until it's unable to plan.
    then the environment steps will start.

    Failure is denoted by the agent setting its stuck flag to True.
    """
    metadata = {"render_modes": ["human"]}
    def _fast_forward(self):
        # fast forward the environment until the agent in interest is reached.
        agent = self.env.agent_selection
        while agent != self.agent_name or \
              not getattr(self.env.agent_manager.agents[agent].agent, "stuck", False):
            if len(self.env.dones) == 0 or (agent == self.agent_name and self.env.dones[agent]):
                # episode is done, restart a new episode.
                if self.env.enable_render:
                    print("------Episode is finished internally.------")
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
