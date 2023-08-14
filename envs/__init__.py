from gymnasium.envs.registration import register
from .planning_until_failure import PlanningUntilFailureEnv
from .single_agent import SingleAgentEnv
from .single_agent_rs import SingleAgentRSShorterPlanEnv

register(
    id='Gym-SingleAgent-v0',
    entry_point='envs:SingleAgentEnv',
    reward_threshold=980,
    max_episode_steps=1000
    # reward_threshold =
)
register(
    id='Gym-PlanningUntilFail-v0',
    entry_point='envs:PlanningUntilFailureEnv',
    reward_threshold=980,
    max_episode_steps=300
    # reward_threshold =
)
register(
    id='RewardShapingShorterPlan-v0',
    entry_point='envs:SingleAgentRSShorterPlanEnv',
    reward_threshold=980,
    max_episode_steps=1000
    # reward_threshold =
)

__all__ = ["SingleAgentEnv", "DiarcRapidLearn", ""]
