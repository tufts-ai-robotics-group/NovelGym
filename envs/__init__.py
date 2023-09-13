from gymnasium.envs.registration import register
from .planning_until_failure import PlanningUntilFailureEnv
from .single_agent_standard import SingleAgentWrapper
from .reward_shaping import RewardShapingWrapper

register(
    id='Gym-PlanningUntilFail-v0',
    entry_point='envs:PlanningUntilFailureEnv',
    reward_threshold=980,
    max_episode_steps=300
    # reward_threshold =
)

__all__ = ["SingleAgentWrapper", "RewardShapingWrapper"]
