from gymnasium.envs.registration import register
from .planning_until_failure import PlanningUntilFailureEnv
from .diarc_env import DiarcRapidLearn
from .single_agent import SingleAgentEnv

register(
    id='Gym-SingleAgent-v0',
    entry_point='envs:SingleAgentEnv',
    reward_threshold=980,
    max_episode_steps=300
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
    id="DiarcSocketSimulated-v0",
    entry_point="envs:DiarcRapidLearn",
    reward_threshold=980
)

__all__ = ["SingleAgentEnv", "DiarcRapidLearn", ""]
