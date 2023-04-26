from gymnasium.envs.registration import register
from .polycraft_simplified import SAPolycraftRL
from .diarc_env import DiarcRapidLearn

register(
    id='NG2-PolycraftMultiInteract-v0', # use id to pass to gym.make(id)
    entry_point='envs:SAPolycraftRL',
    reward_threshold=980,
    max_episode_steps=300
    # reward_threshold =
)
register(
    id="DiarcSocketSimulated-v0",
    entry_point="envs:DiarcRapidLearn",
    reward_threshold=980
)

__all__ = ["SAPolycraftRL", "DiarcRapidLearn"]
