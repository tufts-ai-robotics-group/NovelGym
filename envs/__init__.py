from gymnasium.envs.registration import register
from .polycraft_simplified import SAPolycraftRL

register(
    id='NG2-PolycraftMultiInteract-v0', # use id to pass to gym.make(id)
    entry_point='envs:SAPolycraftRL',
    reward_threshold=980
    # max_episode_steps =
    # reward_threshold =
)
