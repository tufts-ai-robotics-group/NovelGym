import gymnasium as gym
import numpy as np
from typing import Optional, List

from abc import (
  ABC,
  abstractmethod,
)

from utils.advanced_item_encoder import PlaceHolderItemEncoder

class ObservationGenerator(ABC):
    def __init__(self, *args, **kwargs):
        self.action_set: List[str] = []
        self.novel_action_set: List[str] = []
        self.failed_action: str = []
        self.item_encoder: PlaceHolderItemEncoder = PlaceHolderItemEncoder()
        pass
    
    @staticmethod
    def get_observation_space(self, *args, **kwargs):
        return gym.spaces.Discrete(1)
    
    @abstractmethod
    def generate_observation(self, json_input: dict) -> np.ndarray:
        return [0]
    
    @abstractmethod
    def check_if_effects_met(self, new_state_json: dict) -> bool:
        return True
