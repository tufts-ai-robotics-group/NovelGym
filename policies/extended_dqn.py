from tianshou.policy import DQNPolicy
from tianshou.data import Batch

from typing import Union, List
import numpy as np
import torch

class BiasedDQN(DQNPolicy):
    def __init__(
            self, 
            novel_action_indices: List[int],
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.novel_action_indices = novel_action_indices

    def exploration_noise(self, act: Union[np.ndarray, Batch], batch: Batch) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.max_action_num)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            if len(self.novel_action_indices) > 0:
                q[:, self.novel_action_indices] *= 1.44
                q = np.minimum(q, 1)
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act
