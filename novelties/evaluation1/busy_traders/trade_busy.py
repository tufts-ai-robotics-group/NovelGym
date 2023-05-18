from gym_novel_gridworlds2.actions.action import PreconditionNotMetError
from gym_novel_gridworlds2.contrib.polycraft.actions import Trade

import numpy as np

class BusyTrade(Trade):
    def __init__(self, busy_ratio=0, *args, **kwargs):
        self.busy_ratio = busy_ratio
        super().__init__(*args, **kwargs)

    def do_action(self, agent_entity, target_type=None, target_object=None, **kwargs):
        threashold = self.dynamics.rng.uniform(0, 1)
        if threashold < self.busy_ratio:
            raise PreconditionNotMetError("Trader is busy. Please try again later.")
        return super().do_action(agent_entity, target_type, target_object, **kwargs)
