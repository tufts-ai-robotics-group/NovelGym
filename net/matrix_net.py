import torch, numpy as np
from torch import nn

class BasicNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        size_inventory_obs = state_space["inventory"].n
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, 3, [2, 2]),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(1, -1)
        )
        self.inventory_select_module = nn.Linear(in_features=size_inventory_obs + 1, out_features=32)
        self.combined_module = nn.Sequential(
            nn.LazyLinear(out_features=64), nn.ReLU(inplace=True),
            nn.LazyLinear(out_features=32), nn.ReLU(inplace=True),
            nn.LazyLinear(out_features=action_space.n)
        )

    def forward(self, obs, state=None, info={}):
        map = obs["map"]
        inventory_selected_item = np.concatenate((obs["inventory"], obs["selected_item"]))
        map = torch.tensor(map, dtype=float)
        inventory_selected_item = torch.tensor(inventory_selected_item, dtype=float)

        map_processed = self.conv_module(map)
        inventory_processed = self.inventory_select_module(inventory_selected_item)

        logits = self.combined_module(torch.concat((map_processed, inventory_processed)))
        return logits, state
