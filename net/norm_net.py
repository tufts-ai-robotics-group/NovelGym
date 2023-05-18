import torch, numpy as np
from torch import nn

class NormalizedNet(nn.Module):
    def __init__(self, state_shape, action_shape, preprocess_net=None, hidden_sizes=[256, 128, 64], device="cpu", output_state=True):
        super().__init__()
        input_size = np.prod(state_shape)
        models = []
        self.preprocess_net = preprocess_net
        models.append(nn.BatchNorm1d(input_size))
        for output_size in hidden_sizes:
            models.append(nn.Linear(input_size, output_size))
            models.append(nn.ReLU(inplace=True))
            models.append(nn.BatchNorm1d(output_size))
            input_size = output_size
        models.append(nn.Linear(input_size, np.prod(action_shape)))
        
        self.device = device
        self.model = nn.Sequential(*models).to(device)
        self.output_state = output_state

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        obs = obs.to(self.device)
    
        if self.preprocess_net is not None:
            logits, state = self.preprocess_net(obs, state)
        else:
            batch = obs.shape[0]
            logits = obs.view(batch, -1)
        logits = self.model(logits)
        if self.output_state:
            return logits, state
        else:
            return logits
