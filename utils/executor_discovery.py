import tianshou as ts
import gymnasium as gym
from dataclasses import dataclass
import torch
from typing import Type, Mapping


@dataclass
class ExecutorRep:
    """ A representation of an executor. """
    failed_action: str
    executor: ts.policy.BasePolicy
    action_set: tuple
    observation_space: gym.spaces.Space
    episode: int = 0

class ExecutorDiscoverer:
    def __init__(
            self, 
            model_dir: str, 
            Network: Type[torch.nn.Module],
            Policy: Type[ts.policy.BasePolicy],
            policy_hyperparams: dict,
            lr: float = 1e-4,
        ):
        self.executors: Mapping[str, ExecutorRep] = {}
        self.model_dir = model_dir
        self.lr = lr
        self.policy_hyperparams = policy_hyperparams
        self.Network = Network
        self.Policy = Policy

    def _create_executor(self, failed_action, action_set, observation_space) -> ExecutorRep:
        """ Create an executor for the given failed action. """
        action_set_rl = [action for action, _ in action_set.actions if action not in ["nop", "give_up"]]
        action_space = gym.spaces.Discrete(len(action_set_rl))
        model = self.Network(observation_space.get("n") or observation_space.shape, action_space.get("n") or action_set.shape)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        policy = self.Policy(model, optim, action_space, **self.policy_hyperparams)
        executor_rep = ExecutorRep(failed_action, policy, action_set, observation_space, episode=0)
        return executor_rep

    def get_executor(self, failed_action, action_set, observation_space) -> ts.policy.BasePolicy:
        executor_rep = self.executors.get(failed_action)
        if executor_rep is None or executor_rep.action_set != action_set or executor_rep.observation_space != observation_space:
            print("Creating new executor for failed action:", failed_action)
            executor_rep = self._create_executor(failed_action, action_set, observation_space)
            self.executors[failed_action] = executor_rep
        return executor_rep.executor
    
    def save_executors(self):
        for failed_action, executor_rep in self.executors.items():
            torch.save(executor_rep.executor, f"{self.model_dir}/{failed_action}.pth")

