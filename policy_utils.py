import torch

import tianshou as ts
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic

from net.basic import BasicCriticNet
from utils.hint_utils import get_novel_action_indices
from policies import BiasedDQN
from config import POLICIES, POLICY_PROPS


def create_policy(rl_algo, state_shape, action_shape, all_actions, novel_actions=[]):
    PolicyModule = POLICIES[rl_algo]
    policy_props = POLICY_PROPS.get(rl_algo) or {}

    if state_shape[0] < 50 and action_shape < 40:
        net = Net(state_shape, action_shape, hidden_sizes=[128, 64], softmax=True)
    else:
        net = Net(state_shape, action_shape, hidden_sizes=[256, 128, 64], softmax=True)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    if rl_algo == "dqn":
        policy = PolicyModule(
            model=net, 
            optim=optim, 
            discount_factor=0.99, 
            estimation_step=3,
        )
    elif rl_algo == "novel_boost":
        policy = PolicyModule(
            model=net, 
            optim=optim, 
            discount_factor=0.99, 
            estimation_step=3, 
            novel_action_indices=get_novel_action_indices(all_actions, novel_actions),
            num_actions=action_shape,
            **policy_props
        )
    elif rl_algo == "ppo":
        critic = BasicCriticNet(state_shape, 1)
        policy = ts.policy.PPOPolicy(
            actor=net,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
        )
    elif rl_algo == "dsac":
        net_c1 = Net(state_shape, action_shape, hidden_sizes=[256, 128, 64])
        net_c2 = Net(state_shape, action_shape, hidden_sizes=[256, 128, 64])
        critic1 = Critic(net_c1, last_size=action_shape)
        critic2 = Critic(net_c2, last_size=action_shape)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=1e-4)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=1e-4)
        policy = ts.policy.DiscreteSACPolicy(
            actor=net,
            critic1=critic1,
            critic2=critic2,
            actor_optim=optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
        )
    return policy