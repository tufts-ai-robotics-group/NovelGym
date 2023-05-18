import torch

import tianshou as ts
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.discrete import Actor, Critic

import numpy as np

from net.basic import BasicCriticNet
from net.norm_net import NormalizedNet
from utils.hint_utils import get_novel_action_indices
from policies import BiasedDQN
from config import POLICIES, POLICY_PROPS


def create_policy(
        rl_algo, 
        state_shape, 
        action_shape, 
        all_actions, 
        novel_actions=[], 
        hidden_sizes=[256, 128, 64],
        buffer=None, 
        lr=None,
        device="cpu",
        checkpoint=None
    ):
    if lr is not None:
        lr = float(lr)
    if hidden_sizes is None:
        hidden_sizes = [256, 128, 64]
    
    PolicyModule = POLICIES[rl_algo]
    policy_props = POLICY_PROPS.get(rl_algo) or {}

    if state_shape[0] < 50 and action_shape < 40:
        net = Net(state_shape, action_shape, hidden_sizes=[128, 64], softmax=True)
    else:
        net = Net(state_shape, action_shape, hidden_sizes=[256, 128, 64], softmax=True)
    optim = torch.optim.Adam(net.parameters(), lr=lr or 1e-4)
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
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr or 1e-4)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr or 1e-4)
        policy = ts.policy.DiscreteSACPolicy(
            actor=net,
            critic1=critic1,
            critic2=critic2,
            actor_optim=optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
        )
    ## imitation learning
    elif rl_algo == "crr":
        net = Net(state_shape, hidden_sizes[0], device=device)
        actor = NormalizedNet(
            hidden_sizes[0],
            action_shape,
            preprocess_net=net,
            hidden_sizes=hidden_sizes[1:],
            device=device
        )
        critic = NormalizedNet(
            hidden_sizes[0],
            action_shape,
            preprocess_net=net,
            hidden_sizes=hidden_sizes[1:],
            device=device,
            output_state=False
        )
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr or 1e-6)
        lr_scheduler = ts.utils.MultipleLRSchedulers()
        policy = ts.policy.DiscreteCRRPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            discount_factor=0.99,
            target_update_freq=320,
            policy_improvement_mode="all"
        ).to(device)
        # lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    elif rl_algo == "gail":
        critic = BasicCriticNet(state_shape, 1)
        disc_net = Net(state_shape + action_shape, 1, hidden_sizes=[256, 128, 64])
        disc_optim = torch.optim.Adam(disc_net.parameters(), lr=lr or 1e-4)
        policy = ts.policy.GAILPolicy(
            actor=net,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
            expert_buffer=buffer,
            disc_net=disc_net,
            disc_optim=disc_optim
        )
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        policy.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
    return policy