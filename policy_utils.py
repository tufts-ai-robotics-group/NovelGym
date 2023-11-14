import torch

import tianshou as ts
from tianshou.utils.net.common import Net, ActorCritic, MLP
from tianshou.utils.net.discrete import Actor, Critic, IntrinsicCuriosityModule

import numpy as np

from net.basic import BasicCriticNet
from net.norm_net import NormalizedNet
from utils.hint_utils import get_novel_action_indices
from policies import BiasedDQN
from config import POLICIES, POLICY_PROPS


def create_policy_for_matrix(
        rl_algo,
        state_space,
        action_space,
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
        
        if rl_algo == "ppo":
            policy = ts.policy.PPOPolicy(
                # TODO
            )



def create_policy(
        rl_algo, 
        state_shape, 
        action_shape, 
        all_actions, 
        novel_actions=[], 
        hidden_sizes=[256, 64],
        buffer=None, 
        lr=None,
        device="cpu",
        checkpoint=None
    ):
    if lr is not None:
        lr = float(lr)
    if hidden_sizes is None:
        hidden_sizes = [256, 64]
    
    PolicyModule = POLICIES[rl_algo]
    policy_props = POLICY_PROPS.get(rl_algo) or {}

    net = Net(state_shape, action_shape, hidden_sizes=[128, 64], softmax=True, device=device)
    optim = torch.optim.Adam(net.parameters(), lr=lr or 1e-4)

    # prepare policy
    if "ppo_shared_net" in rl_algo:
        net = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = Actor(net, action_shape, device=device)
        critic = Critic(net, device=device)
        actor_critic = ActorCritic(net, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr or 5e-5)
        ppo_policy = ts.policy.PPOPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical
        ).to(device)
    elif "ppo" in rl_algo:
        # non-shared net ppo
        critic = BasicCriticNet(state_shape, 1, device=device)
        # net = Net(state_shape, hidden_sizes[0], device=device)
        # actor = Actor(net, action_shape, hidden_sizes=hidden_sizes, softmax_output=True, device=device)
        # critic = Critic(net, hidden_sizes=hidden_sizes, last_size=1, device=device)
        actor_critic = ActorCritic(net, critic).to(device)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr or 1e-5)
        ppo_policy = ts.policy.PPOPolicy(
            actor=net,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
        ).to(device)


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
        policy = ppo_policy
    elif rl_algo == "ppo_shared_net":
        policy = ppo_policy
    elif rl_algo == "icm_ppo" or rl_algo == "icm_ppo_shared_net":
        feature_dim = 16
        lr_scale = 1.
        reward_scale = 0.01
        forward_loss_weight = 0.2

        feature_net = MLP(
            np.prod(state_shape),
            output_dim=feature_dim,
            hidden_sizes=hidden_sizes[:-1],
            device=device
        )
        icm_module = IntrinsicCuriosityModule(
            feature_net=feature_net,
            feature_dim=feature_dim,
            action_dim=np.prod(action_shape),
            hidden_sizes=hidden_sizes[-1:],
            device=device
        ).to(device)
        icm_optim = torch.optim.Adam(icm_module.parameters(), lr=lr or 1e-4)
        policy = ts.policy.ICMPolicy(
            ppo_policy, icm_module, icm_optim, lr_scale, reward_scale,
            forward_loss_weight
        )
    elif rl_algo == "dsac":
        net_a = Net(state_shape, action_shape, hidden_sizes=hidden_sizes[:-1])
        actor = Actor(net_a, action_shape, softmax_output=False)
        a_optim = torch.optim.Adam(actor.parameters(), lr=lr or 1e-4)
        net_c1 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes[:-1])
        net_c2 = Net(state_shape, action_shape, hidden_sizes=hidden_sizes[:-1])
        critic1 = Critic(net_c1, last_size=action_shape)
        critic2 = Critic(net_c2, last_size=action_shape)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr or 1e-4)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr or 1e-4)
        policy = ts.policy.DiscreteSACPolicy(
            actor=net_a,
            critic1=critic1,
            critic2=critic2,
            actor_optim=a_optim,
            critic1_optim=critic1_optim,
            critic2_optim=critic2_optim,
            alpha=0.05,
            # reward_normalization=True
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
    elif rl_algo == "crr_separate_net":
        actor = Net(state_shape, action_shape, hidden_sizes=hidden_sizes, device=device)
        critic = BasicCriticNet(state_shape, action_shape).to(device)
        actor_critic = ActorCritic(actor, critic)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=lr or 1e-6)
        policy = ts.policy.DiscreteCRRPolicy(
            actor=actor,
            critic=critic,
            optim=optim,
            discount_factor=0.99,
            target_update_freq=320,
            policy_improvement_mode="all"
        ).to(device)
    # elif rl_algo == "gail":
    #     critic = BasicCriticNet(state_shape, 1)
    #     disc_net = Net(state_shape + action_shape, 1, hidden_sizes=[256, 128, 64])
    #     disc_optim = torch.optim.Adam(disc_net.parameters(), lr=lr or 1e-4)
    #     policy = ts.policy.GAILPolicy(
    #         actor=net,
    #         critic=critic,
    #         optim=optim,
    #         dist_fn=torch.distributions.Categorical,
    #         expert_buffer=buffer,
    #         disc_net=disc_net,
    #         disc_optim=disc_optim
    #     )
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        policy.load_state_dict(checkpoint["model"])
        policy.optim.load_state_dict(checkpoint["optim"])
    return policy