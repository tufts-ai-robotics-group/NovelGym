
import os
import argparse
from envs.polycraft_simplified import SAPolycraftRL
from shutil import rmtree
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet, BasicCriticNet
from net.basic_small import BasicNetSmall
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic
import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
from ts_extensions.custom_logger import CustomTensorBoardLogger

from args import args, NOVELTIES, OBS_TYPES, HINTS, POLICIES, POLICY_PROPS, NOVEL_ACTIONS, OBS_GEN_ARGS
from policies import BiasedDQN
from utils.hint_utils import get_hinted_actions, get_novel_action_indices, get_hinted_items
from utils.pddl_utils import get_all_actions, KnowledgeBase


def set_train_eps(epoch, env_step):
    max_eps = 0.2
    min_eps = 0.05
    if epoch > 20:
        return min_eps
    else:
        return max_eps - (max_eps - min_eps) / 10 * epoch

def generate_stop_fn(length, threshold):
    result_hist = [0] * length
    result_index = 0
    sum_result = 0
    def stop_fn(mean_reward):
        nonlocal sum_result
        nonlocal result_index
        sum_result -= result_hist[result_index]
        result_hist[result_index] = mean_reward
        result_index = (result_index + 1) % len(result_hist)
        sum_result += mean_reward
        return sum_result / len(result_hist) >= threshold
    return stop_fn

def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    ckpt_path = os.path.join(log_path, "checkpoint.pth")
    # Example: saving by epoch num
    # ckpt_path = os.path.join(log_path, f"checkpoint_{epoch}.pth")
    torch.save(
        {
            "model": policy.state_dict(),
            "optim": optim.state_dict(),
        }, ckpt_path
    )
    buffer_path = os.path.join(log_path, "train_buffer.pkl")
    pickle.dump(train_collector.buffer, open(buffer_path, "wb"))
    return ckpt_path


if __name__ == "__main__":
    seed = args.seed
    if seed == None:
        seed = np.random.randint(0, 10000000)

    # novelty
    novelty_name = args.novelty
    novelty_path = NOVELTIES[novelty_name]
    config_file_paths = ["config/polycraft_gym_rl_single.json"]
    config_file_paths.append(novelty_path)

    # object list
    kb_tmp = KnowledgeBase(config=config_file_paths)
    all_objects = kb_tmp.get_all_objects()
    hinted_objects = get_hinted_items(all_objects, HINTS.get(novelty_name) or "", True)

    # action list
    all_actions = get_all_actions(config_file_paths)

    # observation generator
    RepGenerator = OBS_TYPES[args.obs_type]
    rep_gen_args = OBS_GEN_ARGS.get(args.obs_type, {})

    # env
    envs = [lambda: gym.make(
        "NG2-PolycraftMultiInteract-v0",
        config_file_paths=config_file_paths,
        agent_name="agent_0",
        task_name="main",
        show_action_log=False,
        RepGenerator=RepGenerator,
        rep_gen_args={
            "hints": HINTS.get(novelty_name) or "",
            "hinted_objects": hinted_objects,
            "novel_objects": [], # TODO
            **rep_gen_args
        }
    ) for _ in range(args.num_threads)]
    # tianshou env
    venv = ts.env.SubprocVectorEnv(envs)
    if seed is not None:
        torch.manual_seed(seed)
        venv.seed(seed=[seed + i * 112 for i in range(args.num_threads)])

    hints = str(HINTS.get(args.novelty))
    novel_actions = (NOVEL_ACTIONS.get(args.novelty) or []) + get_hinted_actions(all_actions, hints, True)

    PolicyModule = POLICIES[args.rl_algo]
    policy_props = POLICY_PROPS.get(args.rl_algo) or {}

    # net
    state_shape = venv.observation_space[0].shape or venv.observation_space[0].n
    action_shape = venv.action_space[0].shape or venv.action_space[0].n
    if state_shape[0] < 50 and action_shape < 40:
        net = Net(state_shape, action_shape, hidden_sizes=[128, 64], softmax=True)
    else:
        net = Net(state_shape, action_shape, hidden_sizes=[256, 128, 64], softmax=True)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    if args.rl_algo == "dqn":
        policy = PolicyModule(
            model=net, 
            optim=optim, 
            discount_factor=0.99, 
            estimation_step=3,
        )
    elif args.rl_algo == "novel_boost":
        policy = PolicyModule(
            model=net, 
            optim=optim, 
            discount_factor=0.99, 
            estimation_step=3, 
            novel_action_indices=get_novel_action_indices(all_actions, novel_actions),
            num_actions=action_shape,
            **policy_props
        )
    elif args.rl_algo == "ppo":
        critic = BasicCriticNet(state_shape, 1)
        policy = ts.policy.PPOPolicy(
            actor=net,
            critic=critic,
            optim=optim,
            dist_fn=torch.distributions.Categorical,
        )
    elif args.rl_algo == "dsac":
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

    print("----------- metadata -----------")
    print("Novelty:", novelty_name)
    print("Algorithm:", args.rl_algo)
    print("Observation type:", args.obs_type)
    print("hints:", hints)
    print()
    print("Novel actions: ", novel_actions)
    print("Hinted Objects: ", hinted_objects)
    print("State space: ", state_shape)
    print("Action space: ", action_shape)
    print("Network:", net.__class__.__name__)
    print("--------------------------------")
    print()
    if args.metadata:
        exit(0)

    # logging
    exp_name = args.exp_name
    log_path = os.path.join(
        args.logdir, 
        exp_name or "default_exp",
        args.novelty,
        args.obs_type,
        args.rl_algo,
        str(seed)
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = CustomTensorBoardLogger(writer)

    # collector
    train_collector = ts.data.Collector(policy, venv, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, venv, exploration_noise=True)

    result = ts.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=300, step_per_epoch=1200, step_per_collect=1200,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        repeat_per_collect=1,
        train_fn=set_train_eps if args.rl_algo == "dqn" else None,
        test_fn=(lambda epoch, env_step: policy.set_eps(0.05)) if args.rl_algo == "dqn" else None,
        stop_fn=generate_stop_fn(length=20, threshold=venv.spec[0].reward_threshold),
        logger=logger
    )
    
    print(f'Finished training! Use {result["duration"]}')

