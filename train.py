
import os
import tianshou as ts
import gymnasium as gym

from envs.planning_until_failure import SingleAgentEnv

import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
from ts_extensions.custom_logger import CustomTensorBoardLogger

from args import parser, NOVELTIES, OBS_TYPES, HINTS, POLICIES, POLICY_PROPS, NOVEL_ACTIONS, OBS_GEN_ARGS, AVAILABLE_ENVS
from utils.hint_utils import get_hinted_actions, get_novel_action_indices, get_hinted_items
from utils.pddl_utils import get_all_actions, KnowledgeBase
from policy_utils import create_policy
from utils.train_utils import set_train_eps, create_save_best_fn, generate_stop_fn, create_save_checkpoint_fn

args = parser.parse_args()
seed = args.seed
if seed == None:
    seed = np.random.randint(0, 10000000)
exp_name = args.exp_name
log_path = os.path.join(
    args.logdir, 
    exp_name or "default_exp",
    args.novelty,
    args.obs_type,
    args.rl_algo,
    str(seed)
)

def set_train_eps(epoch, env_step):
    max_eps = 0.2
    min_eps = 0.05
    if epoch > 20:
        return min_eps
    else:
        return max_eps - (max_eps - min_eps) / 10 * epoch

def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, "best_policy.pth"))

def generate_stop_fn(length, threshold):
    """
    Generates a stop function that takes a running mean of the last `length` 
    rewards and returns True if the mean is better than `threshold`.
    """
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
            "optim": policy.optim.state_dict(),
        }, ckpt_path
    )
    return ckpt_path


if __name__ == "__main__":
    # novelty
    novelty_name = args.novelty
    novelty_path = NOVELTIES.get(novelty_name)
    config_file_paths = ["config/polycraft_gym_rl_single.json"]
    if novelty_name is not None and novelty_name != "none":
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
    env_name = AVAILABLE_ENVS[args.env]
    envs = [lambda: gym.make(
        env_name,
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

    hints = str(HINTS.get(args.novelty))
    novel_actions = (NOVEL_ACTIONS.get(args.novelty) or []) + get_hinted_actions(all_actions, hints, True)

    # net
    state_shape = venv.observation_space[0].shape or venv.observation_space[0].n
    action_shape = venv.action_space[0].shape or venv.action_space[0].n

    if args.hidden_sizes is not None:
        hidden_sizes = [int(x) for x in args.hidden_sizes.split(",")]
    else:
        hidden_sizes = None
    
    policy = create_policy(
        args.rl_algo, state_shape, action_shape, 
        all_actions, novel_actions, 
        checkpoint=args.checkpoint, lr=args.lr, 
        hidden_sizes=hidden_sizes
    )

    print("----------- metadata -----------")
    print("using", args.num_threads, "threads")
    print("Novelty:", novelty_name)
    print("Seed:", seed)
    print("Algorithm:", args.rl_algo)
    print("lr:", args.lr or "default")
    if args.checkpoint:
        print("loaded checkpoint", args.checkpoint)
    if hidden_sizes:
        print("hidden size:", hidden_sizes)
    print("Observation type:", args.obs_type)
    print("hints:", hints)
    print()
    print("Novel actions: ", novel_actions)
    print("Hinted Objects: ", hinted_objects)
    print("State space: ", state_shape)
    print("Action space: ", action_shape)
    print("--------------------------------")
    print()
    if args.metadata:
        exit(0)

    # logging
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = CustomTensorBoardLogger(writer)

    # collector
    train_collector = ts.data.Collector(policy, venv, ts.data.VectorReplayBuffer(20000, buffer_num=args.num_threads), exploration_noise=True)
    test_collector = ts.data.Collector(policy, venv, exploration_noise=True)

    result = ts.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=300, step_per_epoch=1200, step_per_collect=1200,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        repeat_per_collect=4,
        train_fn=set_train_eps if args.rl_algo == "dqn" else None,
        test_fn=(lambda epoch, env_step: policy.set_eps(0.05)) if args.rl_algo == "dqn" else None,
        # stop_fn=generate_stop_fn(length=20, threshold=venv.spec[0].reward_threshold),
        stop_fn=lambda mean_rewards: False,
        save_best_fn=create_save_best_fn(log_path),
        save_checkpoint_fn=create_save_checkpoint_fn(log_path, policy),
        logger=logger
    )
    
    print(f'Finished training! Use {result["duration"]}')

