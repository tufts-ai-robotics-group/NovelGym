
import os
import tianshou as ts
import gymnasium as gym

import torch
import numpy as np
import pickle
from torch.utils.tensorboard import SummaryWriter
from ts_extensions.custom_logger import CustomTensorBoardLogger

from args import parser, NOVELTIES, OBS_TYPES, HINTS, POLICIES, POLICY_PROPS, NOVEL_ACTIONS, OBS_GEN_ARGS, AVAILABLE_ENVS
from utils.hint_utils import get_hinted_actions, get_novel_action_indices, get_hinted_items
from utils.pddl_utils import get_all_actions, KnowledgeBase
from policy_utils import create_policy
from utils.train_utils import set_train_eps, create_save_best_fn, generate_min_rew_stop_fn, create_save_checkpoint_fn

from utils.make_env import make_env

args = parser.parse_args()
seed = args.seed
if seed == None:
    seed = np.random.randint(0, 10000000)
exp_name = args.exp_name
log_path = os.path.join(
    args.logdir, 
    exp_name or "default_exp",
    args.env,
    args.novelty,
    args.obs_type,
    args.rl_algo,
    str(seed)
)


if __name__ == "__main__":
    # novelty
    novelty_name = args.novelty
    novelty_path = NOVELTIES.get(novelty_name)
    config_file_paths = ["config/polycraft_gym_rl_single.yaml"]
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
    if args.num_threads is None:
        if novelty_name == "none":
            num_threads = 8
            max_time_step = 1200
        else:
            num_threads = 4
            max_time_step = 400
    else:
        num_threads = args.num_threads
        max_time_step = 400
    envs = [
        lambda: make_env(
                    env_name=args.env, 
                    config_file_paths=config_file_paths,
                    RepGenerator=RepGenerator,
                    rep_gen_args={
                        "hints": HINTS.get(novelty_name) or "",
                        "hinted_objects": hinted_objects,
                        "novel_objects": [], # TODO
                        "num_reserved_extra_objects": 2 if novelty_name == "none" else 0,
                        "item_encoder_config_path": "config/items.json",
                        **rep_gen_args
                    },
                    max_time_step=max_time_step
                )
        for _ in range(num_threads)
    ]
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
    
    if args.resume:
        checkpoint = os.path.join(log_path, "checkpoint.pth")
    else:
        checkpoint = args.checkpoint
    policy = create_policy(
        args.rl_algo, state_shape, action_shape, 
        all_actions, novel_actions, 
        checkpoint=args.checkpoint, lr=args.lr, 
        hidden_sizes=hidden_sizes,
        device=args.device
    )

    print("----------- Run Info -----------")
    print("using", num_threads, "threads")
    print("Novelty:", novelty_name)
    print("Seed:", seed)
    print("Env:", args.env)
    print("Algorithm:", args.rl_algo)
    print("lr:", args.lr or "default")
    if args.resume:
        print("Resuming training from last run:", log_path)
    if checkpoint is not None:
        print("loaded checkpoint", checkpoint)
    if hidden_sizes:
        print("hidden size:", hidden_sizes)
    print("Device:", args.device)
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
    rew_min = 500 if args.env == "sa_a" else 0
    logger = CustomTensorBoardLogger(writer, epi_max_len=max_time_step, rew_min=rew_min)

    # collector
    if args.resume:
        try:
            path = os.path.join(log_path, "buffer_ckpt.pth")
            train_buffer = ts.data.VectorReplayBuffer.load_hdf5(path)
        except:
            train_buffer = ts.data.VectorReplayBuffer(20000, buffer_num=num_threads)
    else:
        train_buffer = ts.data.VectorReplayBuffer(20000, buffer_num=num_threads)
    train_collector = ts.data.Collector(policy, venv, train_buffer, exploration_noise=True)
    test_collector = ts.data.Collector(policy, venv, exploration_noise=True)
    
    if novelty_name == "none":
        # Training the base pre-novelty model. 
        # To speed up 
        step_per_epoch = 28800
        step_per_collect = 2400
        num_threads = 8
        episode_per_test = 100
        max_epoch = 300
    else:
        step_per_epoch = 4800
        step_per_collect = 800
        num_threads = 4
        episode_per_test = 100
        max_epoch = 100 if args.env == "pf" else 200
    result = ts.trainer.onpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=max_epoch, step_per_epoch=step_per_epoch, step_per_collect=step_per_collect,
        episode_per_test=episode_per_test, batch_size=128,
        repeat_per_collect=8,
        train_fn=set_train_eps if args.rl_algo == "dqn" else None,
        test_fn=(lambda epoch, env_step: policy.set_eps(0.05)) if args.rl_algo == "dqn" else None,
        # stop_fn=generate_stop_fn(length=20, threshold=venv.spec[0].reward_threshold),
        stop_fn=generate_min_rew_stop_fn(min_length=22, min_rew_threshold=950),
        save_best_fn=create_save_best_fn(log_path),
        save_checkpoint_fn=create_save_checkpoint_fn(log_path, policy, train_buffer),
        logger=logger,
        resume_from_log=args.resume
    )
    
    print(f'Finished training! Use {result["duration"]}')

