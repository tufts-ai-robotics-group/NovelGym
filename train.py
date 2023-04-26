
import os
import argparse
from envs.polycraft_simplified import SAPolycraftRL
from shutil import rmtree
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet
import torch
from torch.utils.tensorboard import SummaryWriter
from ts_extensions.custom_logger import CustomTensorBoardLogger

from args import args, NOVELTIES, OBS_TYPES, HINTS, POLICIES, POLICY_PROPS, NOVEL_ACTIONS, OBS_GEN_ARGS
from policies import BiasedDQN
from utils.hint_utils import get_hinted_actions, get_novel_action_indices, get_hinted_items
from utils.pddl_utils import get_all_actions, KnowledgeBase


def set_train_eps(epoch, env_step):
    max_eps = 0.4
    min_eps = 0.1
    if epoch > 10:
        return min_eps
    else:
        return max_eps - (max_eps - min_eps) / 10 * epoch

if __name__ == "__main__":
    seed = args.seed

    # novelty
    novelty_name = args.novelty
    novelty_path = NOVELTIES[novelty_name]
    config_file_paths = ["config/polycraft_gym_rl_single.json"]
    config_file_paths.append(novelty_path)

    # object list
    kb_tmp = KnowledgeBase(config=config_file_paths)
    all_objects = kb_tmp.get_all_objects()
    hinted_objects = get_hinted_items(all_objects, HINTS[novelty_name], True)

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
            "hints": HINTS[novelty_name],
            "hinted_objects": hinted_objects,
            "novel_objects": [], # TODO
            **rep_gen_args
        }
    ) for _ in range(args.num_threads)]
    # tianshou env
    venv = ts.env.SubprocVectorEnv(envs)

    hints = str(HINTS.get(args.novelty))
    novel_actions = (NOVEL_ACTIONS.get(args.novelty) or []) + get_hinted_actions(all_actions, hints, True)

    PolicyModule = POLICIES[args.rl_algo]
    policy_props = POLICY_PROPS.get(args.rl_algo) or {}

    # net
    state_shape = venv.observation_space[0].shape or venv.observation_space[0].n
    action_shape = venv.action_space[0].shape or venv.action_space[0].n
    net = BasicNet(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    if args.rl_algo == "dqn":
        policy = PolicyModule(
            model=net, 
            optim=optim, 
            discount_factor=0.99, 
            estimation_step=3,
        )
    else:
        policy = PolicyModule(
            model=net, 
            optim=optim, 
            discount_factor=0.99, 
            estimation_step=3, 
            novel_action_indices=get_novel_action_indices(all_actions, novel_actions),
            num_actions=action_shape,
            **policy_props
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
    print("--------------------------------")
    print()

    # logging
    log_path = os.path.join(
        args.logdir, 
        args.novelty,
        args.obs_type,
        args.rl_algo
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = CustomTensorBoardLogger(writer)

    # collector
    train_collector = ts.data.Collector(policy, venv, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, venv, exploration_noise=True)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=100, step_per_epoch=1000, step_per_collect=12,
        update_per_step=0.1, episode_per_test=100, batch_size=64,
        train_fn=set_train_eps,
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        stop_fn=lambda mean_rewards: mean_rewards >= venv.spec[0].reward_threshold,
        logger=logger
    )
    print(f'Finished training! Use {result["duration"]}')

