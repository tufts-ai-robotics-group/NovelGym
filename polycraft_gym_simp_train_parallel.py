
import os
import argparse
from envs.polycraft_simplified import SAPolycraftRL
import tianshou as ts
import gymnasium as gym
from net.basic import BasicNet
import torch
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from obs_convertion import LidarAll, OnlyFacingObs


OBS_TYPES = {
    "lidar_all": LidarAll,
    "only_facing": OnlyFacingObs,
}

NOVELTIES = {
    "mi": "novelties/evaluation1/multi_interact/multi_interact.json",
    "kibt": "novelties/evaluation1/key_inventory_break_tree/key_inventory_break_tree.json",
    "rdb": "novelties/evaluation1/random_drop_break/random_drop_break.json",
    "space_ar": "novelties/evaluation1/space_around_crafting_table/space_around_crafting_table.json",
}

RL_ALGOS = {
    "dqn": ts.policy.DQNPolicy,
}


parser = argparse.ArgumentParser(description="Polycraft Gym Environment")
# parser.add_argument("filename", type=str, nargs='+', help="The path of the config file.", default="polycraft_gym_main.json")
parser.add_argument(
    "--novelty",
    type=str, 
    help="The name of the novelty.", 
    required=False,
    default="mi",
    choices=NOVELTIES.keys()
)
# parser.add_argument(
#     '--rendering',
#     type=str,
#     help="The rendering mode.",
#     required=False,
#     default="human"
# )
parser.add_argument(
    '--seed',
    type=str,
    help="The seed.",
    required=False,
    default=None
)
parser.add_argument(
    '--num_threads',
    type=int,
    help="Number of sub threads used to run the env.",
    required=False,
    default=4
)
parser.add_argument(
    '--logdir',
    type=str,
    help="The directory to save the logs.",
    required=False,
    default="results"
)
parser.add_argument(
    '--obs_type',
    type=str,
    help="Type of observation.",
    required=False,
    default="lidar_all",
    choices=OBS_TYPES.keys()
)
parser.add_argument(
    '--rl_algo',
    type=str,
    help="The algorithm for RL.",
    required=False,
    default="dqn",
    choices=RL_ALGOS.keys()
)


verbose = False

args = parser.parse_args()

def set_train_eps(epoch, env_step):
    max_eps = 0.4
    min_eps = 0.1
    length = 30
    if epoch > length:
        epsilon = min_eps
    else:
        epsilon = max_eps - (max_eps - min_eps) / length * epoch
    return epsilon


if __name__ == "__main__":
    seed = args.seed

    # novelty
    novelty_name = args.novelty
    novelty_path = NOVELTIES[novelty_name]
    config_file_paths = ["config/polycraft_gym_rl_single.json"]
    config_file_paths.append(novelty_path)

    # observation generator
    RepGenerator = OBS_TYPES[args.obs_type]

    # env
    envs = [lambda: gym.make(
        "NG2-PolycraftMultiInteract-v0",
        config_file_paths=config_file_paths,
        agent_name="agent_0",
        task_name="main",
        show_action_log=False,
        RepGenerator=RepGenerator
    ) for _ in range(args.num_threads)]
    # tianshou env
    venv = ts.env.SubprocVectorEnv(envs)

    # net
    state_shape = venv.observation_space[0].shape or venv.observation_space[0].n
    action_shape = venv.action_space[0].shape or venv.action_space[0].n
    net = BasicNet(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-4)
    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.95, estimation_step=3)

    # logging
    log_path = os.path.join(
        args.logdir, 
        args.novelty,
        args.obs_type,
        args.rl_algo
    )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    # collector
    train_collector = ts.data.Collector(policy, venv, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, venv, exploration_noise=True)

    # train_collector.collect(n_step=5000, random=True)
    # print("Done Collecting Experience. Starting Training...")


    # policy.set_eps(0.1)

    # for i in range(int(1e6)):
    #     collect_result = train_collector.collect(n_step=10)

    #     # once if the collected episodes' mean returns reach the threshold,
    #     # or every 1000 steps, we test it on test_collector
    #     if collect_result['rews'].mean() >= env.spec.reward_threshold or i % 1000 == 0:
    #         policy.set_eps(0.05)
    #         result = test_collector.collect(n_episode=100)
    #         print("episode:", i, "  test_reward:", result['rews'].mean())
    #         if result['rews'].mean() >= env.spec.reward_threshold:
    #             print(f'Finished training! Test mean returns: {result["rews"].mean()}')
    #             break
    #         else:
    #             # back to training eps
    #             policy.set_eps(0.1)

    #     # train policy with a sampled batch data from buffer
    #     losses = policy.update(64, train_collector.buffer)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=300, step_per_epoch=1000, step_per_collect=12,
        update_per_step=0.1, episode_per_test=50, batch_size=64,
        train_fn=set_train_eps,
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        # stop_fn=lambda mean_rewards: mean_rewards >= venv.spec[0].reward_threshold,
        logger=logger
    )
    print(f'Finished training! Use {result["duration"]}')
