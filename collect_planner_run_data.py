from tianshou.data import Batch, ReplayBuffer
import tianshou as ts
from multiprocessing import Manager, Pool, Queue, Process
from multiprocessing.connection import wait
from tqdm import tqdm
from gym_novel_gridworlds2.agents.agent import Agent
import gymnasium as gym
import numpy as np
import time

from args import parser 
from config import NOVELTIES
from envs import SingleAgentEnv

args = parser.parse_args()

num_extra_items = 4
TOTAL = 100000
MAX_STEPS_PER_EPISODE = 1000
EXPECTED_STEPS_PER_EPI = 50

filename = "results/{}/planner_buffer.hdf5".format(args.exp_name)

# novelty_name = args.novelty
# novelty_path = NOVELTIES[novelty_name]
config_file_paths = ["config/polycraft_gym_rl_single.json"]
# config_file_paths.append(novelty_path)

run_count = int(TOTAL / args.num_threads)

seed = args.seed or 0

def run_collection(i, progress_q: Queue, result_q: Queue):
    env = gym.make(
        "Gym-SingleAgent-v0",
        config_file_paths=config_file_paths,
        agent_name="agent_0",
        task_name="main",
        show_action_log=False,
        enable_render=False,
        seed=i + seed
    )
    env.seed(i + seed)

    # planning agent
    agent: Agent = env.env.agent_manager.agents["agent_0"].agent

    buffer = ReplayBuffer(run_count * 100)

    for epi in range(run_count):
        obs, info = env.reset()
        
        for _ in range(MAX_STEPS_PER_EPISODE):
            act = agent.policy(obs)
            obs, rew, terminated, truncated, info = env.step(act)
            buffer.add(Batch(obs=obs, act=act, rew=rew, terminated=terminated, truncated=truncated, info=info))

            if terminated or truncated:
                break
        progress_q.put((i, epi))
    result_q.put((i, buffer))

def test_parallel(i, q: Queue):
    for epi in range(run_count):
        time.sleep(0.01)
        q.put((i, epi))


if __name__ == "__main__":
    num_threads = args.num_threads
    # buffers = process_map(run_collection, range(0, 100000), max_workers=num_threads)
    with tqdm(total=TOTAL) as pbar:
        with Manager() as manager:
            progress_arr = [0] * num_threads
            progress_q = manager.Queue()
            result_q = manager.Queue()
            processes = [Process(target=run_collection, args=(i, progress_q, result_q)) for i in range(num_threads)]
            for p in processes:
                p.start()
            while any(p.is_alive() for p in processes):
                while not progress_q.empty():
                    i, progress = progress_q.get()
                    progress_arr[i] = progress
                    if np.sum(progress_arr) >= TOTAL - num_threads:
                        break
                    pbar.update()
            for p in processes:
                p.join()

            full_buffer = ReplayBuffer(TOTAL * EXPECTED_STEPS_PER_EPI)
            while not result_q.empty():
                i, buffer = result_q.get()
                full_buffer.update(buffer)
            pbar.update()
            full_buffer.save_hdf5(filename, compression="gzip")
    print("Done! Saved to", filename)
