# NovelGym

This is a wrapper on [NovelGridWorldsV2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2), itself a redesigned version of [gym-novel-gridworlds](https://github.com/gtatiya/gym-novel-gridworlds), which

> are [OpenAI Gym](https://github.com/openai/gym) environments for developing and evaluating AI agents that can detect and adapt to unknown sudden novelties in their environments. In each environment, the agent needs to craft objects using multiple recipes, which requires performing certain steps in some sequence.

For more details on [NovelGridWorldsV2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2), see the repository.

## Installation

### Install NovelGridWorldsV2

First clone the [NovelGridWorldsV2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2) repository, activate your Python environment such as `venv` or `conda` (if applicable), and install dependencies using `pip`.

```
git clone https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2.git
cd NovelGridWorldsV2
pip install .
```

### Install NovelGym

Next, clone this repository, keeping the same virtual environment activated (if applicable).

```
cd ..
git clone https://github.com/tufts-ai-robotics-group/NovelGym.git
```

### Compile MetricFF

Finally, install your C compiler by going to `planners/Metric-FF-v2.1`, and running `make`, as follows.

```
cd planners/Metric-FF-v2.1
make
```

## Basic Usage

### Project Structure

The project consists of an environment, taken from [NovelGridWorldsV2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2), and a single-agent training architecture. See the [documentation](https://clarech712.github.io/ng-website/) for a detailed explanation of the project structure.

### Online Training

A single agent can be trained by running `train.py`. To see the full list of usage conventions and optional arguments, run the below. The results of the run will be saved in the [results](results) folder.

```
python train.py -h
```

An example command would be the below, where we train an RapidLearn<sup>+</sup>(PPO+ICM) agent with an LiDAR-based observation space on a task. We do not test the agent on a novelty.

```
python train.py --novelty none --seed 1 --obs_type lidar_all --env sa --rl_algo icm_ppo
```

For specifics of the task being trained for and the details of the available agents, see the [documentation](https://clarech712.github.io/ng-website/).

### Testing Novelties

In order to train the same agent as above and test it on the built-in fence novelty, run the following.

```
python train.py --novelty fence --seed 1 --obs_type lidar_all --env sa --rl_algo icm_ppo
```

All the command line options of `train.py` can be obtained by running `python train.py --help`.

```
usage: train.py [-h]
                [--novelty {none,axe,dist_trade,fence,fire,chest,mi_h,mi_cantplan,kibt,rdb,space_ar_hard,space_ar,moving_traders,busy_traders,multi_rooms}]
                [--seed SEED] [--num_threads NUM_THREADS] [--logdir LOGDIR] [--obs_type {lidar_all,lidar_lite,facing_only,hinted_only,matrix}]
                [--rl_algo {dqn,novel_boost,ppo,dsac,crr,crr_separate_net,gail,ppo_shared_net,icm_ppo,icm_ppo_shared_net}] [--metadata]
                [--exp_name EXP_NAME] [--env {sa,pf,rs,rs_s}] [--resume] [--checkpoint CHECKPOINT] [--lr LR] [--hidden_sizes HIDDEN_SIZES]
                [--device DEVICE]

Polycraft Gym Environment

optional arguments:
  -h, --help            show this help message and exit
  --novelty {none,axe,dist_trade,fence,fire,chest,mi_h,mi_cantplan,kibt,rdb,space_ar_hard,space_ar,moving_traders,busy_traders,multi_rooms}, -n {none,axe,dist_trade,fence,fire,chest,mi_h,mi_cantplan,kibt,rdb,space_ar_hard,space_ar,moving_traders,busy_traders,multi_rooms}
                        The name of the novelty.
  --seed SEED, -s SEED  The seed.
  --num_threads NUM_THREADS, -j NUM_THREADS
                        Number of sub threads used to run the env.
  --logdir LOGDIR, -o LOGDIR
                        The directory to save the logs.
  --obs_type {lidar_all,lidar_lite,facing_only,hinted_only,matrix}, -b {lidar_all,lidar_lite,facing_only,hinted_only,matrix}
                        Type of observation.
  --rl_algo {dqn,novel_boost,ppo,dsac,crr,crr_separate_net,gail,ppo_shared_net,icm_ppo,icm_ppo_shared_net}, -a {dqn,novel_boost,ppo,dsac,crr,crr_separate_net,gail,ppo_shared_net,icm_ppo,icm_ppo_shared_net}
                        The algorithm for RL.
  --metadata            Print metadata about the training and quit.
  --exp_name EXP_NAME   The name of the experiment, used to save results.
  --env {sa,pf,rs,rs_s}
                        The type of environment.
  --resume, -r          whether to resume training from a saved checkpoint.
  --checkpoint CHECKPOINT, --ckpt CHECKPOINT
                        The path to the checkpoint to load the model. This is used to fine tune a model. To resume training, use --resume instead.
  --lr LR               Learning Rate
  --hidden_sizes HIDDEN_SIZES
                        Size of the hidden layer, separated by comma.
  --device DEVICE, -d DEVICE
                        device to be run on
```

For more detail on how to implement and inject your own novelties, see the [documentation](https://clarech712.github.io/ng-website/). For the built-in novelties for the NovelGym project, explore the [novelties](novelties) folder.

### Other Usage

For testing the integration of newly created novelties, see `manual_novelty_test1.py`.

For running rendering and a demonstration of a saved model, see `manual_sanity_checker.py`.

## Motivation

A wrapper on [NovelGridWorldsV2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2), [NovelGym](https://github.com/tufts-ai-robotics-group/NovelGym) focuses on the modularization and customization of the original project. Specifically, it

1. only uses one agent and focuses on environment development (the behavior of the remaining agents is determined by an environment entity controller),

2. adds and modularizes agent observation spaces,

3. adds and modularizes agent strategies in novelty encounters,

4. demonstrates the use of different libraries such as [tianshou](https://tianshou.readthedocs.io/en/master/).

## License

This repository bundles [Metric-FF](https://fai.cs.uni-saarland.de/hoffmann/metric-ff.html), distributed under the [GPLv2 License](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html).

## Citation

To cite the [NovelGym](https://clarech712.github.io/ng-website/) project, please use

```
@article{novelgym2023,
    title={NovelGym: A Flexible Ecosystem for Hybrid Planning and Learning Agents Designed for Open Worlds},
    author={Shivam Goel and Yichen Wei and Panagiotis Lymperopoulos and Klára Churá and Matthias Scheutz and Jivko Sinapov},
    booktitle={submitted for publication},
    year={2023}
}
```

