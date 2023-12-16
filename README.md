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

Next, clone this repository and install dependencies using `pip`, keeping the same virtual environment activated (if applicable).

```
cd ..
git clone https://github.com/tufts-ai-robotics-group/NovelGym.git
cd NovelGym
pip install .
```

### Compile MetricFF

Finally, install your C compiler by going to `planners/Metric-FF-v2.1`, and running `make`, as follows.

```
cd planners/Metric-FF-v2.1
make
```

## Basic Usage

### Project Structure

The project consists of an environment, taken from [NovelGridWorldsV2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2), and a single-agent training architecture. See [docs/project_structure.md](docs/project_structure.md) for a detailed explanation of the project structure.

### Online Training

A single agent can be trained by running `train.py`. To see the full list of usage conventions and optional arguments, run the below. The results of the run will be saved in the [results](results) folder.

```
python train.py -h
```

An example command would be the below, where we train an RapidLearn<sup>+</sup>(PPO+ICM) agent with an LiDAR-based observation space on a task. We do not test the agent on a novelty.

```
python train.py --novelty none --seed 1 --obs_type lidar_all --env sa --rl_algo icm_ppo
```

For specifics of the task being trained for and the details of the available agents, see [docs/project_structure.md](docs/project_structure.md)

### Testing Novelties

In order to train the same agent as above and test it on the built-in fence novelty, run the following.

```
python train.py --novelty fence --seed 1 --obs_type lidar_all --env sa --rl_algo icm_ppo
```

For more detail on how to implement and inject your own novelties, see NovelGridWorldsV2 documentation. For the built-in novelties for the NovelGym project, explore the [novelties](novelties) folder.

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