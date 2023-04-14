# SAILON RL Gym V2

## Installation
### Install NovelGridWorlds
Clone [NovelGridWorlds V2](https://github.com/tufts-ai-robotics-group/NovelGridWorldsV2), and install it by running the command
```
pip install -e .
```

### Compile MetricFF
Install your C compiler and make. Goto `planners/Metric-FF-v2.1`, and
then run 
```
make
```


## Running
Run `train.py` for testing, and for a parallel environment, Run `train_parallel.py`.

The command line options are:
```
usage: train.py [-h] [--novelty {mi,kibt,rdb,space_ar}] [--seed SEED] [--num_threads NUM_THREADS] [--logdir LOGDIR]
                [--obs_type {lidar_all,only_facing,only_hinted}] [--rl_algo {dqn,novel_boost}]

Polycraft Gym Environment

options:
  -h, --help            show this help message and exit
  --novelty {mi,kibt,rdb,space_ar}
                        The name of the novelty.
  --seed SEED           The seed.
  --num_threads NUM_THREADS
                        Number of sub threads used to run the env.
  --logdir LOGDIR       The directory to save the logs.
  --obs_type {lidar_all,only_facing,only_hinted}
                        Type of observation.
  --rl_algo {dqn,novel_boost}
                        The algorithm for RL.
```

The results of the run will be saved in the "results" folder.

## License
This repo bundles MetricFF, which is distributed under the GPLv2 License.
