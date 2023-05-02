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
usage: train.py [-h] [--novelty {mi,mi_cantplan,kibt,axe,rdb,space_ar,fence}] [--seed SEED] [--num_threads NUM_THREADS] [--logdir LOGDIR] [--obs_type {lidar_all,lidar_lite,facing_only,hinted_only}] [--rl_algo {dqn,novel_boost}] [--metadata] [--exp_name EXP_NAME]

Polycraft Gym Environment

options:
  -h, --help            show this help message and exit
  --novelty {mi,mi_cantplan,kibt,axe,rdb,space_ar,fence}, -n {mi,mi_cantplan,kibt,axe,rdb,space_ar,fence}
                        The name of the novelty.
  --seed SEED, -s SEED  The seed.
  --num_threads NUM_THREADS, -j NUM_THREADS
                        Number of sub threads used to run the env.
  --logdir LOGDIR, -o LOGDIR
                        The directory to save the logs.
  --obs_type {lidar_all,lidar_lite,facing_only,hinted_only}, -b {lidar_all,lidar_lite,facing_only,hinted_only}
                        Type of observation.
  --rl_algo {dqn,novel_boost}, -a {dqn,novel_boost}
                        The algorithm for RL.
  --metadata            Print metadata about the training and quit.
  --exp_name EXP_NAME   The name of the experiment, used to save results.
```

The results of the run will be saved in the "results" folder.

## License
This repo bundles MetricFF, which is distributed under the GPLv2 License.
