import math
import random

#Smart-exploration
# MAX_EPSILON = 0.3

# #Epsilon-greedy
MAX_EPSILON = 0.1

MAX_TIMESTEPS = 300
MAX_RHO = 0.35
MIN_RHO = 0.05
EXPLORATION_STOP = 50 #10000
# EXPLORATION_STOP = 10000 #10000
SCORE_TO_CHECK = 750
NO_OF_SUCCESSFUL_DONE = 15

# remains same always
UPDATE_RATE = 5 # network weights update rate
MAX_EPISODES = 5000
EPS_TO_EVAL = 10
EVAL_INTERVAL = 10
NUM_HIDDEN = 512
GAMMA = 0.98
LEARNING_RATE = 3e-5
# LEARNING_RATE = 1e-3
DECAY_RATE = 0.99
MIN_EPSILON = 0.05
random_seed = random.randint(0, 199)
# print ("Seed ->:", random_seed)
LAMBDA = -math.log(0.01) / EXPLORATION_STOP # speed of decay
PRINT_EVERY = 101 # logging
# convergence criteria
NO_OF_EPS_TO_CHECK = 100
NO_OF_DONES_TO_CHECK = 20

POSITIVE_REINFORCEMENT = 1000
NEGATIVE_REINFORCEMENT = -350
NORMAL_REINFORCEMENT = -1
