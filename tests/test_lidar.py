from utils.item_encoder import SimpleItemEncoder
from utils.env_reward_utils import PolycraftRewardGenerator, parse_failed_action_statement
from utils.pddl_utils import generate_obj_types

import json
from obs_convertion import LidarAll
from gym_novel_gridworlds2.utils.json_parser import load_json, ConfigParser
import numpy as np
import os

JSON_CONFIG_PATH = "config/polycraft_gym_main.json"

def test_observation_space():
    config_json = load_json(JSON_CONFIG_PATH)
    all_objects = generate_obj_types(config_json)
    obs_space = LidarAll.get_observation_space(all_objects)
    print(obs_space)
    print(obs_space.shape)
    assert obs_space.shape == ((len(all_objects.items()) + 2) * (8 + 1) + 1,)
