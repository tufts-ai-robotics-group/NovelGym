from utils.item_encoder import SimpleItemEncoder
from utils.env_reward_rapidlearn import RapidLearnRewardGenerator, parse_failed_action_statement
from utils.pddl_utils import generate_obj_types
from utils.hint_utils import get_hinted_items


import json
from obs_convertion import LidarAll
from gym_novel_gridworlds2.utils.json_parser import load_json, ConfigParser
import numpy as np
import os

JSON_CONFIG_PATH = "config/polycraft_gym_main.json"

def test_observation_space():
    config_json = load_json(JSON_CONFIG_PATH)
    all_objects = generate_obj_types(config_json)
    hints = [
        "Sorry, you need a key to trade with me.",
        "(trade_block_of_titanium_1)"
    ]
    assert get_hinted_items(all_objects, hints) == ["block_of_titanium"]
    assert get_hinted_items(all_objects, hints, split_words=True) == [
        'block_of_platinum', 
        'block_of_titanium', 
        'block_of_diamond', 
        'blue_key'
    ]
