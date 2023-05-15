from utils.item_encoder import SimpleItemEncoder
from utils.env_reward_rapidlearn import RapidLearnRewardGenerator, parse_failed_action_statement
import json
from obs_convertion import LidarAll
import numpy as np
import os
import cProfile



np.set_printoptions(threshold=np.inf)

def get_json(json_dict):
    env = LidarAll(json_dict, RL_test=True)
    env.generate_observation(json_dict)

    res = env.generate_observation(json_dict)
    print(res)

path = os.path.dirname(os.path.abspath(__file__))

f = open(os.path.join(path, 'init.json'))
data = json.load(f)


def test1():
    print("Testing Reward utility:")
    print("test 1")
    env = LidarAll(data, RL_test=True)
    state = env.get_state_for_evaluation(data['state'])
    pos = state['pos']

    a = RapidLearnRewardGenerator(
        pddl_domain=data['domain'].encode().decode('unicode_escape'), 
        initial_state=state,
        failed_action_exp="(break oak_log)",
        item_encoder=env.item_encoder,
        RL_test=True
    )
    assert a.check_if_effect_met(state)[0] == False

    # change so that it's true:
    state = env.get_state_for_evaluation(data['state'])
    state['map'][pos[0] - 2, pos[1]] = 0
    state['inventory'][env.item_encoder.get_id('oak_log')] += 1
    assert a.check_if_effect_met(state)[0] == True

    # change so that it's facing another item and is thus false
    state = env.get_state_for_evaluation(data['state'])
    state['map'][pos[0] - 2, pos[1]] = -1
    state['inventory'][env.item_encoder.get_id('oak_log')] += 1
    assert a.check_if_effect_met(state)[0] == False

    # change so that the obj decreased too much and is thus false
    state = env.get_state_for_evaluation(data['state'])
    state['map'][pos[0] - 2, pos[1]] = 0
    state['inventory'][env.item_encoder.get_id('oak_log')] += 1
    state['world'][env.item_encoder.get_id('oak_log')] -= 2
    assert a.check_if_effect_met(state)[0] == False

    # change back so that it's true
    state = env.get_state_for_evaluation(data['state'])
    state['map'][pos[0] - 2, pos[1]] = 0
    state['inventory'][env.item_encoder.get_id('oak_log')] += 2
    state['world'][env.item_encoder.get_id('oak_log')] -= 1
    assert a.check_if_effect_met(state)[0] == True


def test_whole_module():
    env = LidarAll(data, RL_test=True)
    # assert env.check_if_effects_met(data['state'])[0] == True
    print(env.reward_generator.plannable_state.to_condition_tokens())


