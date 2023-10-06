from obs_convertion import LidarAll, OnlyFacingObs, NovelOnlyObs, Matrix
import tianshou as ts
from policies import BiasedDQN
from net.basic import BasicNet
from net.norm_net import NormalizedNet
from envs.single_agent_standard import SingleAgentWrapper

OBS_TYPES = {
    "lidar_all": LidarAll,
    "lidar_lite": LidarAll,
    "facing_only": OnlyFacingObs,
    "hinted_only": NovelOnlyObs,
    "matrix": Matrix
}

OBS_GEN_ARGS = {
    "lidar_lite": {
        "num_beams": 4,
        "max_beam_range": 2
    }
}

NOVELTIES = {
    "none": "",
    "axe": "novelties/evaluation1/axe_to_break/axe_to_break.yaml",
    "dist_trade": "novelties/evaluation1/dist_trade/dist_trade.yaml",
    "fence": "novelties/evaluation1/fence/fence_easy.yaml",
    "chest": "novelties/evaluation1/chest_shortcut/chest_shortcut.yaml",
    "mi_h": "novelties/evaluation1/multi_interact/multi_interact.yaml",
    "mi_cantplan": "novelties/evaluation1/multi_interact/multi_interact_cant_plan.yaml",
    "kibt": "novelties/evaluation1/key_inventory_trade/key_inventory_trade.yaml",
    "rdb": "novelties/evaluation1/random_drop_break/random_drop_break.yaml",
    "space_ar_hard": "novelties/evaluation1/space_around/space_around_hard_high_radius.yaml",
    "space_ar": "novelties/evaluation1/space_around/space_around.yaml",
    "moving_traders": "novelties/evaluation1/moving_traders/moving_traders.yaml",
    "busy_traders": "novelties/evaluation1/busy_traders/busy_traders.yaml",
    "multi_rooms": "novelties/evaluation1/multi_rooms/multi_rooms.yaml",
}

HINTS = {
    "mi_h": "The trader will reward you for interacting with him.",
    "mi_cantplan": "The trader will reward you for interacting with him.",
    "kibt": str([
        "Sorry, you need a key to trade with me.",
        "Find the key in the chest.",
        "(trade_block_of_titanium_1)"
    ]),
    "axe": "",
    "rdb": "",
    "space_ar": "",
}

NOVEL_ACTIONS = {
    "mi_h": [],
    "mi_cantplan": [],
    "kibt": ["approach_plastic_chest", "collect"],
    "axe": [],
    "rdb": [],
    "space_ar": [],
}

POLICIES = {
    "dqn": ts.policy.DQNPolicy,
    "novel_boost": BiasedDQN,
    "ppo": ts.policy.PPOPolicy, 
    "dsac": ts.policy.DiscreteSACPolicy,
    "crr": ts.policy.DiscreteCRRPolicy,
    "crr_separate_net": ts.policy.DiscreteCRRPolicy,
    "gail": ts.policy.GAILPolicy,
    "ppo_shared_net": ts.policy.PPOPolicy,
    "icm_ppo": None,
    "icm_ppo_shared_net": None
}

POLICY_PROPS = {
    "dqn": {},
    "novel_boost": {
        "novel_boost": 2
    }
}

AVAILABLE_ENVS = {
    "sa": None,
    "pf": None,
    "rs": None,
    "rs_s": None
}

AVAILABLE_WRAPPERS = {
    "sa": [SingleAgentWrapper]
}


RL_ALGOS = {
    "dqn": ts.policy.DQNPolicy,
}

NETS = {
    "basic": BasicNet,
    "normalized": NormalizedNet,
}

REWARDS = {
    "positive": 1000,
    "negative": -250,
    "step": -1,
}
