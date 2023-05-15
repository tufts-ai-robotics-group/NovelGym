from obs_convertion import LidarAll, OnlyFacingObs, NovelOnlyObs
import tianshou as ts
from policies import BiasedDQN
from net.basic import BasicNet
from net.norm_net import NormalizedNet

OBS_TYPES = {
    "lidar_all": LidarAll,
    "lidar_lite": LidarAll,
    "facing_only": OnlyFacingObs,
    "hinted_only": NovelOnlyObs,
}

OBS_GEN_ARGS = {
    "lidar_lite": {
        "num_beams": 4,
        "max_beam_range": 2
    }
}

NOVELTIES = {
    "none": "",
    "mi_h": "novelties/evaluation1/multi_interact/multi_interact.json",
    "mi_cantplan": "novelties/evaluation1/multi_interact/multi_interact_cant_plan.json",
    "kibt": "novelties/evaluation1/key_inventory_trade/key_inventory_trade.json",
    "axe": "novelties/evaluation1/axe_to_break/axe_to_break.json",
    "rdb": "novelties/evaluation1/random_drop_break/random_drop_break.json",
    "space_ar_hard": "novelties/evaluation1/space_around/space_around_hard_high_radius.json",
    "space_ar": "novelties/evaluation1/space_around/space_around.json",
    "fence": "novelties/evaluation1/fence/fence_easy.json",
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
    "crr": ts.policy.DiscreteCRRPolicy
}

POLICY_PROPS = {
    "dqn": {},
    "novel_boost": {
        "novel_boost": 2
    }
}

AVAILABLE_ENVS = {
    "sa": "Gym-SingleAgent-v0",
    "pf": "Gym-PlanningUntilFail-v0"
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
