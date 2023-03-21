from gym_novel_gridworlds2.utils.json_parser import load_json, ConfigParser
from utils.diarc_json_utils import generate_diarc_json_from_state
from utils.pddl_utils import generate_obj_types
import json

from utils.plan_utils import call_planner

def test_diarc_str():
    JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
    config_json = load_json(JSON_CONFIG_PATH)
    parser = ConfigParser()
    state, dynamics, agent_manager = parser.parse_json(None, config_json, 100)

    dynamics.all_objects = generate_obj_types(config_json)
    diarc_json = generate_diarc_json_from_state(0, state, dynamics, "cannotplan", False)
    print(json.dumps(diarc_json))
    assert diarc_json["failedAction"] == "cannotplan"


def test_init_json():
    JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
    with open("pddl_domain.pddl", "r") as f:
        pddl_domain = f.read()
    plan, translated = call_planner("pddl_domain.pddl", "pddl_problem.pddl")

    config_json = load_json(JSON_CONFIG_PATH)
    parser = ConfigParser()
    state, dynamics, agent_manager = parser.parse_json(None, config_json, 100)

    dynamics.all_objects = generate_obj_types(config_json)
    diarc_json = generate_diarc_json_from_state(0, state, dynamics, "(break oak_log)", False)

    action_set = agent_manager.agents['agent_0'].action_set
    init_dict = {
        "state": diarc_json,
        "domain": pddl_domain,
        "plan": "\n".join("(" + " ".join(item) + ")" for item in plan),
        "novelActions": [],
        "actionSet": [action[0] for action in action_set.actions if action not in ["nop", "give_up"]],
    }
    print(init_dict)
    with open("tests/init.json", "w") as f:
        json.dump(init_dict, f)


if __name__ == "__main__":
    test_diarc_str()

