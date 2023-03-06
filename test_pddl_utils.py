from utils.pddl_utils import generate_pddl
from gym_novel_gridworlds2.utils.json_parser import load_json, ConfigParser
import json

def test_generate_pddl():
    JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
    config_json = load_json(JSON_CONFIG_PATH)
    parser = ConfigParser()
    state, dynamics, agent_manager = parser.parse_json(None, config_json, 100)
    
    
    pddl_domain, pddl_problem = generate_pddl(config_json, state, dynamics)

    with open("pddl_domain.pddl", "w") as f:
        f.write(pddl_domain)
    with open("pddl_problem.pddl", "w") as f:
        f.write(pddl_problem)

def test_plan():
    pass


if __name__ == "__main__":
    test_generate_pddl()

