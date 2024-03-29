from utils.pddl_utils import KnowledgeBase
from utils.plan_utils import call_planner
from gym_novel_gridworlds2.utils.json_parser import load_json, ConfigParser
import json

def test_generate_pddl():
    JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
    config_json = load_json(JSON_CONFIG_PATH)
    parser = ConfigParser()
    state, dynamics, agent_manager = parser.parse_json(None, config_json, 100)
    
    kb = KnowledgeBase(config_json)
    pddl_domain, pddl_problem = kb.generate_pddl(state, dynamics)

    with open("pddl_domain.pddl", "w") as f:
        f.write(pddl_domain)
    with open("pddl_problem.pddl", "w") as f:
        f.write(pddl_problem)

def test_plan():
    plan, translated = call_planner("pddl_domain.pddl", "pddl_problem.pddl")
    print(plan)
    for t in translated:
        print(t)



if __name__ == "__main__":
    test_generate_pddl()
    test_plan()

