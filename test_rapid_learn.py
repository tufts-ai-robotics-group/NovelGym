from gym_novel_gridworlds2.utils.json_parser import load_json, ConfigParser
from utils.diarc_json_utils import generate_diarc_json_from_state
import json

def test_diarc_str():
    JSON_CONFIG_PATH = "config/polycraft_gym_main.json"
    config_json = load_json(JSON_CONFIG_PATH)
    parser = ConfigParser()
    state, dynamics, agent_manager = parser.parse_json(None, config_json, 100)

    diarc_json = generate_diarc_json_from_state(0, state, "cannotplan", False)
    print(json.dumps(diarc_json))


if __name__ == "__main__":
    test_diarc_str()

