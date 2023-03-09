# This file contains the functions to call the planner and parse the output.
#
# By Yichen Wei
#
# Contains code in Shivam Goel's original Rapid-Learn Work.
#

import subprocess
import os
import re
import copy

FF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "planners", "Metric-FF-v2.1", "ff")

def call_planner(domain, problem):
    '''
        Given a domain and a problem file
        This function return the ffmetric Planner output.
        In the action format
    '''
    # run the planner.
    # using mode 0 because that's the most basic and stable mode.
    run_script = f"{FF_PATH} -o {domain} -f {problem} -s 0"
    output = subprocess.getoutput(run_script)
    plan, game_action_set = _output_to_plan(output, {})
    return plan, game_action_set

def _output_to_plan(output, action_map):
    '''
    Helper function to perform regex on the output from the planner.
    ### I/P: Takes in the ffmetric output and
    ### O/P: converts it to a action sequence list.
    '''

    if "unsolvable" in output:
        print ("Plan not found with FF! Error: {}".format(
            output))
        return None, None

    ff_plan = re.findall(r"\d+?: (.+)", output.lower()) # matches the string to find the plan bit from the ffmetric output.
    action_set = [tuple(statement.split(" ")) for statement in ff_plan]

    if len(ff_plan) == 0:
        return ["nop", "nop"], ["nop", "nop"]
    
    if ff_plan[-1] == "reach-goal":
        ff_plan = ff_plan[:-1]
    
    # convert the action set to the actions permissable in the domain
    game_action_set = [translate_action(action) for action in action_set]

    # for i in range(len(game_action_set)):
    #     if game_action_set[i].split(" ")[0] != "approach" and game_action_set[i].split("_")[0] != "Select":
    #         game_action_set[i] = action_map[game_action_set[i]]
    # for i in range(len(game_action_set)):
    #     if game_action_set[i] in action_map:
    #         game_action_set[i] = env.actions_id[game_action_set[i]]
    return action_set, game_action_set

def translate_action(action):
    if "approach" in action[0]:
        return f"approach_{action[2]}"
    elif action[0] == "select":
        return f"select_{action[2]}"
    elif "break" in action[0]:
        return "break_block"
    elif "collect" in action[0]:
        return "collect"
    else:
        return action[0]

