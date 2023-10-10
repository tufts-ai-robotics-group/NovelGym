from typing import List, Mapping, Tuple
from xmlrpc.client import Boolean
import numpy as np
import re
from functools import lru_cache

from .env_condition_set import ConditionSet

# from env_utils import SimpleItemEncoder
from .advanced_item_encoder import PlaceHolderItemEncoder

"""
state = {
    "selectedItem": ...
    "map": ...
    "inventory": ...
    "location": ...
    "facing": ...
}
"""

ALTERNATIVE_NAMES = {
    "log": ["oak_log"]
}


def facing_to_coord(facing: str, distance: str):
    dis = 1
    if distance == "two":
        dis = 2
    
    if facing == "north":
        return (-dis, 0)
    elif facing == "south":
        return (dis, 0)
    elif facing == "east":
        return (0, dis)
    elif facing == "west":
        return (0, -dis)
    else:
        return (0, 0)


def scan_tokens(pddl_content: str, allow_multiple_statements: bool=False):
    """
    Given a string containing the content of a pddf file,
    returns the tokens in the pddl.

    -----
    This part of code adapted from
    https://github.com/pucrs-automated-planning/pddl-parser/blob/098e71b71109ae08e5970ccf497274dda127a11d/PDDL.py
    Licensed under GPL v3
    """
    # remove single line comments
    cleaned_str = re.sub(r';.*$', '', pddl_content, flags=re.MULTILINE).lower()
    # Tokenize
    stack = []
    list = []
    for t in re.findall(r'[()]|[^\s()]+', cleaned_str):
        if t == '(':
            stack.append(list)
            list = []
        elif t == ')':
            if stack:
                l = list
                list = stack.pop()
                list.append(l)
            else:
                raise Exception('PDDL Parse Error: Missing open parentheses')
        else:
            list.append(t)
    # if stack:
    #     raise Exception('PDDL Parse Error: Missing close parentheses') # ignore this error
    # if len(list) != 1 and not allow_multiple_statements:
    #     raise Exception('PDDL Parse Error: Malformed expression')
    return list[0] if len(list) == 1 else list


def parse_failed_action_statement(statement: str):
    return re.findall(r'[^\s(),]+', statement)


class RapidLearnRewardGenerator:
    # known bug: please don't reuse item_encoder among different Env / task!!!
    def __init__(self, pddl_domain, initial_state, failed_action_exp, item_encoder, plan=None, RL_test=False):
        # compatibility layer for RL vs DIARC
        self.RL_test = RL_test

        self.item_encoder = item_encoder
        # self.state = [None] # acts like a pointer so that the proper state can be captured.
        self.update_state(initial_state)
        self.domain_tokens = scan_tokens(pddl_content=pddl_domain)
        self.actions: Mapping[str, list] = {}
        self.action_name_set = []
        self.load_action_list(self.domain_tokens)
        # change back if we're using pddl again: 
        # raw_plan_tokens = scan_tokens(pddl_content=plan_exp, allow_multiple_statements=True)
        raw_action_tokens = parse_failed_action_statement(failed_action_exp)
        self.action_tokens = self._transform_action(tuple(raw_action_tokens))
        self.param_map = self.get_param_mapping(self.domain_tokens, self.action_tokens)
        self.check_func = self.load_check_effect_func(self.domain_tokens, self.action_tokens)
        self.type_dict = ALTERNATIVE_NAMES #self._parse_alternative_names()
        
        # entire plan, use to be expanded later
        self.plan_tokens = None
        self.plannable_state = None
        self.plannable_state_met_checker = lambda a: False # default value

        if plan is not None and len(plan) > 0:
            self.plan_tokens = scan_tokens(pddl_content=plan, allow_multiple_statements=True)

            # plannable state.
            self.plannable_state = ConditionSet()
            # TODO uncomment this to use plannable state
            # self._populate_plannable_state(self.plan_tokens, raw_action_tokens)

            # # create a function that checks whether plannable from the conditionset.
            # self.plannable_state_met_checker = self._make_check_function(
            #     self.plannable_state.to_condition_tokens()
            # )


    def update_state(self, state):
        self.state = state
    
    def get_param_mapping(self, param_tokens, action_tokens):
        """
        Takes in two lists of tokens:
        - param_tokens: the tokens of the parameter property in the definition
        of an action in the domain file
        - action_tokens: the tokens of the statement of an action in a plan

        Returns a mapping.
        """
        # instantiate an empty list
        param_map = {}

        # param statements look like this:
        # ['?actor', '-', 'actor', '?obj', '-', 'breakable']
        # we filter because all we need is '?actor' and '?obj' 
        # to create a one-to-one mapping from parameters and values.
        # auxiliary info like types ('- actor') is not needed.
        param_vars = filter(lambda token : token[0] == '?', param_tokens)

        for param_var, action_var in zip(param_vars, action_tokens[1:]):
            param_map[param_var] = action_var

        return param_map
    

    def _parse_alternative_names(self, types_json):
        print(types_json)
        pass


    def _substitute_params(self, conditions: list, param_mapping: Mapping[str, str]):
        """
        Substitutes parameters in the tokens using the mapping
        """
        new_conditions = []
        for item in conditions:
            if type(item) == str and item[0] == '?' and item in param_mapping:
                new_conditions.append(param_mapping[item])
            elif type(item) == list:
                new_conditions.append(self._substitute_params(item, param_mapping))
            else:
                new_conditions.append(item)
        return new_conditions

    
    def load_action_list(self, tokens):
        self.actions = {}
        for pddl_statement in tokens:
            if isinstance(pddl_statement, list) and \
                                pddl_statement[0] == ":action" and \
                                ":effect" in pddl_statement:
                self.actions[pddl_statement[1]] = pddl_statement
                self.action_name_set.append(pddl_statement[1])
    

    def _populate_plannable_state(self, plan_tokens, failed_action_tokens):
        """
        Populates self.plannable_state.
        - plan_tokens: untransformed, parsed plan
        - failed_action_tokens: untransformed, parsed failed action
        """
        t_failed_action = self._transform_action(tuple(failed_action_tokens))
        failed_action_def = self._get_action_def(t_failed_action[0], t_failed_action)

        for action in reversed(plan_tokens):
            if action == failed_action_tokens:
                break
            elif len(action) == 0:
                continue

            # transform the action in a plan into the new pddl domain 
            # print(action)
            try:
                t_action = self._transform_action(tuple(action))
            except Exception as e:
                raise Exception("Error understanding action " + str(action) + " in plan") from e
            
            # gets the definition of an action in the pddl domain
            action_def = self._get_action_def(t_action[0], t_action)
            
            # TODO fixme update
            is_subset = True      
            for precondition in action_def["preconditions"]:
                if precondition not in failed_action_def['effects']:
                    is_subset = False
            
            # if the precondition of the current operator is not in the 
            # effects of the failed operator
            if not is_subset:
                for precondition in action_def["preconditions"]:
                    # print("added", precondition)
                    self.plannable_state.add_condition(precondition)
                for effect in action_def["effects"]:
                    # print("removed", effect)
                    self.plannable_state.remove_condition(effect)

            
    def _get_action_def(self, action_name: str, action_tokens: List[str]=None) -> dict:
        """
        Gets the definition of an action in the domain file
        """
        for curr_name, curr_tokens in self.actions.items():
            if action_name == curr_name:
                # locate where the actions are
                param_index = curr_tokens.index(":parameters") + 1
                precondition_index = curr_tokens.index(":precondition") + 1
                effect_index = curr_tokens.index(":effect") + 1

                # gets the mapping from ?obj to an actual obj
                arg_mapping = None
                if action_tokens is not None:
                    arg_mapping = self.get_param_mapping(curr_tokens[param_index], action_tokens)

                # gets revelent requirements
                preconditions = self._substitute_params(
                    curr_tokens[precondition_index], 
                    arg_mapping
                )
                effects = self._substitute_params(
                    curr_tokens[effect_index], 
                    arg_mapping
                )
                
                return {
                    "params": curr_tokens[param_index],
                    "preconditions": preconditions[1:] if preconditions[0] == 'and' else [preconditions],
                    "effects": effects[1:] if effects[0] == 'and' else [effects]
                }
                
        raise KeyError(action_name + " not found")
    
    def _transform_action(self, at: List[str]):
        """
        Transforms an action from the general domain into the RL Domain.
        at: action_tokens, an array of tokens
        """
        # in the experiment, we don't need to transform the action
        if self.RL_test:
            return at
        
        # transformations
        if at[0] == "cannotplan":
            return at
        if at[0] in self.actions:
            return at
        specialized_action_name = at[0] + "_" + at[2]

        # Finding the name of the specialized action
        if specialized_action_name not in self.actions:
            if "approach_" in at[0]:
                specialized_action_name += "_by" + at[3]
        if specialized_action_name not in self.actions:
            return self._advanced_search_transform_action(at)
        
        # adding bytwo
        if "approach_" in at[0]:
            if at[0] == "approach_object" and at[3] == "two": # ?
                result = [specialized_action_name + "_bytwo", "self", *at[4:]]
            else:
                result =  [specialized_action_name, "self", *at[4:]]
            return result
        return [specialized_action_name, "self", *at[3:]]


    def _advanced_search_transform_action(self, at: List[str]):
        """
        accounts for more complicated actions
        assumes that we only 
        """
        for i in range(2, len(at)):
            specialized_action_name = at[0] + "_" + at[i]
            if specialized_action_name in self.actions:
                return [specialized_action_name, "self"]

        raise KeyError("Error: Action Name " + at[0] + " not found")


    def load_check_effect_func(self, tokens, action_params):
        if action_params[0] == "cannotplan":
            return self._maker_map['always_true']
        for statement in tokens:
            if isinstance(statement, list) and \
                                statement[0] == ":action" and \
                                ":effect" in statement:
                if statement[1] != action_params[0]:
                    continue
                # print(statement[1])
                param_index = statement.index(":parameters") + 1

                # create an alias from the parameters to its actual object
                mapping = self.get_param_mapping(statement[param_index], action_params)

                # process the list, replacing the parameters with the actual object
                effect_index = statement.index(":effect") + 1
                effects = statement[effect_index]
                transformed_effects = self._substitute_params(effects, param_mapping=mapping)

                # print("effects_tokens: ", transformed_effects)
                try:
                    return self._make_check_function(transformed_effects)
                except PlaceHolderItemEncoder.TooManyItemTypes as e:
                    raise Exception("Error while creating effect function for (" + ",".join(statement) + ")") from e
        print(action_params[0], "action not found!")
        return self._maker_map['always_false']
    
    def get_state(self):
        return self.state


    def check_if_effect_met(self, new_state) -> Tuple[Boolean, Boolean]:
        is_done = self.check_func(new_state)
        # is_plannable_state = self.plannable_state_met_checker(new_state)
        self.update_state(new_state)
        return is_done, False #is_plannable_state


    ##########################################################################
    # Generators for reward functions
    #
    ##########################################################################
    def _make_check_function(self, args):
        # print(args[0])
        if args[0] in self._maker_map:
            return self._maker_map[args[0]](self, *args)
        else: 
            return self._maker_map['always_true']
        
    def _make_check_holding_item(self, _, actor, item):
        def check_holding_item(new_state):
            if new_state["holding"] == self.item_encoder.get_id(item):
                return True
            elif item in self.type_dict:
                for alt_name in self.type_dict:
                    alt_id = self.item_encoder.get_id(alt_name)
                    if new_state["holding"] == alt_id:
                        return True
            return False
        return check_holding_item
    
    # def make_check_property_of(self, property_of):
    #     return lambda new_state: True

    # def make_check_next_to(self, obj1, obj2):
    #     return lambda new_state: True

    def _make_check_facing_object(self, *args):
        if self.RL_test:
            _, target_obj, distance = args
        else:
            _, actor, target_obj, distance, _ = args
        
        target_obj_id = self.item_encoder.get_id(target_obj)
        def check_facing_obj(new_state):
            # x, y inverted in the coord system
            x_diff, y_diff = facing_to_coord(new_state['facing'], distance)
            x, y = new_state['pos']
            try:
                if new_state['map'][y_diff + y, x_diff + x] == target_obj_id:
                    return True
                elif target_obj in self.type_dict:
                    for alternative_name in self.type_dict[target_obj]:
                        # print(alternative_name)
                        alt_id = self.item_encoder.get_id(alternative_name)
                        if new_state['map'][y_diff + y, x_diff + x] == alt_id:
                            return True
                return False
            except IndexError:
                # print("index error")
                return False
        return check_facing_obj
    
    def _make_check_quantity(self, op: str, quantity_exp: list, val: str):
        # creates a function that given a state representation will return
        #    the corresponding quantity given the expression.
        val_int = int(val)
        if 'air' in quantity_exp:
            # do not check quantity of air in the world or in the inventory
            return self._maker_map['always_true']

        get_quantity = self._make_get_quantity(*quantity_exp)
        if op == '>=':
            func = lambda new_state: get_quantity(new_state) >= val_int
        elif op == "decrease":
            func = lambda new_state: \
                get_quantity(new_state) - get_quantity(self.get_state()) >= -val_int
        elif op == "increase":
            func = lambda new_state: \
                get_quantity(new_state) - get_quantity(self.get_state()) >= val_int
        else:
            func = self._maker_map['always_true']
        func.__name__ = f"check_quantity_{op}_{quantity_exp}_{val}"
        return func
    
    def _make_get_quantity(self, quantity_of: str, *objs):
        # in the pddl the last argment is the item type in both world and inventory.
        item_id = self.item_encoder.get_id(objs[-1])

        def get_quantity(state):
            return state[quantity_of][item_id]
        get_quantity.__name__ = f"get_quantity_{quantity_of}_{item_id}"
        return get_quantity

    def _make_check_and(self, *params):
        fun_list = []
        for expression in params[1:]:
            fun_list.append(self._make_check_function(expression))
        
        def check_all(new_state): 
            result = True
            for func in fun_list:
                # print(func, func(new_state))
                result = result and func(new_state)
            return result
        check_all.__name__ = f"and_{params}"
        return check_all
    
    def _make_check_not(self, *params):
        fun = self._make_check_function(params[1])

        def check_not(new_state):
            result = fun(new_state)
            return not result
        check_not.__name__ = f"not_{params}"
        return check_not
    
    _maker_map = {
        "and": _make_check_and,
        "not": _make_check_not,
        "facing_obj": _make_check_facing_object,
        ">=": _make_check_quantity,
        "increase": _make_check_quantity,
        "decrease": _make_check_quantity,
        "holding": _make_check_holding_item,
        # default func that always returns true
        "always_true": lambda new_state: True,
        "always_false": lambda new_state: False
    }
