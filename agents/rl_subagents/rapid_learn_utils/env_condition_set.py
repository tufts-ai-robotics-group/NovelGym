from typing import Mapping, Tuple, List

QUANTITY_CMDS = ["increase", "decrease", ">="]


def q_token_to_rep(tokens):
    return '-'.join(tokens)

def rep_to_q_token(tokens_str: str):
    return tokens_str.split("-")


class ConditionSet:
    """
    Assumes all and
    """
    def __init__(self) -> None:
        self.conditions: list = []

        # dict of 
        # list of {"op": ">=", "quantity": 2}
        self.quantities: Mapping[str, List[Mapping[str, int]]] = {}
    
    def to_condition_tokens(self):
        """
        Converts itself into tokens for creation of a reward function
        """
        conditions = ["and"] + self.conditions
        for key, req_list in self.quantities.items():
            for req in req_list:
                conditions.append([req['op'], rep_to_q_token(key), req['quantity']])
        return conditions


    def add_condition(self, new_cond) -> None:
        """
        Add a condition to the ConditionSet.
        new_cond is the tokens represented as a list.
        """
        # Generic add condition function that accounts for the add_condition list
        if new_cond[0] == "not":
            self.remove_condition(new_cond[1])
        elif new_cond[0] == "and":
            for cond in new_cond[1]:
                self.add_condition(cond)
        elif new_cond[0] in QUANTITY_CMDS:
            self._add_quantity_condition(new_cond)
        else:
            self._add_fact_condition(new_cond)


    def _add_fact_condition(self, new_cond) -> None:
        """
        Addes fact-based condition to the list
        """
        if new_cond not in self.conditions:
            self.conditions.append(new_cond)
    

    def _add_quantity_condition(self, cond):
        """
        Adds quantity-based condition (>=, increase, etc) to the list
        """
        # assert q_token_to_rep(cond1[1]) == q_token_to_rep(cond2[1])
        cond_rep = q_token_to_rep(cond[1])
        quantity_requirements = self.quantities.get(cond_rep)
        if quantity_requirements is None:
            quantity_requirements = []
            self.quantities[cond_rep] = quantity_requirements
        
        if cond[0] == "increase":
            # increase of requirements
            if len(quantity_requirements) == 0:
                # if there's no such requirement for this value before,
                # which is equivalent to the requirement that it is >= 0,
                # We need to add this to the list to make sure an increase 
                # of the quantity happened
                quantity_requirements.append({
                    "op": ">=", 
                    "quantity": int(cond[2])
                })
            else:
                # increase / decrease the limits accordingly
                for item in quantity_requirements:
                    item["quantity"] += int(cond[2])
        elif cond[0] == "decrease":
            # decrease of requirements
            if len(quantity_requirements) != 0:
                for item in quantity_requirements:
                    item["quantity"] -= int(cond[2])
                    if item["quantity"] < 0:
                        # make sure we never get to negative
                        item["quantity"] = 0
        else:
            # just limits, we compare and take the sufficient requirements
            for item in quantity_requirements:
                if item["op"] == cond[0]:
                    if item["op"] == ">=":
                        item["quantity"] = max(item["quantity"], int(cond[2]))
                    elif item["op"] == "<=":
                        item["quantity"] = min(item["quantity"], int(cond[2]))
                


    def remove_condition(self, condition) -> None:
        """
        Removes a condition to the ConditionSet.
        new_cond is the tokens represented as a list.
        """
        # Generic add condition function that accounts for the add_condition list
        # TODO: does not account for NOT: assuming not condition 
        #       will never specify quantities
        if condition[0] in QUANTITY_CMDS:
            # self._add_quantity_condition(condition)
            # print(self.quantities)
            pass
        else:
            if condition in self.conditions:
                self.conditions.remove(condition)


    def _remove_fact_condition(self, condition):
        """
        Addes fact-based condition to the list
        """
        try:
            self.conditions.remove(condition)
        except ValueError:
            pass

    def _remove_quantity_conditon(self, cond):
        # assert q_token_to_rep(cond1[1]) == q_token_to_rep(cond2[1])
        cond_rep = q_token_to_rep(cond[1])
        quantity_requirements = self.quantities.get(cond_rep)
        if quantity_requirements is None:
            quantity_requirements = []
            self.quantities[cond_rep] = quantity_requirements
        
        if cond[0] == "decrease":
            # decrease of requirements, removal means increase
            for item in quantity_requirements:
                item["quantity"] += int(cond[2])
        elif cond[0] == "increase":
            # decrease of requirements, removal means decrease
            for item in quantity_requirements:
                item["quantity"] -= int(cond[2])
                if item["quantity"] < 0:
                    # make sure we never get to negative
                    item["quantity"] = 0
        else:
            # just limits, we compare and take the sufficient requirements
            for item in quantity_requirements:
                if item["op"] == cond[0]:
                    if item["op"] == ">=":
                        item["quantity"] = max(item["quantity"], cond[2])
                    elif item["op"] == "<=":
                        item["quantity"] = min(item["quantity"], cond[2])
