from typing import Mapping
import json

class PlaceHolderItemEncoder:
    class TooManyItemTypes(Exception):
        pass

    def __init__(self, item_list=None, initial_id=1, id_limit=0, placeholder_count=0):
        self.curr_id = initial_id - 1
        self.item_list: Mapping[str, int] = {}
        self.reverse_look_up_table = {}
        self.id_limit = id_limit
        if item_list is not None:
            self.load_item_list(item_list)
        
        # placeholders
        self.placeholders = []
        self.alloc_placeholders(placeholder_count)

    def alloc_placeholders(self, placeholder_count: int):
        """
        generates and preallocates spots for the placeholders.
        """
        for i in range(placeholder_count):
            # trivial name for the placeholder
            item_name = "__placeholder_" + str(i)
            # we use the get_id function, without using the placeholder,
            # to get allocate new id for the placeholder.
            item_id = self.get_id(item_name, use_placeholder=False)
            self.placeholders.insert(0, item_name)

    def load_item_list(self, item_list: Mapping[str, int]):
        """
        Load the item_list from a pre-set dictionary.
        """
        for key, value in item_list.items():
            self.curr_id = max(value, self.curr_id)
            self.reverse_look_up_table[value] = key
        self.item_list = item_list


    def load_json(self, file_name: str):
        """
        Loads the json file from a previous run
        """
        with open(file_name, 'r') as f:
            item_list = json.load(f)
            self.load_item_list(item_list["item_mapping"])
            self.placeholders = item_list["placeholders"]
            self.id_limit = item_list["id_limit"]
    
    def get_id(self, key: str, use_placeholder=True):
        """
        Takes in a key, returns a list. Not thread-safe.
        """
        if key in self.item_list:
            return self.item_list[key]
        elif len(self.placeholders) <= 0 and self.id_limit > 0 and self.curr_id + 1 >= self.id_limit:
            raise self.TooManyItemTypes(
                "Cannot add item \"" + key + "\" to the encoder because there are " +
                "too many types of items. Consider increasing the number of allowed item types."
            )
        else:
            if use_placeholder and len(self.placeholders) > 0:
                # if using pre-reserved spots
                placeholder_name = self.placeholders.pop()
                item_id = self.item_list[placeholder_name]
                del self.item_list[placeholder_name]
                self.item_list[key] = item_id
                self.reverse_look_up_table[item_id] = key
            else:
                # if no pre-allocated spots available, or if not using it.
                self.curr_id += 1
                self.item_list[key] = self.curr_id
                self.reverse_look_up_table[self.curr_id] = key
            return self.curr_id
    
    def modify_name(self, old_key, new_key, remove_old=False):
        """
        Modifies the name of an item.
        old_key: the old name of the item
        new_key: the new name of the item
        remove_old: the id-object lookup will always return the new key.
                    set this to true if you want to
                    remove the old name from the object-id lookup.
        """
        if old_key in self.item_list:
            self.item_list[new_key] = self.item_list[old_key]
            if remove_old:
                del self.item_list[old_key]
    
    def reverse_look_up(self, id: int):
        return self.reverse_look_up_table[id]
    
    def create_alias(self, alias_dict):
        """
        alias_dict: {"alias": "key"}
        """
        for alias, key in alias_dict.items():
            if key in self.item_list and alias not in self.item_list:
                self.item_list[alias] = self.item_list[key]

    def save_json(self, file_name: str):
        """
        Saves the encoding to a json file for future run
        """
        with open(file_name, 'w') as f:
            serialized = json.dumps({
                "item_mapping": self.item_list,
                "placeholders": self.placeholders,
                "id_limit": self.id_limit
            }, sort_keys=True)
            f.write(serialized)


    def from_json(self, file_name: str):
        with open(file_name, "r") as f:
            encoder_content = json.loads(f.read())
        self.load_item_list(encoder_content["item_dict"])
        self.id_limit = encoder_content["id_limit"]
