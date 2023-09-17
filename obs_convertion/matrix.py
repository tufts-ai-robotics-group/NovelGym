from typing import Tuple
import numpy as np
from gymnasium import spaces

from utils.env_reward_rapidlearn import RapidLearnRewardGenerator
from utils.item_encoder import SimpleItemEncoder
from .base import ObservationGenerator

LOCAL_VIEW_SIZE=5


class Matrix(ObservationGenerator):
    """Matrix-based observation space"""
    def __init__(self,
                 json_input: dict,
                 RL_test=False,
                 local_view_size=LOCAL_VIEW_SIZE,
                 num_reserved_extra_objects=1,
                 *args,
                 **kwargs
        ) -> None:
        """
        The Env is instantiated using the first json input.
        """
        # Encoder for automatically encoding new objects
        self.max_item_type_count, self.item_encoder = self._encode_items(json_input['state'], num_reserved_extra_objects)

        # Representation of the agent's local view
        self.local_view_size = local_view_size  # Size of the local view grid (5x5)
        self.local_view = np.zeros((self.local_view_size, self.local_view_size))

        # Inventory
        self.inventory = self._generate_inventory(json_input)

        # Selected item
        self.selected_item = self._get_selected_item(json_input)

        # Limits for the local view (assume maximum values for now)
        low = np.array([0] * (self.local_view_size ** 2))
        high = np.array([self.max_item_type_count] * (self.local_view_size ** 2))

        # Observation space
        self.observation_space = spaces.Box(low, high, dtype=int)

        # Reward generator (if applicable)
        if 'domain' not in json_input:
            self.reward_generator = None
        else:
            self.reward_generator = RapidLearnRewardGenerator(
                pddl_domain=json_input['domain'],  # Un-escape PDDL string
                initial_state=self.get_state_for_evaluation(json_input['state']),
                failed_action_exp=json_input['state']['action'],
                item_encoder=self.item_encoder,
                plan=json_input.get('plan'),
                RL_test=RL_test  # You may need to adjust this depending on your usage
            )
            self.failed_action = json_input['state']['action'][1:-1].replace(" ", "_")
        
        self.novel_action_set = json_input['novelActions']
        self.action_set = json_input.get('actionSet') or list(self.reward_generator.actions.keys())


    @staticmethod
    def get_observation_space(
            all_objects,
            all_entities,
            local_view_size=LOCAL_VIEW_SIZE,
            *args,
            **kwargs
        ):
        # Calculate the total number of cells in the local view
        num_cells = local_view_size ** 2

        # Define the observation space for each cell
        low = np.array([0] * num_cells)  # Minimum observation value
        high = np.array([len(all_objects) + len(all_entities)] * num_cells)  # Maximum observation value

        observation_space = spaces.Box(low, high, dtype=int)
        return observation_space


    def init_info(self):
        """
        Returns the init info for the discover executor
        """
        return {
            "failed_action": self.failed_action,
            "action_set": self.action_set,
            "observation_space": self.observation_space.sample(),
            "novel_action_set": self.novel_action_set
        }


    def get_action_name(self, action_id: int):
        """Get action name from action id"""
        return self.reward_generator.action_name_set[action_id]


    def get_novel_action_name(self, action_id: int):
        """Get novel action name from action id"""
        return self.novel_action[action_id]


    def check_if_effects_met(self, new_state_json: dict) -> bool:
        """Check if effects of action met"""
        state = self.get_state_for_evaluation(new_state_json)
        return self.reward_generator.check_if_effect_met(state)


    def check_if_plannable_state_reached(self, new_state_json: dict) -> bool:
        """Check if a state has been reached from which a plan can be made"""
        state = self.get_state_for_evaluation(new_state_json)
        return True


    def generate_observation(self, json_input: dict) -> np.ndarray:
        """
        generates a numpy array representing the state from the json file.
        takes in json['state']
        """
        state_json = json_input

        # Calculate the 5x5 local view matrix
        world_map, min_coord, _ = self._generate_map(state_json)
        player_pos = np.array(state_json["player"]["pos"]) - min_coord
        local_view = self._generate_local_view(player_pos, world_map)

        # inventory
        inventory_result = self._generate_inventory(state_json)

        # selected item
        selected_item = self._get_selected_item(state_json)

        # Combine the local view matrix, inventory, and selected item into a single observation array
        observation = np.concatenate((local_view.flatten(), inventory_result, [selected_item]), dtype=int)

        return observation


    def get_state_for_evaluation(self, json_input: dict) -> dict:
        """Gets state for evaluation"""
        pos_x, pos_y = json_input["player"]["pos"]
        map, min, _ = self._generate_map(json_input)
        return {
            "inventory": self._generate_inventory(json_input),
            "world": self._get_object_count_in_world(json_input),
            "holding": self._get_selected_item(json_input),
            "map": map,
            "pos": (pos_x - min[0], pos_y - min[1]),
            "facing": json_input["player"]["facing"]
        }
    

    #################################################################
    # Util to encode items
    #################################################################
    def _encode_items(self, json_data, num_extra_objects):
        """
        Run through the json and loads items into the list. Returns the number of items.
        Used to know how many items to expect so we can instantiate the array accordingly.
        """
        self.item_encoder = SimpleItemEncoder({"air": 0})
        # This is a dry run of some functions to make sure the all items are included.
        # Nothing will be returned. The same function will be run again when generate_observation is called.
        self._generate_map(json_data)

        for slot in json_data['inventory']['slots']:
            self.item_encoder.get_id(slot['item'])

        all_items_keys = sorted(self.item_encoder.item_list)

        # create a new one with sorted keys
        self.item_encoder = SimpleItemEncoder(
            {key: idx for idx, key in enumerate(all_items_keys)},
            placeholder_count=num_extra_objects
        )

        # prevent new items from being encoded since we made the 
        #     assumption that no new items will be discovered after the first run.
        self.item_encoder.id_limit = len(self.item_encoder.item_list)
        return self.item_encoder.id_limit, self.item_encoder


    #################################################################
    # Util to generate a map
    #################################################################
    def _find_bounding_box(self, map) -> Tuple[np.ndarray, np.ndarray]:
        """
        Auxiliary function to find max and min coord when the function is first run.
        input: json['map']
        returns: tuple of coordinates denoting the min and max for each coord component
        """
        coords_ND = np.array([coord.split(",") for coord in map.keys()], dtype=np.float64) # list of all coordinates
        min_coord_D = coords_ND.min(axis=0).astype('int')
        max_coord_D = coords_ND.max(axis=0).astype('int')
        return min_coord_D, max_coord_D


    def _generate_map(self, json_data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a numpy map from the json data. returns the map, min_coord, max_coord
        """
        min_coord, max_coord = self._find_bounding_box(json_data['map'])
        ng_map = -np.zeros((max_coord - min_coord + 1))

        for key, item in json_data['map'].items():
            curr_coord = np.array(key.split(",")).astype(int)
            ng_map[tuple(curr_coord - min_coord)] = self.item_encoder.get_id(item)

        return ng_map, min_coord, max_coord


    #################################################################
    # Util to generate a local view
    #################################################################
    def _generate_local_view(self, player_pos, world_map):
        """
        Generates a 5x5 local view matrix based on the player's position and the world map.
        """
        local_view_size = 5
        half_local_view = local_view_size // 2
        local_view = np.zeros((local_view_size, local_view_size))

        for i in range(local_view_size):
            for j in range(local_view_size):
                x = player_pos[0] - half_local_view + i
                y = player_pos[1] - half_local_view + j

                if 0 <= x < world_map.shape[0] and 0 <= y < world_map.shape[1]:
                    local_view[i, j] = world_map[x, y]

        return local_view


    #################################################################
    # Util to generate state about the inventory
    #################################################################
    def _get_selected_item(self, input: dict) -> int:
        """
        Gets the id of the selected item.
        """
        inventory = input['inventory']
        selected_item = inventory['selectedItem']
        return self.item_encoder.get_id(selected_item)


    def _generate_inventory(self, input: dict) -> np.ndarray:
        """
        Generates the inventory part of the state representation.
        """
        inventory = input['inventory']['slots']
        # minus 1 to exclude air in the inventory. (0 = air for our encoder)
        inventory_quantity_arr = np.zeros(self.max_item_type_count, dtype=int)

        for slot in inventory:
            item_id = self.item_encoder.get_id(slot['item'])
            inventory_quantity_arr[item_id] += slot['count']
        return inventory_quantity_arr


    #################################################################
    # Util to generate the state for reward function evaluation
    #################################################################
    def _get_object_count_in_world(self, json_input: dict) -> np.ndarray:
        """
        counts the number of objects in the world.
        """
        item_count = np.zeros(self.item_encoder.id_limit)

        for _, item in json_input['map'].items():
            item_id = self.item_encoder.get_id(item)
            item_count[item_id] += 1
        return item_count


if __name__ == '__main__':
    RepGenerator = Matrix()
