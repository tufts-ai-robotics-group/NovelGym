from typing import Tuple
import numpy as np
from gymnasium import spaces

from utils.env_reward_rapidlearn import RapidLearnRewardGenerator
from utils.item_encoder import SimpleItemEncoder
from .lidar_all import LidarAll

LOCAL_VIEW_SIZE=5
TARGET_OBJ="bedrock"


class Matrix(LidarAll):
    """Matrix-based observation space"""
    def __init__(self,
                 json_input: dict,
                 items_lidar_disabled=[],
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

        # things to search for. only excludes disabled items
        self.items_disabled = items_lidar_disabled
        self.items = list(filter(lambda item: item not in self.items_disabled, self.item_encoder.item_list.keys()))
        self.items_id = {self.item_encoder.get_id(keys): lidar_item_idx for lidar_item_idx, keys in enumerate(self.items)}

        # Representation of the agent's local view
        self.local_view_size = local_view_size  # Size of the local view grid (5x5)
        self.local_view = np.zeros((self.local_view_size, self.local_view_size))

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
            items_lidar_disabled=[],
            local_view_size=LOCAL_VIEW_SIZE,
            reserved_extra_objects=1, # in case we have new objects in the world
            *args,
            **kwargs
        ):
        # +1 since obj encoder has one extra error margin for unknown objects
        max_item_type_count = len(all_objects) + len(all_entities) + reserved_extra_objects + 1
        # things to search for in lidar. only excludes disabled items

        # maximum of number of possible items
        map_items_max_count = max_item_type_count - len(items_lidar_disabled)

        # Calculate the total number of cells in the local view
        num_cells = local_view_size ** 2

        # Define the observation space for each cell
        low_map = np.zeros((num_cells, num_cells, max_item_type_count))
        high_map = np.ones((num_cells, num_cells, max_item_type_count)) * map_items_max_count
        map_obs_space = spaces.Box(low_map, high_map, dtype=int)

        inventory_obs_space = spaces.Box(np.zeros(max_item_type_count), np.ones(max_item_type_count) * 40)
        selected_item_obs_space = spaces.Box([0], [max_item_type_count])

        observation_space = spaces.Dict({
            "map": map_obs_space,
            "inventory": inventory_obs_space,
            "selected_item": selected_item_obs_space
        })
        return observation_space


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
        observation = {
            "map": local_view, 
            "inventory": inventory_result, 
            "selected_item": [selected_item]
        }
        return observation

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
        Generates a local view matrix based on the player's position and the world map.
        """
        half_local_view = self.local_view_size // 2
        local_view = np.zeros((self.local_view_size, self.local_view_size, self.max_item_type_count))  # Add an extra dimension for channels

        for i in range(self.local_view_size):
            for j in range(self.local_view_size):
                row = player_pos[1] - half_local_view + i
                col = player_pos[0] - half_local_view + j

                if 0 <= col < world_map.shape[0] and 0 <= row < world_map.shape[1]:
                    # One-hot encode the value from the world map
                    value = int(world_map[row, col])
                    local_view[i, j, value] = 1
                else:
                    # Use a channel for the target_obj_encoded value
                    local_view[i, j, self.max_item_type_count - 1] = 1

        return local_view
