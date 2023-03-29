from utils.hint_utils import get_hinted_items
from utils.item_encoder import SimpleItemEncoder
from .lidar_all import LidarAll
import numpy as np
from gymnasium import spaces
from typing import Tuple

class OnlyHinted(LidarAll):
    def __init__(self, hints="", *args, **kwargs):
        # encoder for automatically encoding new objects
        self.item_encoder = SimpleItemEncoder({"air": 0})
        self.max_item_type_count = self._encode_items(kwargs['json_input']['state'])
        self.hinted_objects = get_hinted_items(self.item_encoder.item_list, hints, split_words=True)
        # self.hinted_item_encoder = SimpleItemEncoder(item_list=self.hinted_objects)
        self.items_id_hinted = {self.item_encoder.get_id(keys): item_idx for item_idx, keys in enumerate(self.hinted_objects)}
        super().__init__(*args, **kwargs)
        self.max_beam_range = 8

    @staticmethod
    def get_observation_space(
            all_objects, 
            items_lidar_disabled=[],
            hints: str="",
            *args,
            **kwargs
        ):
        # +1 since obj encoder has one extra error margin for unknown objects
        max_item_type_count = len(all_objects) + 1
        # # things to search for in lidar. only excludes disabled items

        hinted_objects = get_hinted_items(all_objects, hints, split_words=True)
        num_hinted_objects = len(hinted_objects)

        # maximum of number of possible items
        lidar_items_max_count = num_hinted_objects - len(items_lidar_disabled)

        # limits
        low = np.array(
            [0] * (lidar_items_max_count) + 
            [0] * num_hinted_objects + 
            [0]
        )
        high = np.array(
            [1] * (lidar_items_max_count) + 
            [40] * num_hinted_objects + 
            [max_item_type_count] # maximum 40 stick can be crafted (5 log -> 20 plank -> 40 stick)
        )
        observation_space = spaces.Box(low, high, dtype=int)
        return observation_space

    #################################################################
    # Util generate a lidar reading
    #################################################################
    def _lidar_sensors(self, player_pos: Tuple[int, int], player_facing: str, world_map: np.ndarray) -> np.ndarray:
        '''
        get the item that it's facing
        '''

        direction_radian = {'north': np.pi, 'south': 0, 'west': 3 * np.pi / 2, 'east': np.pi / 2}

        # Shoot beams in 360 degrees in front of agent
        angle = direction_radian[player_facing]

        lidar_signals = np.zeros((len(self.items_id_hinted)), dtype=int)
        x, y = player_pos

        x_ratio, y_ratio = np.round(np.cos(angle), 2), np.round((np.sin(angle)), 2)

        # Keep sending longer beams until hit an object or wall
        for beam_range in range(1, self.max_beam_range + 1):
            x_obj = x + np.round(beam_range * x_ratio)
            y_obj = y + np.round(beam_range * y_ratio)
            if x_obj >= world_map.shape[0] or y_obj >= world_map.shape[1] or x_obj < 0 or y_obj < 0:
                # prevent shooting out of the map
                break
            obj_id_rc = world_map[int(x_obj), int(y_obj)]

            if obj_id_rc != 0 and obj_id_rc in self.items_id_hinted:
                index = self.items_id_hinted[obj_id_rc]
                if lidar_signals[index] == 0:
                    lidar_signals[index] = beam_range
                # break # dont break because we want the objects even if we have occlusions.
        return lidar_signals


    def _generate_obs_inventory(self, input: dict) -> np.ndarray:
        """
        Generates the inventory part of the state representation.
        """
        inventory = input['inventory']['slots']
        # minus 1 to exclude air in the inventory. (0 = air for our encoder)
        inventory_quantity_arr = np.zeros(len(self.items_id_hinted), dtype=int)

        for slot in inventory:
            if slot['item'] not in self.items_id_hinted:
                continue
            item_id = self.items_id_hinted[self.item_encoder.get_id(slot['item'])]
            inventory_quantity_arr[item_id] += slot['count']
        return inventory_quantity_arr
    

    def generate_observation(self, json_input: dict) -> np.ndarray:
        """
        generates a numpy array representing the state from the json file.
        takes in json['state']
        """
        state_json = json_input
        # lidar beams
        world_map, min_coord, max_coord = self._generate_map(state_json)
        player_pos = np.array(state_json["player"]["pos"]) - min_coord
        sensor_result = self._lidar_sensors(tuple(player_pos), state_json['player']['facing'], world_map)
        # inventory
        inventory_result = self._generate_obs_inventory(state_json)
        # selected item
        selected_item = self._get_selected_item(state_json)

        return np.concatenate((sensor_result, inventory_result, [selected_item]), dtype=int)
