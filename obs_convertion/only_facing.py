from .lidar_all import LidarAll
import numpy as np
from gymnasium import spaces
from typing import Tuple

class OnlyFacingObs(LidarAll):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_beam_range = 1

    @staticmethod
    def get_observation_space(
            all_objects, 
            all_entities,
            items_lidar_disabled=[],
            *args,
            **kwargs
        ):
        # +1 since obj encoder has one extra error margin for unknown objects
        max_item_type_count = len(all_objects) + len(all_entities) + 1
        # things to search for in lidar. only excludes disabled items

        # maximum of number of possible items
        lidar_items_max_count = max_item_type_count - len(items_lidar_disabled)

        # limits
        low = np.array(
            [0] * (lidar_items_max_count) + 
            [0] * max_item_type_count + 
            [0]
        )
        high = np.array(
            [1] * (lidar_items_max_count) + 
            [40] * max_item_type_count + 
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

        lidar_signals = np.zeros((len(self.items_id_lidar)), dtype=int)
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

            if obj_id_rc != 0 and obj_id_rc in self.items_id_lidar:
                index = self.items_id_lidar[obj_id_rc]
                if lidar_signals[index] == 0:
                    lidar_signals[index] = beam_range
                break # dont break because we want the objects even if we have occlusions.
        
        return lidar_signals
    

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
        inventory_result = self._generate_inventory(state_json)
        # selected item
        selected_item = self._get_selected_item(state_json)

        return np.concatenate((sensor_result, inventory_result, [selected_item]), dtype=int)
