'''
this is the Environment class.
it implements the following
1. converts the JSON to the observation. 
    a. Observation is the conical-LiDAR representation augmented with inventory and selected item.
2. generates the reward function from the existing plan and the PDDL domain file.
3. generates the action candidates for the RL to be used.
4. parses the JSON to get done and info.
5. all this is relayed to the DiscoverExecutor class for action-executor discovery.
'''

import json
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import get_cmap
from gym import spaces

from .env_reward_utils import PolycraftRewardGenerator, scan_tokens
from .env_condition_set import ConditionSet

class SimpleItemEncoder:
    class TooManyItemTypes(Exception):
        pass

    def __init__(self, item_list=None, initial_id=1, id_limit=0):
        self.curr_id = initial_id - 1
        self.item_list = {}
        self.reverse_look_up_table = {}
        self.id_limit = id_limit
        if item_list is not None:
            self.load_item_list(item_list)
    

    def load_item_list(self, item_list):
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
            self.load_item_list(item_list)
    
    def get_id(self, key: str):
        """
        Takes in a key, returns a list. Not thread-safe.
        """
        if key in self.item_list:
            return self.item_list[key]
        elif self.id_limit > 0 and self.curr_id + 1 >= self.id_limit:
            raise self.TooManyItemTypes(
                "Cannot add item \"" + key + "\" to the encoder because there are " +
                "too many types of items. Consider increasing the number of allowed item types."
            )
        else:
            self.curr_id += 1
            self.item_list[key] = self.curr_id
            self.reverse_look_up_table[self.curr_id] = key
            return self.curr_id
    
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
            serialized = json.dumps(self.item_list)
            f.write(serialized)


class Polycraftv2Env(object):
    def __init__(self, json_input: dict, items_lidar_disabled=[]) -> None:
        """
        The Env is instanciated using the first json input.
        """
        # encoder for automatically encoding new objects
        self.item_encoder = SimpleItemEncoder({"air": 0})
        self.max_item_type_count = self._encode_items(json_input['state'])

        # rep of beams
        self.num_beams = 8
        self.max_beam_range = 40

        # things to search for in lidar. only excludes disabled items
        self.items_lidar_disabled = items_lidar_disabled
        self.items_lidar = list(filter(lambda item: item not in self.items_lidar_disabled, self.item_encoder.item_list.keys()))
        self.items_id_lidar = {self.item_encoder.get_id(keys): lidar_item_idx for lidar_item_idx, keys in enumerate(self.items_lidar)}

        # maximum of number of possible items
        lidar_items_max_count = self.max_item_type_count - len(self.items_lidar_disabled)

        # limits
        low = np.array(
            [0] * (lidar_items_max_count * self.num_beams) + 
            [0] * self.max_item_type_count + 
            [0]
        )
        high = np.array(
            [self.max_beam_range] * (lidar_items_max_count * self.num_beams) + 
            [40] * self.max_item_type_count + 
            [0] # maximum 40 stick can be crafted (5 log -> 20 plank -> 40 stick)
        )
        self.observation_space = spaces.Box(low, high, dtype=int)

        # reward generator
        self.reward_generator = PolycraftRewardGenerator(
            pddl_domain=json_input['domain'],  # un-escape pddl string
            initial_state=self.get_state_for_evaluation(json_input['state']),
            failed_action_exp=json_input['state']['action'],
            item_encoder=self.item_encoder,
            plan=json_input.get('plan')
        )
        self.failed_action = json_input['state']['action'].split('(')[0]
        self.novel_action_set = json_input['novelActions']
        print("actions: ", self.reward_generator.actions.keys())
        print("Novel actions: ", self.novel_action_set)
    
    def init_info(self):
        """
        Returns the init info for the discover executor
        """
        return {
            "failed_action": self.failed_action,
            "action_set": self.reward_generator.action_name_set,
            "observation_space": self.observation_space.sample(),
            "novel_action_set": self.novel_action_set
        }
    
    def get_action_name(self, action_id: int):
        return self.reward_generator.action_name_set[action_id]
    
    def get_novel_action_name(self, action_id: int):
        # print ("novel actions = ",self.novel_action)
        return self.novel_action[action_id]
    
    def check_if_effects_met(self, new_state_json: dict) -> bool:
        state = self.get_state_for_evaluation(new_state_json)
        return self.reward_generator.check_if_effect_met(state)
    
    def check_if_plannable_state_reached(self, new_state_json: dict) -> bool:
        state = self.get_state_for_evaluation(new_state_json)
        return True


    def generate_observation(self, json_input: dict) -> np.ndarray:
        """
        generates a numpy array representing the state from the json file.
        takes in json['state']
        """
        state_json = json_input
        # lidar beams
        world_map, min_coord, max_coord = self._generate_map(state_json)
        player_pos = np.array(state_json["player"]["pos"]) - min_coord
        sensor_result = self._lidar_sensors(tuple(player_pos), state_json['player']['facing'], world_map).reshape(-1)
        # inventory
        inventory_result = self._generate_inventory(state_json)
        # selected item
        selected_item = self._get_selected_item(state_json)

        return np.concatenate((sensor_result, inventory_result, [selected_item]))


    def _encode_items(self, json_data):
        """
        Run through the json and loads items into the list. Returns the number of items.
        Used to know how many items to expect so we can instantiate the array accordingly.
        """
        # This is a dry run of some functions to make sure the all items are included.
        # Nothing will be returned. The same function will be run again when generate_observation is called.
        self._generate_map(json_data)
        
        for slot in json_data['inventory']['slots']:
            self.item_encoder.get_id(slot['item'])

        # prevent new items from being encoded since we made the 
        #     assumption that no new items will be discovered after the first run.
        self.item_encoder.id_limit = len(self.item_encoder.item_list)
        return self.item_encoder.id_limit


    #################################################################
    # Util generate a map
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
    # Util generate a lidar reading
    #################################################################
    def _lidar_sensors(self, player_pos: Tuple[int, int], player_facing: str, world_map: np.ndarray) -> np.ndarray:
        '''
        Send 8 beams of 45 degrees each from 0 to 360 degrees
        Return the euclidean distances of the objects that strike the LiDAR.
        Assume that the occlusions dont hold true.
        '''

        direction_radian = {'north': np.pi, 'south': 0, 'west': 3 * np.pi / 2, 'east': np.pi / 2}

        # Shoot beams in 360 degrees in front of agent
        angles_list = np.linspace(direction_radian[player_facing] - np.pi,
                                  direction_radian[player_facing] + np.pi,
                                  self.num_beams + 1)[:-1]  # 0 and 360 degree is same, so removing 360

        lidar_signals = np.zeros((len(self.items_id_lidar), len(angles_list)))
        x, y = player_pos
        for angle_idx, angle in enumerate(angles_list):
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
                    if lidar_signals[index, angle_idx] == 0:
                        lidar_signals[index, angle_idx] = beam_range
                    # break # dont break because we want the objects even if we have occlusions.
        return lidar_signals
    

    # def conical_lidar_sensors(self):
    #     '''
    #     Send 4 cones of 90 degrees each from 0 to 360 degrees
    #     Return the euclidean distances of the objects inside each cone.
    #     '''
    #     pass

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
        inventory_quantity_arr = np.zeros(self.max_item_type_count)

        for slot in inventory:
            item_id = self.item_encoder.get_id(slot['item'])
            inventory_quantity_arr[item_id - 1] += slot['count']
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
    
    
    def get_state_for_evaluation(self, json_input: dict) -> dict:
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
    # Util to render the map
    #################################################################
    def render(self, title=None, map=None, input=None, mode='human', hold=True):
        """
        Renders a map of the environment.
        """
        if map is None:
            map, min_coord, max_coord = self._generate_map(input)
        items_in_map = np.unique(map)
        
        # plot map
        color_map = "gist_ncar"
        if title is None:
            title = "NovelGridworld-Phase2-Test"
        plt.figure(title, figsize=(10, 6))
        plt.imshow(map, cmap=color_map, vmin=0, vmax=len(items_in_map))

        # plot agent
        agent_loc = np.array(input["player"]["pos"]) - min_coord
        agent_facing = input['player']['facing']
        if agent_loc is not None and agent_facing is not None:
            r, c = agent_loc
            x2, y2 = 0, 0
            if agent_facing == 'north':
                r += 0.4
                x2, y2 = 0, -0.01
            elif agent_facing == 'south':
                r -= 0.4
                x2, y2 = 0, 0.01
            elif agent_facing == 'west':
                c += 0.4
                x2, y2 = -0.01, 0
            elif agent_facing == 'east':
                c -= 0.4
                x2, y2 = 0.01, 0
            plt.arrow(c, r, x2, y2, head_width=0.8, head_length=0.8, facecolor='white', edgecolor="black")
        
        # titles
        plt.title('NORTH', fontsize=10)
        plt.xlabel('SOUTH')
        plt.ylabel('WEST')
        plt.text(max_coord[1] - min_coord[1] + 1, (max_coord[0] - min_coord[0] + 1) / 2, 'EAST', rotation=90)

        # debug info: max and min coord
        plt.text(0, max_coord[0] - min_coord[0] + 8,
             "min_x: %4d; max_x: %4d; \nmin_y: %4d; max_y: %4d;" % 
             (min_coord[0], max_coord[0], min_coord[1], max_coord[1])
        )

        # legend
        cmap = get_cmap(color_map)
        legend_elements = [Line2D([0], [0], marker="^", color='w', label='agent', markerfacecolor='w', markersize=8,
                                  markeredgewidth=2, markeredgecolor='k'),
                           Line2D([0], [0], color='w', label="INVENTORY:")]
        
        for index, id in enumerate(items_in_map):
            rgba = cmap(index / len(items_in_map))
            legend_elements.append(Line2D([0], [0], marker="s", color='w',
                                          label=str(int(id)) + ': ' + self.item_encoder.reverse_look_up(id),
                                          markerfacecolor=rgba, markersize=8))
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1.02), prop={'size': 9})  # x, y 1.55 1.02
        plt.tight_layout()
        if hold == True:
            plt.show()
