from gym_novel_gridworlds2.contrib.polycraft.states import PolycraftState
from gym_novel_gridworlds2.state.dynamic import Dynamic
from gym_novel_gridworlds2.contrib.polycraft.objects import PolycraftEntity
from .pddl_utils import generate_obj_types



def generate_diarc_json_from_state(
        player_id: int, 
        state: PolycraftState, 
        dynamic: Dynamic, 
        failed_action: str, 
        success: bool
    ):
    entity: PolycraftEntity = state.get_entity_by_id(player_id)

    # item count
    # need to put everything in the list so that RL knows what elements are there
    inventory_dict = {}

    # initialize the dict with every known item, excepting for entities
    for item_name, item_type in dynamic.all_objects:
        if item_type not in ["agent", "trader", "pogoist"]:
            inventory_dict[item_name] = 0
    # fill in the actual counts
    for item_name, count in entity.inventory.items():
        inventory_dict[item_name] = count
    
    inventory_info = {
        "slots": [
            {
                "item": item_name,
                "count": count
            } for item_name, count in inventory_dict.items()
        ],
        "selectedItem": entity.selectedItem or "air"
    }

    # other info
    player_info = {
        "pos": entity.loc,
        "facing": entity.facing.lower()
    }
    room_coord = None
    for coord in state.room_coords:
        if tuple(entity.loc) in coord:
            room_coord = coord
            break

    map_info = {
        loc.replace(",17,", ","): obj['name'] \
            for loc, obj in state.get_map_rep_in_range([room_coord]).items() \
            if obj['name'] != "air" and obj['name'] != "minecraft:air"
    }
    return {
        "inventory": inventory_info,
        "player": player_info,
        "map": map_info,
        "action": failed_action,
        "actionSuccess": success,
        "failedAction": failed_action,
    }
