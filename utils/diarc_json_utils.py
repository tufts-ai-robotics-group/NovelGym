from gym_novel_gridworlds2.contrib.polycraft.states import PolycraftState
from gym_novel_gridworlds2.contrib.polycraft.objects import PolycraftEntity

def generate_diarc_json_from_state(player_id: int, state: PolycraftState, failed_action, success):
    entity: PolycraftEntity = state.get_entity_by_id(player_id)
    inventory_info = {
        "slots": [
            {
                "item": item_name,
                "count": count
            } for item_name, count in entity.inventory.items()
        ],
        "selectedItem": entity.selectedItem or "air"
    }
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
