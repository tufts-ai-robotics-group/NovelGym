
def get_hinted_items(all_objects, hints, split_words=False):
    hinted_items = []
    hints = str(hints)
    for obj, value in all_objects.items():
        if split_words:
            obj_components = obj.split("_") + value.split("_")
        else:
            obj_components = [obj, value]
        for component in obj_components:
            if component in hints:
                hinted_items.append(obj)
                break
    return hinted_items


def get_hinted_actions(all_objects, hints, split_words=False):
    hinted_items = []
    hints = str(hints)
    for obj in all_objects:
        if split_words:
            obj_components = obj.split("_")
        else:
            obj_components = [obj]
        for component in obj_components:
            if component in hints:
                hinted_items.append(obj)
                break
    return hinted_items

def get_novel_action_indices(actions: list, novel_actions: list):
    novel_action_indices = []
    for i, action in enumerate(actions):
        if action in novel_actions:
            novel_action_indices.append(i)
    return novel_action_indices
