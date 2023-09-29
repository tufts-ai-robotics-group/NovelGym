from utils.advanced_item_encoder import PlaceHolderItemEncoder

def test_item_encoder():
    item_encoder = PlaceHolderItemEncoder({"air": 0}, placeholder_count=1)
    assert item_encoder.item_list == {"air": 0, "__placeholder_0": 1}
    assert item_encoder.reverse_look_up_table == {0: "air", 1: "__placeholder_0"}
    assert item_encoder.get_id("air") == 0

    # add item    
    assert item_encoder.get_id("oak_log") == 1
    assert item_encoder.item_list == {"air": 0, "oak_log": 1}
    assert item_encoder.reverse_look_up_table == {0: "air", 1: "oak_log"}

    # add item which is beyond the placeholder
    assert item_encoder.get_id("plank") == 2
    assert item_encoder.item_list == {"air": 0, "oak_log": 1, "plank": 2}
    assert item_encoder.reverse_look_up_table == {0: "air", 1: "oak_log", 2: "plank"}

# TODO test it with lidar
