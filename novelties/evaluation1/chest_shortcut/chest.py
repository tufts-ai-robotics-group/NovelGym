from gym_novel_gridworlds2.contrib.polycraft.objects import PlasticChest as VanillaPlasticChest

# 26 0 17 4

class PlasticChest(VanillaPlasticChest):
    def __init__(self, type="plastic_chest", loc=(0, 0), state="block", mode="easy", **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.loc = loc  # update such that we update the 3D arr and add the item to it
        self.state = state  # two states: block and floating
        self.empty = False
        self.mode = mode
        
    @staticmethod
    def placement_reqs(map_state, loc):
        return True

    def acted_upon(self, action_name, agent):
        if action_name == "collect" and not self.empty:
            agent.add_to_inventory("block_of_titanium", 1)
            agent.add_to_inventory("diamond", 4)
            agent.add_to_inventory("rubber", 1)
            agent.add_to_inventory("stick", 4)
            self.empty = True
