from gym_novel_gridworlds2.contrib.polycraft.objects import PlacablePolycraftObject
from gym_novel_gridworlds2.object import Entity

class ObjOnFire(PlacablePolycraftObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_fire = True
    
    def acted_upon(self, action_name, agent: Entity):
        if action_name == "use" or action_name == "collect":
            if agent.selectedItem == "water_bucket":
                self.on_fire = False
                agent.inventory['water_bucket'] -= 1
                agent.add_to_inventory('bucket', 1)
                agent.selectedItem = "bucket"
        return super().acted_upon(action_name, agent)
