from gym_novel_gridworlds2.contrib.polycraft.actions.craft import Craft
import numpy as np

class FireAwareCraft(Craft):
    def is_near_target(self, agent_entity):
        """
        Decides Whether agent is near a crafting table
        """
        direction = (0, 0)
        if agent_entity.facing == "NORTH":
            direction = (-1, 0)
        elif agent_entity.facing == "SOUTH":
            direction = (1, 0)
        elif agent_entity.facing == "EAST":
            direction = (0, 1)
        else:
            direction = (0, -1)

        self.temp_loc = tuple(np.add(agent_entity.loc, direction))
        objs = self.state.get_objects_at(self.temp_loc)
        if len(objs[0]) == 1:
            # see if the crafting table is on fire
            if objs[0][0].type == "crafting_table" and \
                    not getattr(objs[0][0], "on_fire", True):
                return True
            else:
                return False
