from gym_novel_gridworlds2.contrib.polycraft.objects import PolycraftEntity, PolycraftObject
from gym_novel_gridworlds2.actions.action import PreconditionNotMetError
from ngw_extensions.objects.easy_oak_log import OakLog

class HardTree(OakLog):
    breakable = False
    breakable_holding = ["axe"]
