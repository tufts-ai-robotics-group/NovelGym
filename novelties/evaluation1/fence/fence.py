import pygame
import os
from gym_novel_gridworlds2.contrib.polycraft.objects import PolycraftObject

FENCE_IMG = pygame.image.load(
    os.path.join(os.path.dirname(__file__), "fence.png")
)
FENCE_IMG = pygame.transform.scale(FENCE_IMG, (20, 20))

class Fence(PolycraftObject):
    breakable = True

    def get_img(self):
        return FENCE_IMG
