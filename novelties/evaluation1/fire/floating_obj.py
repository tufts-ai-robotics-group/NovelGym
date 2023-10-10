from gym_novel_gridworlds2.contrib.polycraft.objects import PolycraftObject

import pygame, os

# src: https://minecraft.fandom.com/wiki/Water_Bucket
BUCKET_IMG = pygame.image.load(
    os.path.join(os.path.dirname(__file__), "water_bucket.png")
)
BUCKET_IMG = pygame.transform.scale(BUCKET_IMG, (20, 20))

class FloatingObj(PolycraftObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = "floating"

    def get_img(self):
        return BUCKET_IMG