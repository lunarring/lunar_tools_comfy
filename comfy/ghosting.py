from ..ghosting import GhostingGenerator

class LRGhostingGenerator:
    def __init__(self):
        self.ghosting_generator = GhostingGenerator()
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "ghost_index": ("INT", {
                    "default": 10, 
                    "min": 0,
                    "max": 300,
                    "step": 1,
                    "display": "number"
                }),
                "do_ghost_hue_rotation": ("BOOLEAN", {"default": False}),
            },
        }
            
    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("image with ghosts", )
    FUNCTION = "process"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"

    def process(
        self, 
        image, 
        ghost_index=None,
        do_ghost_hue_rotation=None
    ):
        if ghost_index is not None:
            self.ghosting_generator.set_ghost_index(ghost_index)
            
        if do_ghost_hue_rotation is not None:
            self.ghosting_generator.set_ghost_hue_rotation(do_ghost_hue_rotation)
        
        image = self.ghosting_generator.process(image)
        image = [image]
        return image

