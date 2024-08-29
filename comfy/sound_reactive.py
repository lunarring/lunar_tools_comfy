from ..sound_reactive import SoundReactive

class LRSoundReactive:
    def __init__(self):
        self.sound_volume = SoundReactive()
        
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    @classmethod 
    def IS_CHANGED(self):
        return float("nan")
    RETURN_TYPES = ("FLOAT", )  
    RETURN_NAMES = ("Sound level", )
    FUNCTION = "get_sound_level"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"

    def get_sound_level(self):
        volume = self.sound_volume.get_last()
        
        return ([volume])
        


