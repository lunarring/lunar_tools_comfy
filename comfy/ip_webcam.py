import numpy as np
import time
from ..ip_webcam import MultiIPWebcam

class LRIPWebcam:
    def __init__(self):
        self.cam = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ip_address": ("STRING", {"multiline": False, "default": "192.168.1.1"})
                }
            }
            
    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("IP Webcam Image", )
    FUNCTION = "get_img"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"

    def get_img(self, ip_address):
        self._try_init(ip_address)
        
        self.cam.refresh()
        img = self.cam.list_imgs[0].astype(np.uint8)
        img = [img]
        
        return (img)
    
    def _try_init(self, ip_address):
        if self.cam is None:
            self.ip_address = ip_address
            
            self.cam = MultiIPWebcam([ip_address])
            time.sleep(1)
        # else:
        #     for param in self.parameters:
        #         if getattr(self, param) != locals()[param]:
        #             print(f"{param} cannot change after initialization!")    
    
