import lunar_tools as lt
import numpy as np

class LRWebCam:
    DEFAULT_CAM_ID = 0
    DEFAULT_HEIGHT = 576
    DEFAULT_WIDTH = 1024
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Webcam Image", )
    FUNCTION = "get_img"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"
    
    @classmethod 
    def IS_CHANGED(self, cam_id, height, width):
        """Run every time"""
        return float("nan")
    
    def __init__(self):
        self.parameters = ["height", "width", "cam_id"]
        self.cam = None
    
    def initialize_once(self, cam_id, height, width):
        if self.cam is None:
            self.cam_id = cam_id
            self.height = int(height)
            self.width = int(width)
            self.cam = lt.WebCam(cam_id=cam_id, shape_hw=(self.height, self.width))
        else:
            for param in self.parameters:
                if getattr(self, param) != locals()[param]:
                    print(f"{param} cannot change after initialization!")

    def get_img(self, cam_id, height, width):
        self.initialize_once(cam_id, height, width)
        
        img = self.cam.get_img()
        img = [img]
        return (img)
            
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cam_id": ("INT", {
                    "default": cls.DEFAULT_CAM_ID, 
                    "min": -1,
                    "max": 10,
                    "step": 1,
                    "display": "number"
                }),
                "height": ("FLOAT", {
                    "default": cls.DEFAULT_HEIGHT, 
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "width": ("FLOAT", {
                    "default": cls.DEFAULT_WIDTH, 
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
            },
        }




# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")

