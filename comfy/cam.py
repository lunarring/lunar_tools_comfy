import lunar_tools as lt
import cv2
import numpy as np

class LRWebCam:
    DEFAULT_CAM_ID = 0
    DEFAULT_HEIGHT = 576
    DEFAULT_WIDTH = 1024
    DEFAULT_AUTOFOCUS = True
    DEFAULT_MANUALFOCUSVAL = 0
    DEFAULT_WIDTH = 1024
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("Webcam Image", )
    FUNCTION = "get_img"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"
    
    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    def __init__(self):
        self.parameters = ["height", "width", "cam_id"]
        self.cam = None
        self.autofocus = self.DEFAULT_AUTOFOCUS
        self.manualfocusval = self.DEFAULT_MANUALFOCUSVAL
        self.cam_focus = self.DEFAULT_MANUALFOCUSVAL
    
    def initialize_once(self, cam_id, height, width):
        if self.cam is None or self.height != int(height) or self.width != int(width):
            if self.cam is not None:
                self.cam.cam.release()
            self.cam_id = cam_id
            self.height = int(height)
            self.width = int(width)
            self.cam = lt.WebCam(cam_id=cam_id, shape_hw=(self.height, self.width))
            
            self.cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.DEFAULT_AUTOFOCUS else 0)
            self.cam.cam.set(cv2.CAP_PROP_FOCUS, self.DEFAULT_MANUALFOCUSVAL)
            
    def update_params(self, autofocus, cam_focus):
        if self.autofocus != autofocus:
            self.autofocus = autofocus
            self.cam.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)
        if self.cam_focus != cam_focus:
            self.cam_focus = cam_focus
            self.cam.cam.set(cv2.CAP_PROP_FOCUS, self.cam_focus)

    def get_img(self, cam_id, height, width, autofocus, cam_focus):
        self.initialize_once(cam_id, height, width)
        self.update_params(autofocus, cam_focus)
        
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
                "autofocus": ("BOOLEAN", {
                    "default": True,
                }),
                "manual focus value": ("FLOAT", {
                    "default": 0,
                    "min": 0,
                    "max": 255,
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

