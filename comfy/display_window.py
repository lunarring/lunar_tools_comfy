# import lunar_tools as lt
import torch
import numpy as np
# import lunar_tools as lt
from ..display_window import Renderer

class LRRenderer:
    DEFAULT_HEIGHT = 576
    DEFAULT_WIDTH = 1024
    DEFAULT_WINDOW_TITLE = "lunar_render_window"
    RETURN_TYPES = ()
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/visual"

    def __init__(self):
        self.renderer = None
        self.render_size = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "height": ("FLOAT", {
                    "default": s.DEFAULT_HEIGHT, 
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "display": "number"
                }),
                "width": ("FLOAT", {
                    "default": s.DEFAULT_WIDTH, 
                    "min": 0,
                    "max": 4096,
                    "step": 64,
                    "display": "number"
                }),
                "window_title": ("STRING", {
                    "default": s.DEFAULT_WINDOW_TITLE,
                }),
            },
        }

    def render(self, image=None, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, window_title=DEFAULT_WINDOW_TITLE):
        if image is None:
            return ()
        if self.renderer is None or height != self.render_size[0] or width != self.render_size[1]:
            self.render_size = (height, width)
            self.renderer = Renderer(width=int(width), height=int(height), window_title=window_title, backend='pygame')
        
        image = torch.from_numpy(image.copy())
        self.renderer.render(image)
        return ()


# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")
