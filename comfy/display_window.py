# import lunar_tools as lt
import torch
import numpy as np
import lunar_tools as lt
import time
import PIL
# from ..display_window import Renderer
class LRRenderer:
    DEFAULT_HEIGHT = 576
    DEFAULT_WIDTH = 1024
    DEFAULT_WINDOW_TITLE = "lunar_render_window"
    RETURN_TYPES = ()
    FUNCTION = "render"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/visual"
    MAX_FPS = 30

    def __init__(self):
        self.renderer = None
        self.render_size = None
        self.last_exec_time = time.time()

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
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
                "cap_fps": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    def render(self, image=None, height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, window_title=DEFAULT_WINDOW_TITLE, cap_fps=False):
        current_time = time.time()
        elapsed_time = current_time - self.last_exec_time
        
        if cap_fps:
            sleep_time = max(0, (1.0 / self.MAX_FPS) - elapsed_time)
            time.sleep(sleep_time)

        if image is None:
            return ()
        
        if isinstance(image, PIL.Image.Image):
            image = np.asarray(image)
            image = torch.from_numpy(image.copy())
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            if image.max() <= 1.0:
                image = image * 255
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            image = torch.from_numpy(image.copy())
        elif torch.is_tensor(image):
            if image.ndim == 2:
                image = image.unsqueeze(-1).expand(-1, -1, 3)
            if image.ndim == 2:
                image = image.unsqueeze(-1).expand(-1, -1, 3)
            if torch.max(image) <= 1.0:
                image = image * 255
            image = image.to(torch.uint8)
            image = image.squeeze(0)

        if self.renderer is None or height != self.render_size[0] or width != self.render_size[1]:
            self.render_size = (height, width)
            self.renderer = lt.Renderer(width=int(width), height=int(height), window_title=window_title)
        
        
        self.renderer.render(image)
        
        self.last_exec_time = time.time()
        
        return ()


# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")
