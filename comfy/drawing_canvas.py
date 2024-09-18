from ..drawing_canvas import DrawingCanvas

class LRDrawingCanvas:
    DEFAULT_HEIGHT = 600
    DEFAULT_WIDTH = 800
    DEFAULT_GPU_ID = 0
    DEFAULT_X = 0
    DEFAULT_Y = 0
    DEFAULT_COLOR_ANGLE = 0
    DEFAULT_MASK_RADIUS = 0.08
    DEFAULT_DRAWING_INTENSITY = 1
    DEFAULT_DECAY_RATE = 0.5

    def __init__(self):
        self.drawingCanvas = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
                "gpu_id": ("FLOAT", {
                    "default": cls.DEFAULT_GPU_ID, 
                    "min": 0,
                    "max": 4,
                    "step": 1,
                    "display": "number"
                }),
                "x": ("FLOAT", {
                    "default": cls.DEFAULT_X, 
                    "min": 0,
                    "max": 1,
                    "step": 0.001,
                    "defaultInput": True,
                    "display": "number"
                }),
                "y": ("FLOAT", {
                    "default": cls.DEFAULT_Y, 
                    "min": 0,
                    "max": 1,
                    "step": 0.001,
                    "defaultInput": True,
                    "display": "number"
                }),
                "color_angle": ("FLOAT", {
                    "default": cls.DEFAULT_COLOR_ANGLE, 
                    "min": 0,
                    "max": 360,
                    "step": 1,
                    "display": "number"
                }),                
                "mask_radius": ("FLOAT", {
                    "default": cls.DEFAULT_MASK_RADIUS, 
                    "min": 0.000001,
                    "max": 1,
                    "step": 0.001,
                    "display": "number"
                }),                
                "drawing_intensity": ("FLOAT", {
                    "default": cls.DEFAULT_DRAWING_INTENSITY, 
                    "min": 0,
                    "max": 5,
                    "step": 1,
                    "display": "number"
                }),                
                "decay_rate": ("FLOAT", {
                    "default": cls.DEFAULT_DECAY_RATE, 
                    "min": 0,
                    "max": 1,
                    "step": 1e-2,
                    "display": "number"
                }),                
            },
        }
    
    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("Canvas Image", )
    FUNCTION = "get_canvas"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/visual"
    
    def initialize_once(self, height, width, gpu_id):
        if self.drawingCanvas is None:
            if height is None:
                height = self.DEFAULT_HEIGHT
            if width is None:
                width = self.DEFAULT_WIDTH
            if gpu_id is None:
                gpu_id = self.DEFAULT_GPU_ID
            self.drawingCanvas = DrawingCanvas(int(height), int(width), int(gpu_id))

    def get_canvas(
        self,
        height=None,
        width=None,
        gpu_id=None,
        x=None,
        y=None,
        color_angle=None,
        mask_radius=None,
        drawing_intensity=None,
        decay_rate=None
    ):
        self.initialize_once(height, width, gpu_id)
        
        if decay_rate is None:
            decay_rate = self.DEFAULT_DECAY_RATE
        self.drawingCanvas.set_decay_rate(decay_rate)
        
        if x is None:
            x = self.DEFAULT_X
        if y is None:
            y = self.DEFAULT_Y
        if width is None:
            width = self.DEFAULT_WIDTH
        if height is None:
            height = self.DEFAULT_HEIGHT
        x *= width
        y *= height
        
        if color_angle is None:
            color_angle = self.DEFAULT_COLOR_ANGLE
        if mask_radius is None:
            mask_radius = self.DEFAULT_MASK_RADIUS
        if drawing_intensity is None:
            drawing_intensity = self.DEFAULT_DRAWING_INTENSITY

        # scale back mask radius
        mask_radius = min(width, height) * mask_radius
        
        canvas = self.drawingCanvas.update(int(x), int(y),
                                  color_angle=int(color_angle),
                                  mask_radius=int(mask_radius),
                                  drawing_intensity=drawing_intensity)
        
        return ([canvas])

