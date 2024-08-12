import numpy as np
import torch
import colorsys

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
    def IS_CHANGED(self):
        return True    
            
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

class DrawingCanvas():
    def __init__(self, height, width, gpu_id):
        self.canvas = torch.zeros([height, width, 3], device=f'cuda:{gpu_id}')
        self.decay_rate = 0.5
        
        self.height = height
        self.width = width
        
        # define meshgrid for fast brush implementation
        Y, X = torch.meshgrid(torch.arange(self.height, device=f'cuda:{gpu_id}'), 
                              torch.arange(self.width, device=f'cuda:{gpu_id}'), indexing='ij')                
        self.X = X.float()
        self.Y = Y.float()
        
        self.noise_patch = torch.rand([height, width], device=f'cuda:{gpu_id}')
        
    def set_decay_rate(self,decay_rate):
        self.decay_rate = decay_rate
    
    def update(self, x, y, color_angle=0, mask_radius=10, drawing_intensity=1):
        canvas = self.canvas
        canvas = canvas * self.decay_rate
        
        # ensure coordinates are within image bounds
        x = np.clip(x, 0, self.width)
        y = np.clip(y, 0, self.height)
        
        # ensure mask_radius is within bounds
        mask_radius = np.clip(mask_radius, 1, 100)
        
        color_vec = angle_to_rgb(color_angle)
        color_vec = torch.from_numpy(np.array(color_vec)).float().cuda(canvas.device)
        
        patch = draw_circular_patch(self.Y,self.X,y,x, mask_radius)
        colors = patch.unsqueeze(2)*self.noise_patch.unsqueeze(2)*color_vec[None][None]

        # Add the color gradient to the image
        colors /= (colors.max() + 0.0001)
        canvas += colors * drawing_intensity * 255
        canvas = canvas.clamp(0, 255)
        
        self.canvas = canvas

        canvas_numpy = canvas.cpu().numpy()
        canvas_numpy = np.clip(canvas_numpy, 0, 255)
        canvas_numpy = canvas_numpy.astype(np.uint8)
        
        return canvas_numpy

def draw_circular_patch(Y,X,y,x, brush_size):
    # Calculate the distance from the center (x, y)
    distance = ((X - x) ** 2 + (Y - y) ** 2).float().sqrt()
    
    mask = distance > brush_size
    patch = brush_size - distance
    patch[mask] = 0
    # distance = 1 / (distance + 1e-3)
    # distance[distance < brush_size] = 0
    
    return patch    

def angle_to_rgb(angle):
    """
    Convert an angle in radians (0 to 2*pi) to an RGB color vector.
    
    Parameters:
        angle (float): Angle in radians, where 0 to 2*pi maps to 0 to 1 in the hue.

    Returns:
        tuple: RGB color as a 3-element tuple, each component in the range 0 to 1.
    """
    # Normalize the angle to a range from 0 to 1
    hue = angle / (2 * 3.141592653589793)
    # Set saturation and value to 1 for maximum intensity and brightness
    saturation = 0.9
    value = 1
    # Convert HSV to RGB
    return colorsys.hsv_to_rgb(hue, saturation, value)


                    
if __name__ == '__main__':
    height, width = (300,600)
    gpu_id = 0
    drawingCanvas = DrawingCanvas(height, width, gpu_id)
    
    x = 50
    y = 50
    color_angle = 0
    mask_radius = 10
    drawing_intensity = 1
    
    canvas = drawingCanvas.update(x,y,
                                  color_angle=color_angle,
                                  mask_radius=mask_radius,
                                  drawing_intensity=drawing_intensity)
    
    
