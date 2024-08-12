import numpy as np
import torch
import colorsys

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
    
    
