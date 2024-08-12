import numpy as np
import cv2

class GhostingGenerator():
    def __init__(self, max_buffer_frames=300):
        self.list_frames = []
        self.max_buffer_frames = max_buffer_frames
        self.ghost_index = 10
        self.do_ghost_hue_rotation = False
        self.hue_rotation_angle = 180

    def set_ghost_index(self, ghost_index=10):
        if ghost_index >= self.max_buffer_frames:
            ghost_index = self.max_buffer_frames - 1
            
        self.ghost_index = ghost_index
        
    def set_ghost_hue_rotation(self, do_ghost_hue_rotation=False):
        self.do_ghost_hue_rotation = do_ghost_hue_rotation        
        
    # @exception_handler
    def process(self, img):
        
        # buffer is full, ready for ghosting
        if len(self.list_frames) > self.max_buffer_frames:
            img_ghost = self.list_frames[-self.ghost_index].copy()
            
            # do ghost hue rotation
            
            if self.do_ghost_hue_rotation:
                # convert the image to HSV
                img_hsv = cv2.cvtColor(img_ghost, cv2.COLOR_BGR2HSV).astype(float)
    
                # Rotate the hue
                # Hue is represented in OpenCV as a value from 0 to 180 instead of 0 to 360...
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + (self.hue_rotation_angle / 2)) % 180
    
                # clip the values to stay in valid range
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
                # convert the image back to BGR
                img_ghost = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            
            img = (img.astype(np.float32) + img_ghost.astype(np.float32))/2
            img = np.clip(img, 0, 255)
            
            img = img.astype(np.uint8)
            
            self.list_frames = self.list_frames[1:]
        
        self.list_frames.append(img.copy())

        return img

