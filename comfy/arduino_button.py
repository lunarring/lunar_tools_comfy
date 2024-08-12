from ..arduino_button import PushButtonController

class LRPushButtonArduino:
    def __init__(self):
        self.push_button_ctrl = PushButtonController(idx_button1=2, idx_button2=3)
        
    @classmethod
    def INPUT_TYPES(s):
        return {}
    
    @classmethod 
    def IS_CHANGED(self):
        return float("nan")
            
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN")  
    RETURN_NAMES = ("Arduino Button 1", "Arduino Button 2")
    FUNCTION = "get_button_press"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"

    def get_button_press(self, dummy):
        button_state1, button_state2 = self.push_button_ctrl.button_get()
        
        # print(f'button_state2 {button_state2}')
        
        if button_state1 is None:
            button_state1 = False
            
        if button_state2 is None:
            button_state2 = False
                    
        
        return ([button_state1, button_state2])
