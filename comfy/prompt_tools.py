from ..prompt_tools import Mic2Text, GPT4Vision

class LRMic2Text:
    def __init__(self):
        self.mic2text = Mic2Text()
        self.init_prompt = 'giant cat'
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "button_trigger": ("BOOLEAN", {"default": False, "defaultInput": True}),
            },
        }
    
    RETURN_TYPES = ("STRING", )  
    RETURN_NAMES = ("Prompt", )
    FUNCTION = "get_prompt"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"
    
    def get_prompt(self, button_trigger):
        prompt = self.init_prompt
        if button_trigger:
            self.mic2text.start_recording()
        else:
            prompt_transcript = self.mic2text.stop_recording()
            if prompt_transcript is not None:
                print('preparing transcript....')
                prompt = prompt_transcript
                self.init_prompt = prompt
        
        return ([prompt])


class LRGPT4Vision:
    def __init__(self):
        self.gpt4vision = GPT4Vision()
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "do_run": ("BOOLEAN", {"default": False, "defaultInput": True}),
                "prompt": ("STRING", {"default": "Describe the scene"}),
                "min_interval": ("FLOAT", {"default": 5.0}),
            },
        }
        
    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("description", )
    FUNCTION = "run"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/vision"
    
    def run(self, image, prompt, min_interval, do_run):
        # print("LRGPT4Vision: trying to run...")
        description = ""
        if do_run:
            description = self.gpt4vision.run_with_interval(image, prompt, min_interval)
        else:
            description = self.gpt4vision.last_description
        return ([description])
