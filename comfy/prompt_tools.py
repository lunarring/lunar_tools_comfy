from ..prompt_tools import Mic2Text, GPT4Vision
import json
import time
import numpy as np

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


class LRJsonPromptReader:
    def __init__(self):
        self.json_data = None

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "file_path": ("STRING", {"multiline": False, "default": ""})
                }
            }

    RETURN_TYPES = ("DICT", )  
    RETURN_NAMES = ("json_prompt_data", )
    FUNCTION = "read_json"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/data"

    def read_json(self, file_path):
        if self.json_data is None or self.file_path != file_path:
            self.file_path = file_path
            with open(file_path, 'r') as f:
                self.json_data = json.load(f)
        if not self.verify_json_prompts(self.json_data):
            raise ValueError("Invalid JSON data: Each item must be a dictionary containing a 'prompt' field.")
        return (self.json_data, )

    def verify_json_prompts(self, json_data):
        """
        Verifies that the JSON data contains a list of dictionaries, each with a 'prompt' field.
        If there are other fields, they must be present in all dictionaries.

        Args:
            json_data (list): The JSON data to verify.

        Returns:
            bool: True if the JSON data is valid, False otherwise.
        """
        if not isinstance(json_data, list):
            return False

        required_field = 'prompt'
        optional_fields = set()

        for item in json_data:
            if not isinstance(item, dict) or required_field not in item:
                return False
            # Collect optional fields from the first item
            if not optional_fields:
                optional_fields = set(item.keys()) - {required_field}
            # Ensure all optional fields are present in each item
            if not optional_fields.issubset(item.keys()):
                return False

        return True


class LRJsonPromptScheduler:
    def __init__(self):
        self.json_data = None
        self.file_path = None
        self.current_index = 0
        self.last_update_time = None
        self.next_prompt = ""
        self.time_started = time.time()
        self.return_blended = False
    
    @classmethod 
    def IS_CHANGED(cls):
        return True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_prompt_data": ("DICT", {"defaultInput": True})
            },
            "optional": {
                "interval": ("FLOAT", {"default": 2.0, "min": 0.1}),
                "force_interval": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("Current Prompt", )
    FUNCTION = "get_current_prompt"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/data"

    def get_current_prompt(self, json_prompt_data, interval, force_interval):
        # time.sleep(0.2)
        if self.json_data is None or self.json_data != json_prompt_data:
            self.json_data = json_prompt_data
            self.current_index = 0
            self.last_update_time = time.time()
        # Check if there are "time" fields in the json data
        has_time_fields = any('time' in item for item in self.json_data)
        # If there are no "time" fields, set force_interval to True
        if not has_time_fields:
            force_interval = True

        current_time = time.time()
        if force_interval:
            fract_progress = (current_time - self.last_update_time) / interval
            if (current_time - self.last_update_time >= interval):
                self.current_index = (self.current_index + 1) % len(self.json_data)
        else:
            if has_time_fields:
                total_elapsed_time = current_time - self.time_started
                next_index = self.current_index + 1
                if next_index < len(self.json_data):
                    time_diff = self.json_data[next_index]['time'] - self.json_data[self.current_index]['time']
                    elapsed_time_this_round = current_time - self.last_update_time
                    fract_progress = elapsed_time_this_round / time_diff
                    if total_elapsed_time >= self.json_data[next_index]['time']:
                        self.current_index = next_index
                        self.last_update_time = current_time
                else:
                    fract_progress = 0.0
            else:
                raise ValueError("JSON data does not contain 'time' fields and force_interval is not set.")

        next_prompt = self.json_data[self.current_index]['prompt']
        if self.next_prompt != next_prompt:
            self.next_prompt = next_prompt
            self.last_update_time = current_time
            print(f"Changing to prompt: {next_prompt}")

        if not self.return_blended:
            return (next_prompt, )
        else:
            current_prompt = self.json_data[self.current_index]['prompt']
            if self.current_index + 1 < len(self.json_data):
                next_prompt = self.json_data[self.current_index + 1]['prompt']
            else:
                next_prompt = current_prompt
            fract_progress = np.clip(fract_progress, 0, 1)
            # print(f"Previous: {current_prompt}, Current: {next_prompt}, Fractional Progress: {fract_progress}")
            return (current_prompt, next_prompt, fract_progress)


class LRJsonPromptSchedulerBlended(LRJsonPromptScheduler):
    def __init__(self):
        super().__init__()
        self.return_blended = True

    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("current_prompt", "next_prompt", "fract_progress")
    FUNCTION = "get_current_prompt"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/data"


