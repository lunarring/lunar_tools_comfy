from collections import deque
import random
import time
import numpy as np
import lunar_tools as lt
import re
import datetime

class LRScaleVariable:
    DEFAULT_MIN_INPUT = 0.0
    DEFAULT_MAX_INPUT = 1.0
    DEFAULT_MIN_OUTPUT = 0.0
    DEFAULT_MAX_OUTPUT = 1.0
    DEFAULT_VARIABLE = 1.0

    def __init__(self):
        pass

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable": ("FLOAT", {"defaultInput": True}),
            },
            "optional": {
                "input_minimum_leftside": ("FLOAT", {"default": cls.DEFAULT_MIN_INPUT, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "input_maximum_leftside": ("FLOAT", {"default": cls.DEFAULT_MAX_INPUT, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "output_minimum_rightside": ("FLOAT", {"default": cls.DEFAULT_MIN_OUTPUT, "min": -1000.0, "max": 1000.0, "step": 0.01}),
                "output_maximum_rightside": ("FLOAT", {"default": cls.DEFAULT_MAX_OUTPUT, "min": -1000.0, "max": 1000.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("rescaled",)
    FUNCTION = "scale"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/util"

    def scale(self, variable=None, input_minimum_leftside=None, input_maximum_leftside=None, output_minimum_rightside=None, output_maximum_rightside=None):
        """
        Scales the input variable from the input range [min_input, max_input] to the output range [min_output, max_output].

        Parameters:
        variable (float): The input variable to be scaled.
        input_minimum_leftside (float): The minimum value of the input range.
        input_maximum_leftside (float): The maximum value of the input range.
        output_minimum_rightside (float): The minimum value of the output range.
        output_maximum_rightside (float): The maximum value of the output range.

        Returns:
        float: The scaled variable.
        """
        if variable is None:
            variable = self.DEFAULT_VARIABLE
        if input_minimum_leftside is None:
            input_minimum_leftside = self.DEFAULT_MIN_INPUT
        if input_maximum_leftside is None:
            input_maximum_leftside = self.DEFAULT_MAX_INPUT
        if output_minimum_rightside is None:
            output_minimum_rightside = self.DEFAULT_MIN_OUTPUT
        if output_maximum_rightside is None:
            output_maximum_rightside = self.DEFAULT_MAX_OUTPUT
        return (lt.scale_variable(variable, input_minimum_leftside, input_maximum_leftside, output_minimum_rightside, output_maximum_rightside),)


class EquationEvaluator:
    def __init__(self):
        pass

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "defaultInput": True}),
                "equation": ("STRING", {"multiline": False, "default": "a + 2*b"}),
            },
            "optional": {
                "b": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "defaultInput": True}),
                "c": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "defaultInput": True}),
                "d": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01, "defaultInput": True}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "evaluate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/util"

    def evaluate(self, a=0, b=0, c=0, d=0, equation="a"):
        # Replace variable names with their values
        equation = equation.replace('a', str(a))
        equation = equation.replace('b', str(b))
        equation = equation.replace('c', str(c))
        equation = equation.replace('d', str(d))

        # Use regex to ensure only safe characters are in the equation
        if not re.match(r'^[\d\s\+\-\*/\(\)\.]+$', equation):
            raise ValueError("Invalid characters in equation. Only digits, spaces, and the characters + - * / ( ) . are allowed.")

        try:
            result = eval(equation)
            # print(result)
        except Exception as e:
            raise ValueError(f"Error evaluating equation: {str(e)}")

        return (float(result),)


class MovingWindowCalculator:
    DEFAULT_WINDOW_SIZE = 10
    DEFAULT_METRIC = 'mean'
    
    def __init__(self):
        self.window_size = self.DEFAULT_WINDOW_SIZE
        self.buffer = lt.SimpleNumberBuffer(buffer_size=self.window_size)

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable": ("FLOAT", {"defaultInput": True}),
                "window_size": ("FLOAT", {"default": cls.DEFAULT_WINDOW_SIZE, "min": 1.0, "max": 1000.0, "step": 1.0}),
                "metric": (["mean", "max", "min", "median", "std", "last_nonzero"],),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "calculate_metric"
    CATEGORY = "LunarRing/util"

    def calculate_metric(self, variable, window_size=None, metric=None):
        if window_size is None:
            window_size = self.DEFAULT_WINDOW_SIZE
        if metric is None:
            metric = self.DEFAULT_METRIC
        
        window_size = int(window_size)
        if window_size != self.window_size:
            self.window_size = window_size
            self.buffer.set_buffer_size(window_size)
        
        self.buffer.append(variable)
        
        buffer_array = self.buffer.get_buffer()
        if len(buffer_array) < window_size:
            return (0,)
        
        if metric == "mean":
            result = np.mean(buffer_array)
        elif metric == "max":
            result = np.max(buffer_array)
        elif metric == "min":
            result = np.min(buffer_array)
        elif metric == "median":
            result = np.median(buffer_array)
        elif metric == "std":
            result = np.std(buffer_array)
        elif metric == "last_nonzero":
            nonzero_indices = np.nonzero(buffer_array)[0]
            result = buffer_array[nonzero_indices[-1]] if nonzero_indices.size > 0 else 0
        else:
            raise ValueError(f"Invalid metric: {metric}. Only 'mean', 'max', 'min', 'median', 'std', 'last_nonzero' are allowed.")
        
        return (result,)



class LRNumberBuffer:
    DEFAULT_BUFFER_SIZE = 500

    def __init__(self):
        self.number_buffer = lt.SimpleNumberBuffer(self.DEFAULT_BUFFER_SIZE)

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "variable": ("FLOAT", {"defaultInput": True}),
                "buffer_size": ("INT", {"default": cls.DEFAULT_BUFFER_SIZE, "min": 1, "max": 1000, "step": 1}),
                "normalize": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("ARRAY", "FLOAT")
    RETURN_NAMES = ("buffer", "last_value")
    FUNCTION = "buffer_value"
    CATEGORY = "LunarRing/util"

    def buffer_value(self, variable=None, buffer_size=None, normalize=False):
        if variable is None:
            last_value = self.number_buffer.get_last_value()
            return (self.number_buffer.get_buffer(), last_value) 
        if buffer_size is not None:
            self.number_buffer.set_buffer_size(buffer_size)
        
        self.number_buffer.set_normalize(normalize)
        self.number_buffer.append(variable)
        buffer_array = self.number_buffer.get_buffer()
        last_value = self.number_buffer.get_last_value()
        return (buffer_array, last_value)


class DerivativeBuffer:
    def __init__(self):
        self.buffer = None

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "buffer": ("ARRAY", {"defaultInput": True}),
            },
            "optional": {
                "absolute_value": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("ARRAY",)
    RETURN_NAMES = ("derivative_buffer",)
    FUNCTION = "buffer_derivative"
    CATEGORY = "LunarRing/util"

    def buffer_derivative(self, buffer=None, absolute_value=False):
        if buffer is None:
            return (np.array(self.buffer),) 
        
        derivative_buffer = np.diff(buffer, prepend=buffer[0])
        if absolute_value:
            derivative_buffer = np.abs(derivative_buffer)
        self.buffer = derivative_buffer
        return (self.buffer,)



class DerivativeVariable:
    def __init__(self):
        self.last_value = None

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable": ("FLOAT", {"defaultInput": True}),
            },
            "optional": {
                "absolute_value": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("derivative",)
    FUNCTION = "get_derivative"
    CATEGORY = "LunarRing/util"

    def get_derivative(self, variable=None, absolute_value=False):
        if self.last_value is None:
            derivative = 0
        else:
            derivative = variable - self.last_value
            if absolute_value:
                derivative = abs(derivative)
        self.last_value = variable
        return (derivative,)


class RandomUniformVariableGenerator:
    DEFAULT_MIN_VALUE = 0.0
    DEFAULT_MAX_VALUE = 1.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "min_value": ("FLOAT", {"default": cls.DEFAULT_MIN_VALUE, "min": -10000.0, "max": 10000.0, "step": 0.00001}),
                "max_value": ("FLOAT", {"default": cls.DEFAULT_MAX_VALUE, "min": -10000.0, "max": 10000.0, "step": 0.00001}),
            }
        }
    
    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("random_variable",)
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/util"

    def generate(self, min_value=DEFAULT_MIN_VALUE, max_value=DEFAULT_MAX_VALUE):
        random_variable = random.uniform(min_value, max_value)  # Generate a random float between min_value and max_value
        return (random_variable,)


class CycleVariableGenerator:
    DEFAULT_MIN_VALUE = 0.0
    DEFAULT_MAX_VALUE = 1.0
    DEFAULT_STEP = 0.1

    def __init__(self):
        self.current_value = self.DEFAULT_MIN_VALUE
        self.direction = 1  # 1 for increasing, -1 for decreasing

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("FLOAT", {"default": cls.DEFAULT_MIN_VALUE, "min": -10000.0, "max": 10000.0, "step": 0.00001}),
                "max_value": ("FLOAT", {"default": cls.DEFAULT_MAX_VALUE, "min": -10000.0, "max": 10000.0, "step": 0.00001}),
                "step": ("FLOAT", {"default": cls.DEFAULT_STEP, "min": 0.00001, "max": 10000.0, "step": 0.00001}),
            }
        }
    
    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("cycled_variable",)
    FUNCTION = "generate"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/util"

    def generate(self, min_value=DEFAULT_MIN_VALUE, max_value=DEFAULT_MAX_VALUE, step=DEFAULT_STEP):
        if self.direction == 1:
            self.current_value += step
            if self.current_value >= max_value:
                self.current_value = max_value
                self.direction = -1
        else:
            self.current_value -= step
            if self.current_value <= min_value:
                self.current_value = min_value
                self.direction = 1

        return (self.current_value,)



class DrawBufferImage:
    DEFAULT_HEIGHT = 200
    DEFAULT_WIDTH = 300

    def __init__(self):
        self.shape_hw_vis = [self.DEFAULT_HEIGHT, self.DEFAULT_WIDTH]  # Default image size
        self.simple_number_buffer = []

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")   
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": ("FLOAT", {"default": 0.0, "defaultInput": True}),
                "buffer_size": ("FLOAT", {"default": cls.DEFAULT_WIDTH, "min": 1.0, "max": 1000.0, "step": 1.0}),
            },
            "optional": {
                "height": ("INT", {"default": cls.DEFAULT_HEIGHT, "min": 5, "max": 1000, "step": 1}),
                "use_fixed_min_max": ("BOOLEAN", {"default": False}),
                "fixed_min": ("FLOAT", {"default": 0.0, "step": 0.0001}),
                "fixed_max": ("FLOAT", {"default": 1.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("buffer_image",)
    CATEGORY = "LunarRing/util"
    FUNCTION = "draw"

    def draw(self, input_value=0.0, buffer_size=300.0, height=None, use_fixed_min_max=False, fixed_min=0.0, fixed_max=1.0):
        buffer_size = int(buffer_size)  # Cast buffer_size to int
        if height is not None:
            self.shape_hw_vis = (height, self.shape_hw_vis[1])
        
        self.simple_number_buffer.append(input_value)
        
        if len(self.simple_number_buffer) < 2:
            return (self.create_blank_image(),)

        buffer_length = len(self.simple_number_buffer)
        if buffer_length != buffer_size:
            self.shape_hw_vis = (self.shape_hw_vis[0], buffer_size)

        values = self.simple_number_buffer[-buffer_size:]

        if use_fixed_min_max:
            min_val = fixed_min
            max_val = fixed_max
        else:
            min_val, max_val = np.min(values), np.max(values)

        def normalize(values, min_val, max_val):
            if isinstance(values, list):
                values = np.array(values)
            if max_val - min_val == 0:
                return np.full(len(values), self.shape_hw_vis[0] // 2)
            else:
                values = (values - min_val) / (max_val - min_val) * (self.shape_hw_vis[0] - 1)
                values = values + 1  # Shift values to ensure min value is drawn
                return np.floor(values).astype(np.int16)

        values = normalize(values, min_val, max_val)
        values = self.shape_hw_vis[0] - values
        valid_indices = (0 <= values) & (values < self.shape_hw_vis[0])

        image = np.zeros((*self.shape_hw_vis, 3), dtype=np.uint8)
        image[values[valid_indices].astype(int), np.arange(len(values))[valid_indices], 1] = 255  # Green channel

        return (image,)

    def create_blank_image(self):
        return np.zeros((*self.shape_hw_vis, 3), dtype=np.uint8)



class LRRecursiveAdd:
    def __init__(self):
        pass

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")
     
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variable": ("FLOAT", {"defaultInput": True}),
                "variable_recursive": ("FLOAT", {"defaultInput": True}),
            },
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("summed",)
    FUNCTION = "add"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/util"

    def add(self, variable=None, variable_recursive=None):
        """
        Adds two variables. One of them being recursive originating from looped link.

        Returns:
        float: Sum.
        """
        
        if variable_recursive is None:
            sum_result = variable
        else:
            sum_result = variable + variable_recursive
        
        return [sum_result]

class LRDelayString:
    def __init__(self):
        self.init_val = "INIT"

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_recursive": ("STRING", {"defaultInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("delayed_string",)
    FUNCTION = "delay"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/util"

    def delay(self, input_recursive=None):
        """
        Delays the input string by returning it as is.

        Returns:
        str: The input string.
        """
        if input_recursive is None:
            output = self.init_val
        else:
            output = input_recursive
        return [output]


class LRSaveToFile:
    def __init__(self):
        self.file_path = None
        self.last_input = None  # Store the last input data

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_for_saving": ("FLOAT", {"defaultInput": True}),
                "file_name": ("STRING", {"defaultInput": False}),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save_to_file"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/util"

    def save_to_file(self, data_for_saving, file_name):
        """
        Saves the input data to a file with a timestamp only if it's different from the last input.

        Args:
            data_for_saving (str): The data to be saved.
            file_name (str): The name of the file to save the data in.

        Returns:
            None
        """
        if not isinstance(data_for_saving, (str, int, float)):
            raise ValueError("data_for_saving must be a string or a number.")
        
        if isinstance(data_for_saving, float):
            data_for_saving = str(data_for_saving)

        # Check if the current input is different from the last input
        if data_for_saving != self.last_input:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")[:-3]
            # print(f"Saving data to {file_name} at {timestamp}")
            with open(file_name, 'a') as f:
                f.write(f"{timestamp} {data_for_saving}\n")
            # Update the last input
            self.last_input = data_for_saving
        else:
            print("Input data is the same as the last input. Skipping save.")
        return []


import base64
from io import BytesIO
from PIL import Image

class LRShowImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "LunarRing/util"

    def notify(self, image):
        # Convert the input image array to a PIL image
        if isinstance(image, list) and len(image) > 0:
            image = Image.fromarray(image[0])

        # Convert the image to a base64 string
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the base64 image string for the frontend
        return {"ui": {"image": image_base64}, "result": (image_base64,)}

class LRARCurve:
    def __init__(self):
        self.vmin = 0
        self.vmax = 1
        self.t1 = 1
        self.t2 = 2
        self.t3 = 3
        self.t4 = 4
    
    @classmethod
    def INPUT_TYPES(self,s):
        return {
            "required": {
                "trigger": ("BOOLEAN"),
                "vmin": ("FLOAT", {"default": self.vmin}),
                "vmax": ("FLOAT", {"default": self.vmax}),
                "t1": ("FLOAT", {"default": self.t1}),
                "t2": ("FLOAT", {"default": self.t2}),
                "t3": ("FLOAT", {"default": self.t3}),
                "t4": ("FLOAT", {"default": self.t4}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "update"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)

    CATEGORY = "LunarRing/util"
    
    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")    

    def update(self, trigger, vmin=None, vmax=None, t1=None, t2=None, t3=None, t4=None):
        
        # # Convert the input image array to a PIL image
        # if isinstance(image, list) and len(image) > 0:
        #     image = Image.fromarray(image[0])

        # # Convert the image to a base64 string
        # buffered = BytesIO()
        # image.save(buffered, format="PNG")
        # image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # # Return the base64 image string for the frontend
        # return {"ui": {"image": image_base64}, "result": (image_base64,)}
        

# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")
