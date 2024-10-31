from collections import deque
import random
import time
import numpy as np
import lunar_tools as lt
import re
import torch

def scale_variable(variable, min_input, max_input, min_output, max_output):
    """
    Scales the input variable from the input range [min_input, max_input] to the output range [min_output, max_output].

    Parameters:
    variable (float): The input variable to be scaled.
    min_input (float): The minimum value of the input range.
    max_input (float): The maximum value of the input range.
    min_output (float): The minimum value of the output range.
    max_output (float): The maximum value of the output range.

    Returns:
    float: The scaled variable.
    """
    # Clip the variable between min_input and max_input using np.clip
    variable = np.clip(variable, min_input, max_input)
    # Scale the variable between min_output and max_output
    scaled_variable = min_output + (variable - min_input) * (max_output - min_output) / (max_input - min_input)
    return scaled_variable



