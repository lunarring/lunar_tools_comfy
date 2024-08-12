from collections import deque
import random
import time
import numpy as np
import lunar_tools as lt
import re


class SimpleNumberBuffer:
    """
    A class used to manage a buffer of numerical values with optional normalization.

    Attributes
    ----------
    buffer_size : int
        The maximum size of the buffer.
    buffer : deque
        The buffer storing numerical values.
    default_return_value : int
        The default value to return when the buffer is empty.
    normalize : bool
        A flag indicating whether to normalize the buffer values.

    Methods
    -------
    append(value)
        Appends a value to the buffer.
    get_buffer()
        Returns the buffer as a numpy array, optionally normalized.
    get_last_value()
        Returns the last value in the buffer or the default return value if the buffer is empty.
    set_buffer_size(buffer_size)
        Sets a new buffer size and adjusts the buffer accordingly.
    set_normalize(normalize)
        Sets the normalization flag.
    """

    def __init__(self, buffer_size=500, normalize=False):
        """
        Initializes the SimpleNumberBuffer with a specified buffer size and normalization flag.

        Parameters
        ----------
        buffer_size : int, optional
            The maximum size of the buffer (default is 500).
        normalize : bool, optional
            A flag indicating whether to normalize the buffer values (default is False).
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        self.default_return_value = 0
        self.normalize = normalize

    def append(self, value):
        """
        Appends a value to the buffer.

        Parameters
        ----------
        value : float
            The numerical value to append to the buffer.
        """
        self.buffer.append(value)

    def get_buffer(self):
        """
        Returns the buffer as a numpy array, optionally normalized.

        Returns
        -------
        numpy.ndarray
            The buffer as a numpy array. If normalization is enabled, the values are scaled between 0 and 1.
        """
        buffer_array = np.array(self.buffer)
        if self.normalize:
            min_val = np.min(buffer_array)
            max_val = np.max(buffer_array)
            if min_val != max_val:
                buffer_array = (buffer_array - min_val) / (max_val - min_val)
            else:
                buffer_array = np.full_like(buffer_array, 0.5)
        return buffer_array

    def get_last_value(self):
        """
        Returns the last value in the buffer or the default return value if the buffer is empty.

        Returns
        -------
        float
            The last value in the buffer or the default return value if the buffer is empty.
        """
        return self.buffer[-1] if len(self.buffer) > 0 else self.default_return_value

    def set_buffer_size(self, buffer_size):
        """
        Sets a new buffer size and adjusts the buffer accordingly.

        Parameters
        ----------
        buffer_size : int
            The new maximum size of the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(self.buffer, maxlen=buffer_size)

    def set_normalize(self, normalize):
        """
        Sets the normalization flag.

        Parameters
        ----------
        normalize : bool
            A flag indicating whether to normalize the buffer values.
        """
        self.normalize = normalize


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


class NumpyArrayBuffer:
    def __init__(self, buffer_size=500, default_return_value=None):
        """
        Initializes the NumpyArrayBuffer with a specified buffer size.

        Parameters
        ----------
        buffer_size : int
            The maximum size of the buffer.
        default_return_value : numpy array, optional
            The default return value if the buffer is empty.
        """
        self.buffer_size = buffer_size
        self.default_return_value = default_return_value
        self.buffer = deque(maxlen=buffer_size)
        self.array_shape = None

    def append(self, array):
        """
        Appends a new numpy array to the buffer. Sets the array shape if not already set.

        Parameters
        ----------
        array : numpy array
            The numpy array to be appended to the buffer.
        """
        if self.array_shape is None:
            self.array_shape = array.shape
        if array.shape == self.array_shape:
            self.buffer.append(array)
        else:
            raise ValueError(f"Array shape {array.shape} does not match buffer shape {self.array_shape}")

    def get_last(self):
        """
        Returns the last numpy array in the buffer or the default return value if the buffer is empty.

        Returns
        -------
        numpy array
            The last numpy array in the buffer or the default return value if the buffer is empty.
        """
        return self.buffer[-1] if len(self.buffer) > 0 else self.default_return_value

    def set_buffer_size(self, buffer_size):
        """
        Sets a new buffer size and adjusts the buffer accordingly.

        Parameters
        ----------
        buffer_size : int
            The new maximum size of the buffer.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(self.buffer, maxlen=buffer_size)


