from collections import deque
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from PIL import Image

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



class ARCurve:
    def __init__(self, vmin, vmax, t1, t2, t3, t4):
        self.vmin = vmin
        self.vmax = vmax
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self.total_duration = t1 + t2 + t3 + t4
        self.start_time = time.time()

    def _compute_value(self, elapsed_time):
        # Determine the value based on the elapsed time within the cycle
        if elapsed_time <= self.t1:
            return self.vmin, False
        elif elapsed_time <= self.t1 + self.t2:
            return self.vmin + (self.vmax - self.vmin) * ((elapsed_time - self.t1) / self.t2), False
        elif elapsed_time <= self.t1 + self.t2 + self.t3:
            return self.vmax, False
        elif elapsed_time <= self.total_duration:
            return self.vmax - (self.vmax - self.vmin) * ((elapsed_time - self.t1 - self.t2 - self.t3) / self.t4), False
        else:
            self.reset_timer()
            return self.vmin, True
        
    def reset_timer(self):
        self.start_time = time.time()

    def return_value(self):
        current_time = time.time()
        elapsed_time = (current_time - self.start_time)
        return self._compute_value(elapsed_time)

    def get_curve_image(self):
        # Generate time values for the entire duration of the function
        time_values = np.linspace(0, self.total_duration, 500)
        function_values = [self._compute_value(t) for t in time_values]

        # Plotting
        fig, ax = plt.subplots()
        ax.plot(time_values, function_values)
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        ax.legend()

        # Convert plot to image (NumPy array)
        canvas = FigureCanvas(fig)
        buf = BytesIO()
        canvas.print_png(buf)
        buf.seek(0)
            
        # Open image with PIL and convert to a numpy array with RGB format
        image = Image.open(buf)
        image_np = np.array(image)
    
        buf.close()
        plt.close(fig)  # Close the figure to free memory
        return image_np