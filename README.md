# Lunar Tools ComfyUI [BETA]
Lunar Tools is a comprehensive toolkit designed for real-time interactive AI installations. This repository contains 
* wrappers for ComfyUI for existing Lunar Tools
* special comfy utils that we find useful

# Installation
Simply clone this repository into your custom_nodes subfolder in your comfyUI installation.
```bash
git clone https://github.com/lunarring/lunar_tools_comfy
```
Next, navigate into the lunar_tools_comfy folder and install the requirements.
```bash
pip install -r requirements.txt
```
After restarting comfyUI, you should be able to see "Lunar Ring" after a rightclick/Add Node, or find the below nodes directly.

# Modules
## sources
* LR WebCam: Outputs the webcam feed as an image.
* LR IPWebcam: Captures image or video streams from an IP webcam.
* LR MidiController AkaiLPD8: Interfaces with the Akai LPD8 MIDI controller, sending floats and booleans.
* LR MidiController AkaiMidimix: Interfaces with the Akai Midimix MIDI controller, sending floats and booleans.
* LR SoundReactive: Outputs data reactive to sound inputs, useful for audio visualization. Using just the volume at the moment.
* LR PushButtonArduino: Interfaces with an Arduino push button, outputting button press data.
* LR Mic2Text: Converts microphone input into text using whisper API from OpenAI.

## visual
* LR RenderWindow: Renders graphical output, using OpenGL as backend. Tested on Ubuntu 22.04.

## comms
* LR OSCSender: Sends data using the OSC (Open Sound Control) protocol to networked devices
* LR ZMQSender: Sends data over a network using the ZMQ (ZeroMQ) messaging protocol.
* LR ZMQReceiver: Receives data over a network using the ZMQ messaging protocol.

## util
* LR ScaleVariable: Scales an incoming float to a specified output range. min/max input are the expected min/max for the incoming float, and min/max output the range after linear rescaling.
* LR NumberBuffer: Stores a buffer of numeric data for later use.
* LR DerivativeBuffer: Calculates and stores the derivative of buffered numeric data.
* LR DerivativeVariable: Computes the derivative of the incoming floats.
* LR DrawBufferImage: Draws and displays an image from incoming floats.
* LR EquationEvaluator: Evaluates mathematical expressions based on incoming data, e.g. of type a + b^2 + 3*c^4.
* LR MovingWindowCalculator: Computes statistics over a window of incoming data.
* LR RandomUniformVariableGenerator: Generates random variables uniformly distributed across a specified range.
