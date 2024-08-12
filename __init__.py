NODE_CLASS_MAPPINGS = {}
try:
    from .display_window import Renderer
    NODE_CLASS_MAPPINGS["LR RenderWindow"] = Renderer
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import display_window: {e}")

try:
    from .cam import LRWebCam
    NODE_CLASS_MAPPINGS["LR WebCam"] = LRWebCam
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import cam: {e}")

try:
    from .midi import LRMidiInputAkaiLPD8, LRMidiInputAkaiMidimix
    NODE_CLASS_MAPPINGS["LR MidiController AkaiLPD8"] = LRMidiInputAkaiLPD8
    NODE_CLASS_MAPPINGS["LR MidiController AkaiMidimix"] = LRMidiInputAkaiMidimix
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import midi: {e}")

try:
    from .comms import LROSCSender, LRZMQSender, LRZMQReceiver
    NODE_CLASS_MAPPINGS["LR OSCSender"] = LROSCSender
    NODE_CLASS_MAPPINGS["LR ZMQSender"] = LRZMQSender
    NODE_CLASS_MAPPINGS["LR ZMQReceiver"] = LRZMQReceiver
    # NODE_CLASS_MAPPINGS["LR OSCReceiver"] = LROSCReceiver
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import utils: {e}")

try:
    from .ip_webcam import LRIPWebcam
    NODE_CLASS_MAPPINGS["LR IPWebcam"] = LRIPWebcam
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import ip_webcam: {e}")

try:
    from .sound_reactive import LRSoundReactive
    NODE_CLASS_MAPPINGS["LR SoundReactive"] = LRSoundReactive
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import sound_reactive: {e}")

try:
    from .arduino_button import LRPushButtonArduino
    NODE_CLASS_MAPPINGS["LR PushButtonArduino"] = LRPushButtonArduino
except Exception as e:
    print(f"Lunar Ring Nodes: failed to import push_button: {e}")


# try:
#     from .drawing_canvas import LRDrawingCanvas
#     NODE_CLASS_MAPPINGS["LR DrawingCanvas"] = LRDrawingCanvas
# except Exception as e:
#     print(f"Lunar Ring Nodes: failed to import drawing_canvas: {e}")
    




# try:
#     from .ghosting import LRGhostingGenerator
#     NODE_CLASS_MAPPINGS["LR GhostingGenerator"] = LRGhostingGenerator
# except Exception as e:
#     print(f"Lunar Ring Nodes: failed to import ghosting_generator: {e}")
    


# try:
#     from .prompt_builder import LRGPT4Vision, LRMic2Text
#     # NODE_CLASS_MAPPINGS["LR PromptBuilder"] = LRPromptBuilder
#     NODE_CLASS_MAPPINGS["LR GPT4Vision"] = LRGPT4Vision
#     NODE_CLASS_MAPPINGS["LR Mic2Text"] = LRMic2Text
# except Exception as e:
#     print(f"Lunar Ring Nodes: failed to import prompt_builder, gpt4vision, or mic2text: {e}")

    
# try:
#     from .utils import ScaleVariable, NumberBuffer, DrawBufferImage,  EquationEvaluator, MovingWindowCalculator, DerivativeBuffer, DerivativeVariable
#     NODE_CLASS_MAPPINGS["LR ScaleVariable"] = ScaleVariable
#     NODE_CLASS_MAPPINGS["LR NumberBuffer"] = NumberBuffer
#     NODE_CLASS_MAPPINGS["LR DerivativeBuffer"] = DerivativeBuffer
#     NODE_CLASS_MAPPINGS["LR DerivativeVariable"] = DerivativeVariable
#     # NODE_CLASS_MAPPINGS["LR RandomVariableGenerator"] = RandomVariableGenerator
#     NODE_CLASS_MAPPINGS["LR DrawBufferImage"] = DrawBufferImage
#     NODE_CLASS_MAPPINGS["LR EquationEvaluator"] = EquationEvaluator
#     NODE_CLASS_MAPPINGS["LR MovingWindowCalculator"] = MovingWindowCalculator
# except Exception as e:
#     print(f"Lunar Ring Nodes: failed to import utils: {e}")