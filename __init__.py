NODE_CLASS_MAPPINGS = {}
IMPORT_ERROR_MESSAGE = "Lunar Ring Tools: failed to import"

try:
    from .comfy.display_window import LRRenderer
    NODE_CLASS_MAPPINGS["LR RenderWindow"] = LRRenderer
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} display_window: {e}")

try:
    from .comfy.cam import LRWebCam
    NODE_CLASS_MAPPINGS["LR WebCam"] = LRWebCam
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} cam: {e}")

try:
    from .comfy.midi import LRMidiInputAkaiLPD8, LRMidiInputAkaiMidimix
    NODE_CLASS_MAPPINGS["LR MidiController AkaiLPD8"] = LRMidiInputAkaiLPD8
    NODE_CLASS_MAPPINGS["LR MidiController AkaiMidimix"] = LRMidiInputAkaiMidimix
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} midi: {e}")

try:
    from .comfy.comms import LROSCSender, LRZMQSender, LRZMQReceiver
    NODE_CLASS_MAPPINGS["LR OSCSender"] = LROSCSender
    NODE_CLASS_MAPPINGS["LR ZMQSender"] = LRZMQSender
    NODE_CLASS_MAPPINGS["LR ZMQReceiver"] = LRZMQReceiver
    # NODE_CLASS_MAPPINGS["LR OSCReceiver"] = LROSCReceiver
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} utils: {e}")

try:
    from .comfy.ip_webcam import LRIPWebcam
    NODE_CLASS_MAPPINGS["LR IPWebcam"] = LRIPWebcam
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} ip_webcam: {e}")

try:
    from .comfy.sound_reactive import LRSoundReactive
    NODE_CLASS_MAPPINGS["LR SoundReactive"] = LRSoundReactive
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} sound_reactive: {e}")

try:
    from .comfy.arduino_button import LRPushButtonArduino
    NODE_CLASS_MAPPINGS["LR PushButtonArduino"] = LRPushButtonArduino
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} push_button: {e}")

try:
    from .comfy.drawing_canvas import LRDrawingCanvas
    NODE_CLASS_MAPPINGS["LR DrawingCanvas"] = LRDrawingCanvas
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} drawing_canvas: {e}")
    
try:
    from .comfy.ghosting import LRGhostingGenerator
    NODE_CLASS_MAPPINGS["LR GhostingGenerator"] = LRGhostingGenerator
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} ghosting_generator: {e}")
    
try:
    from .comfy.prompt_tools import LRGPT4Vision, LRMic2Text
    NODE_CLASS_MAPPINGS["LR GPT4Vision"] = LRGPT4Vision
    NODE_CLASS_MAPPINGS["LR Mic2Text"] = LRMic2Text
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} prompt_tools: {e}")

    
try:
    from .comfy.utils import LRNumberBuffer, DrawBufferImage,  EquationEvaluator, MovingWindowCalculator, DerivativeBuffer, DerivativeVariable, RandomUniformVariableGenerator, LRScaleVariable, LRRecursiveAdd
    NODE_CLASS_MAPPINGS["LR ScaleVariable"] = LRScaleVariable
    NODE_CLASS_MAPPINGS["LR NumberBuffer"] = LRNumberBuffer
    NODE_CLASS_MAPPINGS["LR DerivativeBuffer"] = DerivativeBuffer
    NODE_CLASS_MAPPINGS["LR DerivativeVariable"] = DerivativeVariable
    NODE_CLASS_MAPPINGS["LR RandomUniformVariableGenerator"] = RandomUniformVariableGenerator
    NODE_CLASS_MAPPINGS["LR DrawBufferImage"] = DrawBufferImage
    NODE_CLASS_MAPPINGS["LR EquationEvaluator"] = EquationEvaluator
    NODE_CLASS_MAPPINGS["LR MovingWindowCalculator"] = MovingWindowCalculator
    NODE_CLASS_MAPPINGS["LR RecursiveAdd"] = LRRecursiveAdd
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} utils: {e}")

try:
    from .comfy.movie import LRMovieReader, LRMovieSaver
    NODE_CLASS_MAPPINGS["LR MovieReader"] = LRMovieReader
    NODE_CLASS_MAPPINGS["LR MovieSaver"] = LRMovieSaver
except Exception as e:
    print(f"{IMPORT_ERROR_MESSAGE} utils: {e}")
    
# try:
#     from .comfy.tmp_test import Teleport
#     NODE_CLASS_MAPPINGS["LR Teleport"] = Teleport
# except Exception as e:
#     print(f"{IMPORT_ERROR_MESSAGE} utils: {e}")