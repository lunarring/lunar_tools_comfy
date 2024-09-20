from collections import deque
import random
import time
import numpy as np
import lunar_tools as lt
import re

class LROSCSender:
    DEFAULT_PORT = 8003

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ip_address": ("STRING", {"default": "127.0.0.1"}),
            },
            "optional": {
                "port": ("INT", {"default": LROSCSender.DEFAULT_PORT}),
            }
        }

    RETURN_TYPES = ("OSCSender", )  
    RETURN_NAMES = ("osc_sender", )  
    FUNCTION = "init_osc_sender"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/comms"

    def __init__(self):
        self.ip_address = None
        self.port = None

    def init_osc_sender(self, ip_address, port=DEFAULT_PORT):
        try:
            if self.ip_address != ip_address or self.port != port:
                sender = lt.OSCSender(ip_receiver=ip_address, port_receiver=port)
                self.port = port
                self.ip_address = ip_address
                
        except Exception as e:
            import pdb; pdb.set_trace()
        return ([sender])

class LROSCSendMessage:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "osc_sender": ("OSCSender", {}),
                "name": ("STRING", {"default": "/example"}),
                "value": ("FLOAT", {"default": 0.0}),
            }
        }

    RETURN_TYPES = ()  
    FUNCTION = "send_osc_message"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/comms"

    def __init__(self):
        pass

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    def send_osc_message(self, osc_sender, name, value):
        if osc_sender is None:
            return ()
        try:
            # print(f"LROSCSendMessage: Sending message '{name}' with value {value}")
            osc_sender.send_message(name, float(value))
        except Exception as e:
            import pdb; pdb.set_trace()
        return ()



class LRZMQSender:
    DEFAULT_PORT = 5556

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_address": ("STRING", {"default": "10.40.49.211"}),
            },
            "optional": {
                "image": ("IMAGE", {}),
                "port": ("INT", {"default": LRZMQSender.DEFAULT_PORT}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_zmq"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/comms"

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    def __init__(self):
        self.server = None

    def send_zmq(self, ip_address, port=DEFAULT_PORT, image=None):
        if self.server is None or self.server.address != f"tcp://{ip_address}:{port}":
            # print("ZMQ pre init")
            self.server = lt.ZMQPairEndpoint(is_server=True, ip=ip_address, port=port, jpeg_quality=100)
            # print("ZMQ post init")
        
        if image is not None:
            # print("ZMQ pre send")
            img_reply = self.server.send_img(image)
            # print("ZMQ post send")
        return ()

class LRZMQReceiver:
    DEFAULT_PORT = 5557

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ip_address": ("STRING", {"default": "192.168.1.79"}),
            },
            "optional": {
                "port": ("INT", {"default": LRZMQSender.DEFAULT_PORT}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("ZMQ Image", )
    FUNCTION = "receive_zmq"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/comms"

    @classmethod 
    def IS_CHANGED(cls, **inputs):
        return float("NaN")

    def __init__(self):
        self.server = None

    def receive_zmq(self, ip_address, port=DEFAULT_PORT):
        if self.server is None or self.server.address != f"tcp://{ip_address}:{port}":
            self.server = lt.ZMQPairEndpoint(is_server=True, ip=ip_address, port=port, jpeg_quality=100)
            # print("ZMQ post init")
            
        while True:
            img = self.server.get_img()
            if img is not None:
                img = [img]
                return (img)
            else:
                time.sleep(0.001)





# class LROSCReceiver:
#     @classmethod
#     def INPUT_TYPES(s):
#         return {
#             "required": {
#                 "ip_address": ("STRING", {"default": "127.0.0.1"}),
#                 "osc_address": ("STRING", {"default": "/env1"}),
#             }
#         }

#     RETURN_TYPES = ("FLOAT",)
#     FUNCTION = "receive_osc"
#     CATEGORY = "LunarRing/comms"

#     @classmethod 
    # def IS_CHANGED(self, ip_address, osc_address):
#         return True   

#     def __init__(self):
#         self.receiver = None

#     def receive_osc(self, ip_address, osc_address):
#         if self.receiver is None or self.receiver.ip != ip_address:
#             self.receiver = lt.OSCReceiver(ip_address)
        
#         values = self.receiver.get_all_values(osc_address)
#         if values:
#             return (values[-1],)
#         else:
#             return (0.0,)

# # Add custom API routes, using router
# from aiohttp import web
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")
