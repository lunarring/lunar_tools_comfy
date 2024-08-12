import lunar_tools as lt

class LRMidiInputAkaiLPD8:
    DEFAULT_BUTTON_NONTOGGLE = "pressed_once"

    RETURN_TYPES = ("BOOLEAN", ) * 8 + ("FLOAT", ) * 8
    RETURN_NAMES = tuple(f"{letter}{i}" for letter in "ABCD" for i in range(2)) + \
                   tuple(f"{letter}{i}" for letter in "EFGH" for i in range(2))
    
    FUNCTION = "update"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"
    
    @classmethod 
    def IS_CHANGED(self):
        """Run every time"""
        return float("nan")

    def __init__(self):
        self.akai_lpd8 = lt.MidiInput(device_name="akai_lpd8")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A0_toggle": ("BOOLEAN", {"default": True}),
                "A1_toggle": ("BOOLEAN", {"default": True}),
                "B0_toggle": ("BOOLEAN", {"default": True}),
                "B1_toggle": ("BOOLEAN", {"default": True}),
                "C0_toggle": ("BOOLEAN", {"default": True}),
                "C1_toggle": ("BOOLEAN", {"default": True}),
                "D0_toggle": ("BOOLEAN", {"default": True}),
                "D1_toggle": ("BOOLEAN", {"default": True}),
            },
        }

    def update(self, A0_toggle, A1_toggle, B0_toggle, B1_toggle, C0_toggle, C1_toggle, D0_toggle, D1_toggle):
        button_modes = {}
        for button, toggle in [("A0", A0_toggle), ("A1", A1_toggle), ("B0", B0_toggle), ("B1", B1_toggle),
                               ("C0", C0_toggle), ("C1", C1_toggle), ("D0", D0_toggle), ("D1", D1_toggle)]:
            button_modes[button] = "toggle" if toggle else self.DEFAULT_BUTTON_NONTOGGLE

        outputs = [
            self.akai_lpd8.get(f"{letter}{i}", button_mode=button_modes[f"{letter}{i}"], variable_name=f"comfy {letter}{i}")
            if letter in "ABCD" else self.akai_lpd8.get(f"{letter}{i}", variable_name=f"comfy {letter}{i}")
            for letter in "ABCDEFGH" for i in range(2)
        ]
        # print(outputs)
        return outputs


class LRMidiInputAkaiMidimix:
    DEFAULT_BUTTON_NONTOGGLE = "pressed_once"

    RETURN_TYPES = []
    RETURN_NAMES = []

    outputs = []
    for letter in "ABCDEFGH":
        for i in range(6):
            variable_name = f"{letter}{i}"
            if i in [3, 4]:
                RETURN_TYPES.append("BOOLEAN")
            else:
                RETURN_TYPES.append("FLOAT")
            RETURN_NAMES.append(variable_name)
            

    for letter in "ABCD":
        for i in range(2):
            RETURN_NAMES += (f"{letter}{i}",)
    for letter in "EFGH":
        for i in range(2, 6):
            RETURN_NAMES += (f"{letter}{i}",)
    
    FUNCTION = "update"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/sources"
    
    
    @classmethod 
    def IS_CHANGED(self):
        """Run every time"""
        return float("nan")

    def __init__(self):
        self.akai_midimix = lt.MidiInput(device_name="akai_midimix")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "A3_toggle": ("BOOLEAN", {"default": True}),
                "A4_toggle": ("BOOLEAN", {"default": True}),
                "B3_toggle": ("BOOLEAN", {"default": True}),
                "B4_toggle": ("BOOLEAN", {"default": True}),
                "C3_toggle": ("BOOLEAN", {"default": True}),
                "C4_toggle": ("BOOLEAN", {"default": True}),
                "D3_toggle": ("BOOLEAN", {"default": True}),
                "D4_toggle": ("BOOLEAN", {"default": True}),
                "E3_toggle": ("BOOLEAN", {"default": True}),
                "E4_toggle": ("BOOLEAN", {"default": True}),
                "F3_toggle": ("BOOLEAN", {"default": True}),
                "F4_toggle": ("BOOLEAN", {"default": True}),
                "G3_toggle": ("BOOLEAN", {"default": True}),
                "G4_toggle": ("BOOLEAN", {"default": True}),
                "H3_toggle": ("BOOLEAN", {"default": True}),
                "H4_toggle": ("BOOLEAN", {"default": True}),
            },
        }

    def update(self, A3_toggle, A4_toggle, B3_toggle, B4_toggle, C3_toggle, C4_toggle, D3_toggle, D4_toggle, 
               E3_toggle, E4_toggle, F3_toggle, F4_toggle, G3_toggle, G4_toggle, H3_toggle, H4_toggle):
        button_modes = {}
        for button, toggle in [("A3", A3_toggle), ("A4", A4_toggle), ("B3", B3_toggle), ("B4", B4_toggle),
                               ("C3", C3_toggle), ("C4", C4_toggle), ("D3", D3_toggle), ("D4", D4_toggle),
                               ("E3", E3_toggle), ("E4", E4_toggle), ("F3", F3_toggle), ("F4", F4_toggle),
                               ("G3", G3_toggle), ("G4", G4_toggle), ("H3", H3_toggle), ("H4", H4_toggle)]:
            button_modes[button] = "toggle" if toggle else self.DEFAULT_BUTTON_NONTOGGLE

        outputs = []
        for letter in "ABCDEFGH":
            for i in range(6):
                variable_name = f"comfy {letter}{i}"
                button_mode = button_modes.get(f"{letter}{i}", self.DEFAULT_BUTTON_NONTOGGLE)
                if i in [3, 4]:
                    output = self.akai_midimix.get(f"{letter}{i}", button_mode=button_mode, variable_name=variable_name)
                else:
                    output = self.akai_midimix.get(f"{letter}{i}", variable_name=variable_name)
                outputs.append(output)
        # print(len(outputs))
        # print(outputs)
        return outputs

# # Add custom API routes, using routers
# from server import PromptServer

# @PromptServer.instance.routes.get("/hello")
# async def get_hello(request):
#     return web.json_response("hello")


