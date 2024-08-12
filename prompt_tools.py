import time
from openai import OpenAI
import lunar_tools as lt
import base64
from PIL import Image
import io
import os
import numpy as np
import hashlib


# Function to encode the image
def encode_image(image_array):
    
    image = Image.fromarray(image_array)
    
    # Step 3: Save the image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')  # Save the image in PNG f(or any format you prefer)
    buffer.seek(0)  # Move the buffer cursor to the beginning
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')



class GPT4Vision:
    def __init__(self):
        self.client = OpenAI()
        self.last_run_time = 0
        self.last_description = ""

    def run_with_interval(self, image, prompt, min_interval):
        current_time = time.time()
        if current_time - self.last_run_time >= min_interval:
            self.last_run_time = current_time
            return self.run(image, prompt)
        else:
            # print(f"Run function called too frequently. Please wait {min_interval - (current_time - self.last_run_time):.2f} seconds.")
            return None

    def run(self, image, prompt):
        base64_image = encode_image(image)
        print('waiting for GPT response...')
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.9,
            max_tokens=100,
        )
        description = response.choices[0].message.content
        print(f"GPT4Vision: {description}")
        self.last_description = description
        return description


class Mic2Text:
    def __init__(self):
        self.speech_detector = lt.Speech2Text()

    def start_recording(self):
        if not self.speech_detector.audio_recorder.is_recording:
            self.speech_detector.start_recording()

    def stop_recording(self):
        prompt_new = None
        if self.speech_detector.audio_recorder.is_recording:
            try:
                prompt_new = self.speech_detector.stop_recording()
                prompt_new = prompt_new.strip().lower()
                is_new_prompt_good = True
                if prompt_new is None or prompt_new in [".", ". .", "you"] or len(prompt_new) == 0:
                    is_new_prompt_good = False

                if is_new_prompt_good:
                    print(f"New prompt mic: {prompt_new}")
                else:
                    prompt_new = None

            except Exception as e:
                print(f"FAIL {e}")

        return prompt_new

