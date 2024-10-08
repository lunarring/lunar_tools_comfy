import sounddevice as sd
import numpy as np
import threading

class SoundReactive:
    def __init__(self, sampling_rate=44100):
        self.sampling_rate = sampling_rate
        self.current_volume = 0
        self.max_volume_since_last_call = 0
        self.last_call_max_volume = 0
        self.lock = threading.Lock()
        self.running = True

        # Start the thread for audio processing
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.start()

    def _audio_callback(self, indata, frames, time, status):
        with self.lock:
            # Compute RMS (Root Mean Square) of the audio signal
            self.current_volume = np.linalg.norm(indata) * 10
            # Update the maximum volume since the last call
            if self.current_volume > self.max_volume_since_last_call:
                self.max_volume_since_last_call = self.current_volume

    def _process_audio(self):
        # Open the audio stream
        with sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.sampling_rate):
            while self.running:
                sd.sleep(int(1000))

    def get_last(self):
        with self.lock:
            # Return the max volume since the last call and reset
            max_volume = self.max_volume_since_last_call
            self.max_volume_since_last_call = 0
            return max_volume

    def stop(self):
        self.running = False
        self.thread.join()
