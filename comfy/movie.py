import lunar_tools as lt
import numpy as np

class LRMovieReader:
    def __init__(self):
        self.movie_reader = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "file_path": ("STRING", {"multiline": False, "default": ""})
                }
            }
            
    @classmethod 
    def IS_CHANGED(cls):
        return True
                
    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("Image Next Frame", )
    FUNCTION = "get_next_frame"
    OUTPUT_NODE = False
    CATEGORY = "LunarRing/movie"

    def get_next_frame(self, file_path):
        if self.movie_reader is None or self.movie_reader.fp_movie != file_path:
            self.movie_reader = lt.MovieReader(file_path)
        
        frame = self.movie_reader.get_next_frame()
        if frame is not None:
            frame = frame.astype(np.uint8)
            return (frame,)
        else:
            return (None,)


class LRMovieSaver:
    def __init__(self):
        self.movie_saver = None
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"defaultInput": True}),
                "file_path": ("STRING", {"multiline": False, "default": ""}),
                "finalize": ("BOOLEAN", {"default": False}),
                "fps": ("INT", {"default": 24, "min": 1}),
                "crf": ("INT", {"default": 21, "min": 0, "max": 51}),
                "codec": ("STRING", {"default": "libx264"}),
                "preset": ("STRING", {"default": "fast"}),
                "pix_fmt": ("STRING", {"default": "yuv420p"}),
                "silent_ffmpeg": ("BOOLEAN", {"default": True})
            }
        }

    @classmethod
    def IS_CHANGED(cls):
        return True

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "process_frame"
    OUTPUT_NODE = True
    CATEGORY = "LunarRing/movie"

    def process_frame(self, file_path, fps, crf, codec, preset, pix_fmt, silent_ffmpeg, finalize, image=None):
        if finalize:
            if self.movie_saver:
                self.movie_saver.finalize()
                self.movie_saver = None
            return (None,)

        if not self.movie_saver or self.movie_saver.fp_out != file_path:
            self.movie_saver = lt.MovieSaver(
                fp_out=file_path,
                fps=fps,
                shape_hw=None,
                crf=crf,
                codec=codec,
                preset=preset,
                pix_fmt=pix_fmt,
                silent_ffmpeg=silent_ffmpeg
            )

        if image is not None:
            self.movie_saver.write_frame(image)

        return ()
