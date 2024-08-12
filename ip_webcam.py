import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import requests as r
import urllib
from threading import Thread
from datetime import datetime

def get_time_ms():
    return int(round(time.time() * 1000))


class GeneralThreader():
    def __init__(self, runfunc, sleeptime=0.5, idx=None):
        """
        runfunc is the function you want to run in the thread
        sleeptime is the delay between executions
        idx is an optional argument for passing a simple argument to the runfunc
        note to self: if more arguments shall be passed, please refactor and use that args= things.
        """
        
        self.t_refreshlast_slow = 0
        self.idx = idx
        self.ready = True
        self.runfunc = runfunc
        self.sleeptime = sleeptime
        self.thread = Thread(target = self.opti_thread, args = (1, ))
        self.thread.daemon = True
        self.thread.start()
        self.stop = False


    def opti_thread(self, arg):
        refreshrate_slow = 10
        while True:
        # while True and not self.stop:
            t_now = get_time_ms()

            t_diff_slow = t_now - self.t_refreshlast_slow
            if t_diff_slow > refreshrate_slow and self.ready:
                self.t_refreshlast_slow = t_now
                if self.idx is None:
                    self.runfunc()
                else:
                    self.runfunc(self.idx)

            time.sleep(self.sleeptime) 
    
    def stop(self):
        self.ready = False

def get_time(resolution=None):
    if resolution==None:
        resolution="second"
    if resolution == "day":
        t = time.strftime('%y%m%d', time.localtime())
    elif resolution == "minute":
        t = time.strftime('%y%m%d_%H%M', time.localtime())
    elif resolution == "second":
        t = time.strftime('%y%m%d_%H%M%S', time.localtime())
    elif resolution == "millisecond":
        t = time.strftime('%y%m%d_%H%M%S', time.localtime())
        t += "_"
        t += str("{:03d}".format(int(int(datetime.utcnow().strftime('%f'))/1000)))
    else:
        raise ValueError("bad resolution provided: %s" %resolution)
    return t

class IPWebcam(object):
    def __init__(self, ip_webcam_address='192.168.2.109:8080', shape_hw=(480,720)):
        self.ip_webcam_address = ip_webcam_address
        self.url = 'http://'+ip_webcam_address
        self.shape_hw = shape_hw
        self.t_last_img = 0
        self.t_last_request_sent = 0
        self.dt_mindist_requests = 50 #in ms! this limits the FPS!
        self.shift_colors = True
        self.do_mirror = False
        self.img = np.zeros((self.shape_hw[0], self.shape_hw[1], 3), dtype=np.uint8)
        self.img[:,:,0] = 255
        self.img_bad = np.copy(self.img)
        
        # start image grabbing as thread (recurring!)
        self.cam_threader = GeneralThreader(self.get_img, sleeptime=0.001)
        
        
    def re_init(self, args=None):
        try:
            # Get just one dummy image.
            imgResp = urllib.request.urlopen(self.url + '/shot.jpg')
            time.sleep(0.1)
            self.set_resolution(self.shape_hw) #directly enforce the resolution
            print("re_init success!")
        except Exception as e:
            print("re_init excepction: {}".format(e))


    def get_img(self):
        if 1000*(time.time() - self.t_last_request_sent) < self.dt_mindist_requests:
            print("get_img called but returning because too many requests per time...")
            return
        
        self.t_last_request_sent = time.time()
        try:
            # Get our image from the phone
            imgResp = urllib.request.urlopen(self.url + '/shot.jpg')
    
            # Convert our image to a numpy array so that we can work with it
            imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
    
            # Convert our image again but this time to opencv format
            img = cv2.imdecode(imgNp,-1)
            
            # reinit if time to last pic was too long!
            if time.time() - self.t_last_img > 2:
                self.re_init()
            if self.shift_colors:
                img = np.flip(img ,  axis=2)
            if self.do_mirror:
                img = np.flip(img, 1)
            
            # Get the last time when a good image was stored
            if img is not None and img.shape[0] > 10:
                self.t_last_img = time.time()
                if img.shape[0] != self.shape_hw[0] or img.shape[1] != self.shape_hw[1]:
                    print("WARNING!! img from cam had wrong shape: {}x{} instead of scheduled {}x{}".format(img.shape[0],img.shape[1],self.shape_hw[0],self.shape_hw[1]))
                    img = cv2.resize(img, self.shape_hw)
                    self.re_init()
                    
                self.img = img
                
        except Exception as e:
            print("FAIL: get_img: {}".format(e))
            time.sleep(1)
        

    def swap_camera(self, option: str ="on"):
        # swap the camera from thte back to the front
        # option: on / off
        return r.get(self.url+"/settings/ffc?set={}".format(option))


    def led(self, option: str ="off"):
        try:
            # turn on or off the flash light
            if option =="IPWEBCAMon":
                return r.get(self.url+"/enabletorch")
            return r.get(self.url+"/disabletorch")
        except:
            print("FAIL: led")

    def set_quality(self,option: int = 50):
        # Set the quality of the image
        # from 0 to 100
        if option > 100:
            option = 100
        if option < 0:
            option = 0
        return r.get(self.url +"/settings/quality?set={}".format(option))


    def set_resolution(self, shape_hw):
        try:
            return r.get("{}/settings/video_size?set={}x{}".format(self.url, shape_hw[1], shape_hw[0]))
        except:
            print("FAIL: get_image")
        

    def zoom(self, option: int = 0):
        if option < 0:
            option = 0
        if option > 100:
            option = 100
        return r.get(f"{self.url}/ptz?zoom={option}")


class MultiIPWebcam():
    
    def __init__(self, list_ips, shape_hw=(480,720), port=8080):
        self.shape_hw = shape_hw
        self.dt_max_alive = 3 # Timeout for camera update. If more than 3s no pic = DEATH
        print("INITIALISATION: {} IP adresses".format(len(list_ips)))
        self.list_cams = []
        self.list_imgs = []
        self.list_ips = []
        self.cam_alive = []
        
        self.img_bad = np.zeros((self.shape_hw[0], self.shape_hw[1], 3), dtype=np.uint8)
        self.img_bad[:,:,0] = 255
        
        for ip_addr in list_ips:
            if ip_addr in self.list_ips:
                print(f"WARNING! IP {ip_addr} is duplicate. This can cause crashes. Skipping...")
                continue
            self.list_cams.append(IPWebcam(shape_hw=shape_hw, ip_webcam_address='{}:{}'.format(ip_addr, port)))
            self.list_imgs.append(np.zeros((self.shape_hw[0], self.shape_hw[1], 3), dtype=np.uint8))        
            self.list_ips.append(ip_addr)        
            self.cam_alive.append(False)
            print("initialized ip: {}".format(ip_addr))
        
        
        # Set up the flexible stacking grid
        if len(self.list_cams) <= 4:
            grid_layout = (2, 2)
        elif len(self.list_cams) <= 6:
            grid_layout = (2, 3)
        elif len(self.list_cams) <= 9:
            grid_layout = (3, 3)
        elif len(self.list_cams) <= 12:
            grid_layout = (3, 4)
        elif len(self.list_cams) <= 16:
            grid_layout = (4, 4)
        else:
            grid_layout = (5, 5)
            
        self.stack_hw = []
        
        for id_cam in range(len(self.list_cams)):
            x = id_cam % grid_layout[1]
            y = int(id_cam / grid_layout[1])
            self.stack_hw.append((y,x))
        
        self.img_stacked = np.zeros((self.shape_hw[0]*grid_layout[0], self.shape_hw[1]*grid_layout[1], 3), dtype=np.uint8)
        
        print("INITIALISATION: finished!")
    
    def refresh(self):
        """
        Collects the images from all cams into: self.list_imgs
        and flags if cam is alive into: self.cam_alive
        """
        t_now = time.time()
        for id_cam in range(len(self.list_cams)):
            # Check if camera alive
            dt = t_now - self.list_cams[id_cam].t_last_img
            
            if dt < self.dt_max_alive:
                self.cam_alive[id_cam] = True
                self.list_imgs[id_cam] = self.list_cams[id_cam].img
            else:
                print("refresh: bad cam {}".format(self.list_cams[id_cam].ip_webcam_address))
                self.list_imgs[id_cam] = self.list_cams[id_cam].img_bad
                self.cam_alive[id_cam] = False
            
            
    def stack(self):
        """
        stacks images together into grid (e.g. for display halal): self.img_stacked
        DANGER FIX IMG SIZ
        """
        for id_cam in range(len(self.list_cams)):
            
            y0 = self.stack_hw[id_cam][0]*self.shape_hw[0]
            y1 = (1+self.stack_hw[id_cam][0])*self.shape_hw[0]
            x0 = self.stack_hw[id_cam][1]*self.shape_hw[1]
            x1 = (1+self.stack_hw[id_cam][1])*self.shape_hw[1]
            
            img_cam = self.list_imgs[id_cam] 
            
            if y1-y0 != img_cam.shape[0]:
                img_cam = self.img_bad
            elif x1-x0 != img_cam.shape[1]:
                img_cam = self.img_bad
            
            self.img_stacked[y0:y1,x0:x1,:] = img_cam
        
    def show_stacked(self):
        self.stack()
        plt.imshow(self.img_stacked)
        
        
    def get_alive_cam_id(self):
        good_cams_ids = np.where(self.cam_alive)[0]
        if len(good_cams_ids) > 0:
            cam_id = np.random.choice(good_cams_ids)
        else:
            cam_id = -1
        return cam_id
        
    def set_leds(self, list_activations=[], disable_all=False):
        """
        Turns the LED on for the phone with index id_cam, turns all other ones off.
        """
        if disable_all:
            for id_cam in range(len(self.list_cams)):
                self.list_cams[id_cam].led("off")    
                
        else:
            if len(list_activations) != len(self.list_cams):
                print("WARNING! len(list_activations)={} and len(self.list_cams)={}. BAD!!!!".format(len(list_activations),len(self.list_cams)))
            else:
                for id_cam in range(len(self.list_cams)):
                    if list_activations[id_cam]:
                        self.list_cams[id_cam].led("on")   
                    else:
                        self.list_cams[id_cam].led("off")   
                        

        
