import tkinter as tk
from tkinter import *
from tkinter import filedialog
import argparse
import tkinter.font as font
import os
import matplotlib.pyplot as plt
import cv2

from src.utils.all_utils import read_yaml, log
from src.prediction import prediction


class module:
    def __init__(self):
        self.window = Tk()
        self.window.title("Croed Density Estimator")

        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        
        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))

        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        # window.geometry('880x600')
        self.window.configure(background='#ffffff')
        
        header = tk.Label(self.window, text="Crowd Density Estimator", width=65, height=2, fg="white", bg="#363e75",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)

        uploadImg = tk.Button(self.window, text='upload image', command=self.collectImage, fg="white", bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        uploadImg.place(x=80, y=120)

        uploadVideo = tk.Button(self.window, text='upload video', command=self.collectVideo, fg="white", bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        uploadVideo.place(x=600, y=120)

        lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
                        font=('times', 15))
        self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
                                activebackground="#e47911", font=('times', 15))
        self.message.place(x=220, y=220)
        lbl3.place(x=80, y=260)

        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
                           font=('times', 15))
        self.message.place(x=220, y=260)

        maxCrowd = tk.Label(self.window, text='Max Crowd Threshold', width=20, fg="white", bg="#363e75", height=2, font=('times', 15))
        maxCrowd.place(x=80, y=350)
        self.maxCrowd = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.maxCrowd.place(x=325, y=359)

        predictImg = tk.Button(self.window, text="Predict", command=self.makePrediction, fg="white", bg="#363e75",
                             width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '), cursor='hand2')
        predictImg.place(x=600, y=350)


        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
                               activebackground="#118ce1", font=('times', 15, 'bold'), cursor='hand2')
        quitWindow.place(x=650, y=510)

        self.image_path = None
        self.video_path = None

        self.window.mainloop()

    def collectImage(self):
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)
        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        logfile = os.path.join('src', log_dir, log_filename)

        file_types = (('jpg files', '*.jpg'), ('jpeg files', '*.jpeg'), ('png files', '*.png'))
        file_open = filedialog.askopenfilename(initialdir='/', title='Select an image file', 
                                                filetypes=file_types)
        self.image_path = file_open
        if self.image_path:
            notification = "Image uploaded"
            self.message.configure(text=notification)
            log('input image uploaded', logfile)
        else:
            notification = "upload an image."
            self.message.configure(text=notification)
        

        
    def collectVideo(self):
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)

        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        logfile = os.path.join('src', log_dir, log_filename)

        file_types = (('mp4 files', '*.mp4'),)
        file_open = filedialog.askopenfilename(initialdir='/', title='Select an video file', 
                                                filetypes=file_types)
        self.video_path = file_open
        if self.video_path:
            self.message.configure(text="Video uploaded")
            log('Video uploaded', logfile)
        else:
            self.message.configure(text="upload a video.")
        

    
    def makePrediction(self):
        self.message.configure(text='processing...')
        maxCrowdVal = self.maxCrowd.get()
        
        config_path = os.path.join('src','config','config.yaml')
        content = read_yaml(config_path)

        data_path = content['base']['input_data_path']
        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        logfile = os.path.join('src', log_dir, log_filename)
        downsample = content['base']['down_sample']

        if (not self.image_path) and (not self.video_path):
            self.message.configure(text= 'Please upload an image or video')

        elif self.image_path:
            image = plt.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            plt.imsave(os.path.join(data_path, 'input.jpg'), image)
            app = prediction(config_path, downsample)
            try:
                maxCrowdVal = int(maxCrowdVal)
                img, s = app.predict_image(image, max_crowd=maxCrowdVal)
                plt.imsave(os.path.join(data_path, 'output.jpg'), img)
                notification = "Estimated Crowd Density: {}".format(s)
                self.message.configure(text=notification)
                self.image_path = None
            except Exception as e:
                print(e)
                self.message.configure(text='Please enter a valid integer value..')

        elif self.video_path:
            app = prediction(config_path, downsample)
            try:
                maxCrowdVal = int(maxCrowdVal)
                s = app.predict_video(self.video_path, max_crowd=maxCrowdVal)
                notification = "Estimated Crowd Density: {}".format(s)
                self.message.configure(text=notification)
                self.video_path = None
            except Exception as e:
                print(e)
                self.message.configure(text='Please enter a valid integer value..')
            



 
    def close_window(self):
        self.window.destroy()



if __name__ == '__main__':
    app = module()