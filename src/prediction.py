from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from tqdm import tqdm
from matplotlib import cm as CM
from src.utils.all_utils import read_yaml, log
from src.model import MCNN

class prediction:
    def __init__(self, config_path, downsample):
        content = read_yaml(config_path)
        self.downsample = downsample

        log_dir = content['base']['log_dir']
        log_filename = content['base']['log_file']
        self.logfile = os.path.join('src', log_dir, log_filename)

        checkpoint_dir =os.path.join('src',content['base']['checkpoint'])
        checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[0])
        result_path = content['base']['output_data_path']
        density_map_path = content['base']['output_density_map']
        self.result_path = os.path.join('src', result_path)
        self.density_map_path = os.path.join(density_map_path)

        self.cuda = torch.cuda.is_available()

        self.model = MCNN()
        if self.cuda:
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        
        
    def get_custom_photo(self, image):
        if len(image.shape)==2: # expand grayscale image to three channel.
            image=image[:,:,np.newaxis]
            image=np.concatenate((image,image,image),2)
        ds_rows=int(image.shape[0]//self.downsample)
        ds_cols=int(image.shape[1]//self.downsample)
        img = cv2.resize(image,(ds_cols*self.downsample,ds_rows*self.downsample))
        img=img.transpose((2,0,1))
        img_tensor=torch.tensor(img,dtype=torch.float)
        return img_tensor

    def get_density_map(self, output):
        path = os.path.join(self.density_map_path, 'output.jpg')
        plt.imsave(path, output)
        log('Density map of your input has been saved in {}'.format(path), self.logfile)

    def predict_image(self, img, from_video=False):
        img_tensor = self.get_custom_photo(img)
        img_tensor = img_tensor.expand(1, img_tensor.shape[0],img_tensor.shape[1],img_tensor.shape[2])
        if self.cuda:
            img_tensor = img_tensor.cuda()
        output = self.model(img_tensor)
        output = output.detach().numpy()
        self.get_density_map(output)
        s = int(((output.sum())/100))
        if s < 200:
            s = s // 2
        print(s)
        text = "Crowd Density: {}".format(s)
        cv2.putText(img, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        if not from_video:
            cv2.imshow('feed', img)
            cv2.waitKey(0)
        return img, s 

    def predict_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            
            if ret == True:
                if frame.shape[0] >= 1024:
                    frame = cv2.resize(frame,(1024,1024))
                image, s = self.predict_image(frame, from_video=True)
                
                cv2.imshow('feed', image)
                if cv2.waitKey(25) & 0xFF==ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        return s

    
            
