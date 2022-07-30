from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random



class crowd_data(Dataset):
    def __init__(self,img_root,gt_dmap_root,gt_downsample=1):
        self.gt_downsample=gt_downsample
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        #self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        #self.transform1 = transforms.Compose([transforms.CenterCrop(10)])

        self.images = [file for file in os.listdir(img_root)]
        self.n_samples=len(self.images)
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self,index):
        assert index <= len(self), 'index range error'
        img_name=self.images[index]
        img=plt.imread(os.path.join(self.img_root,img_name))
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)
            
        gt_dmap=np.load(os.path.join(self.gt_dmap_root,img_name.replace('.jpg','.npy')))
        if self.gt_downsample > 1:
            ds_rows=int(img.shape[0]//self.gt_downsample)
            ds_cols=int(img.shape[1]//self.gt_downsample)
            img = cv2.resize(img,(ds_cols*self.gt_downsample,ds_rows*self.gt_downsample))
            img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
            gt_dmap=cv2.resize(gt_dmap,(ds_cols,ds_rows))
            gt_dmap=gt_dmap[np.newaxis,:,:]*self.gt_downsample*self.gt_downsample
            
        img_tensor=torch.tensor(img,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        
        return img_tensor,gt_dmap_tensor