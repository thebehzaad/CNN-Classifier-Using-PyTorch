"""**************************************************************************************


**************************************************************************************"""
#%% Importing Libraries

import cv2
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

#%%

def get_config():
    config={'device': ("cuda" if torch.cuda.is_available() else "cpu"),
            'nrows': 300,       # Height
            'ncolumns': 400,    # Width
            'nchannels': 3,     # Number of channels
            'training_batch_size': 32,
            'num_of_training_epochs': 10,
            'gradient_clipping': 20,
            'learning_rate': 1e-4}    
    return config


def train_read_and_process_image(train_path, config): 
    good_namelist = [train_path+'/class 0/{}'.format(i) for i in os.listdir(train_path+'/class 0/')]  
    flare_namelist = [train_path+'/class 1/{}'.format(i) for i in os.listdir(train_path+'/class 1/')]  
    X_image=[]
    y_label=[]
    for image in good_namelist:
        X=cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR).astype(float), (config['ncolumns'],config['nrows']), interpolation=cv2.INTER_AREA)
        X=np.transpose(X,(2,0,1))
        X_image.append(X) 
        y_label.append(0)
    for image in flare_namelist:
        X=cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR).astype(float), (config['ncolumns'],config['nrows']), interpolation=cv2.INTER_AREA)
        X=np.transpose(X,(2,0,1))
        X_image.append(X)
        y_label.append(1)
    # Shuffling Images
    temp = list(zip(X_image, y_label))
    random.shuffle(temp)
    X_image, y_label = zip(*temp)
    X_image = np.array(X_image)
    y_label = np.array(y_label)
    return X_image,y_label


class SeriesDataset(Dataset):
    def __init__(self, images, labels, device):        
        self.images = torch.tensor(images,dtype=torch.float32)
        self.labels = torch.tensor(labels,dtype=torch.float32)
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].to(self.device),\
               self.labels[idx].to(self.device)