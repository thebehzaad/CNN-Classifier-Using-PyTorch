"""***********************************************************************************

                      PyTorch Implementation of CNN Classifier 

***********************************************************************************"""
#%% Importing Libraries

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from utils import get_config, train_read_and_process_image, SeriesDataset
from model import ConvNet 
from training_test import CNNTrainer, CNNTester

#%% Configuration

config=get_config()

#%% Loading Data

train_path = 'dataset/train'
X_image, y_label= train_read_and_process_image(train_path, config)
X_train, X_valid, y_train, y_valid=train_test_split(X_image,y_label)

X_train*=(1./255)
X_valid*=(1./255)

train_dataset=SeriesDataset(X_train, y_train, config['device'])
valid_dataset=SeriesDataset(X_valid, y_valid, config['device'])

train_loader = DataLoader(train_dataset, batch_size=config['training_batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=config['training_batch_size'], shuffle=True)

del X_image, y_label, X_train, X_valid, y_train, y_valid, train_dataset, valid_dataset

#test_loader=

#%% Training

CNN_model=ConvNet()
tr=CNNTrainer(CNN_model, train_loader, valid_loader, config)
tr.train_epochs()

#%% Test

tst=CNNTester(CNN_model, valid_loader, config)
accuracy=tst.val()
print('Accurcay: %f' % accuracy)
