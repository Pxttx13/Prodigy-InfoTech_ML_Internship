import os
import time

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import cv2

import torch
from torch import nn, optim
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

DATASET_FILEPATH = Path('/kaggle/input/leapgestrecog/leapGestRecog/')
IMG_SIZE = 256
RANDOM_STATE = 42

import re

def gest_label_encoding(gest):
    code = re.findall('^\d*', gest)[0]
    return int(code)-1

def make_df_from_files(parent_dir_path=DATASET_FILEPATH):
    result = list()
    for person in os.listdir(parent_dir_path):
        person_path = parent_dir_path.joinpath(person)
        for gest in os.listdir(person_path):
            gest_path = person_path.joinpath(gest)
            for img in os.listdir(gest_path):
                result.append([img, gest_path.joinpath(img), gest_label_encoding(gest), person])
                    
    return result
gest_df = pd.DataFrame(make_df_from_files(), columns=('img', 'path', 'gest', 'person'))
gest_df.sample(5)
gest_df['gest'].unique()

class LeapGests(torch.utils.data.Dataset):
    def __init__(self, data, preprocessing=None):
        self.data = data
        self.preprocessing = preprocessing
        
        self.image_paths = self.data.iloc[:, 1]
        self.image_gest = self.data.iloc[:, 2]
        self.data_len = len(self.data.index)
        
    def __len__(self):
        return self.data_len
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)
        
        if self.preprocessing is not None:
            img = self.preprocessing(img)
        
        gest = self.image_gest[idx]

        return img, gest
train_tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                    transforms.Resize([IMG_SIZE, IMG_SIZE]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(30),
                                    transforms.ToTensor()])

valid_tfms = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])

batch_size = 64

train_dataset = LeapGests(gest_df, train_tfms)
valid_dataset = LeapGests(gest_df, valid_tfms)

train_loader = torch.utils.data.DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=1)

for img, lbl in train_loader:
    print(img.shape)
    print(lbl[0])
    plt.imshow(img[0].permute(1, 2, 0)[:,:,0], cmap='gray')
    break
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device  
    
!pip install torchsummary    
from torchsummary import summary
from torchvision import models
    
resnet50 = models.resnet50(pretrained=True)
for param in list(resnet50.parameters())[:]:
    param.requires_grad = False    
resnet50.fc = nn.Linear(2048, 10)
summary(resnet50.to(device), input_size=(3, 32, 32))
resnet50 = resnet50.to(device)

params_to_update = []
for name,param in resnet50.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = torch.optim.Adam(params_to_update, lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 10
epoch_losses = []

for epoch in range(epochs):
    
    running_loss = 0.0
    epoch_loss = []
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = resnet50(data)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        epoch_loss.append(loss.item())
        
    test_running_loss = 0
    test_epoch_loss = []
    for batch_idx, (data, labels) in enumerate(valid_loader):
        resnet50.eval()
        data = data.to(device)
        labels = labels.to(device outputs = resnet50(data)
        loss = F.cross_entropy(outputs, labels)
                
        test_running_loss += loss.item()
        test_epoch_loss.append(loss.item())
        
        
    print(f'Epoch {epoch+1}, loss: ', np.mean(epoch_loss), 'test loss:', np.mean(test_epoch_loss))
    epoch_losses.append(epoch_loss)
torch.save(resnet50, './gest_detection_model.pth')
with torch.no_grad():
    for i, data in enumerate(valid_loader, 3):
        images, labels = data[0].to(device), data[1].to(device)
        
        outputs = resnet50(images)
        plt.title(f'pred - {outputs[0].argmax()}, gt - {labels[0]}')
        plt.imshow(images[0].cpu().permute(1, 2, 0), cmap='gray')
        plt.show()
        if i>15:
            break                          
