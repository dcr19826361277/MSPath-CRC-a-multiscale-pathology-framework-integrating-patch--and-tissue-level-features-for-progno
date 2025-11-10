#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lifelines
lifelines.__version__


# In[2]:


import os
import time
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, Subset

from lifelines.utils import concordance_index


# # Dataset

# In[3]:


class TileDataset(Dataset):
    def __init__(self, img_dir, tissue, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.tissue = tissue
        self.transform = transform
        self.target_transform = target_transform
        
        tissue_labels = pd.read_csv('./tissue_labels10x.csv', index_col = 'tile')
        condition = tissue_labels['tissue'] == tissue
        tissue_labels_tissue_index = list(tissue_labels[condition].index)
        
        labelcsv = pd.read_csv('./survival_COAD_survival.csv', index_col = 'sample')
        
        imagelist = []
        labellist = []
        
        for file in tissue_labels_tissue_index:
            imagelist.append(file)
            
            OS = labelcsv.at[file[0:15], 'OS']
            OS_time = labelcsv.at[file[0:15], 'OS.time']
            labellist.append(torch.tensor([OS, OS_time]))
        
        self.imagelist = imagelist
        self.labellist = labellist
        
    def __len__(self):
        return len(self.imagelist)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imagelist[idx][0:23], self.imagelist[idx])
        image = Image.open(img_path)
        label = self.labellist[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# In[4]:


transform = transforms.Compose([
    transforms.ToTensor(),
])

ADI_Tile  = TileDataset('./randomsampling', 0, transform = transform)
BACK_Tile = TileDataset('./randomsampling', 1, transform = transform)
DEB_Tile  = TileDataset('./randomsampling', 2, transform = transform)
LYM_Tile  = TileDataset('./randomsampling', 3, transform = transform)
MUC_Tile  = TileDataset('./randomsampling', 4, transform = transform)
MUS_Tile  = TileDataset('./randomsampling', 5, transform = transform)
NORM_Tile = TileDataset('./randomsampling', 6, transform = transform)
STR_Tile  = TileDataset('./randomsampling', 7, transform = transform)
TUM_Tile  = TileDataset('./randomsampling', 8, transform = transform)


# ## Split dataset

# In[5]:


# 60% training, 20% validation, and 20% testing

def split_dataset(dataset, BS = 64):
    original_targets = [i[0] for i in dataset.labellist]
    train_val_index, test_index = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=0, stratify=original_targets)
    
    train_val_targets = []
    sort_train_val_index = sorted(train_val_index)
    for i in sort_train_val_index:
        train_val_targets.append(original_targets[i])
    
    train_index, val_index = train_test_split(sort_train_val_index, test_size=0.25, random_state=0, stratify=train_val_targets)
    
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)
    test_dataset = Subset(dataset, test_index)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    dataloader = [train_dataloader, val_dataloader, test_dataloader]
    
    return dataloader


# # Model

# In[6]:


import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
class DeepConvSurv(nn.Module):
    def __init__(self):
        super(DeepConvSurv, self).__init__()
        self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])

        self.fc1 = nn.Sequential(
            nn.Linear(960, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.mobilenet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        self.feature = x.detach()
        risks = self.fc2(x)
        return risks


def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)




# In[8]:


model = DeepConvSurv()
model.apply(init_weights)


# # Loss function

# In[9]:


class negative_log_partial_likelihood(nn.Module):
    def __init__(self):
        super(negative_log_partial_likelihood, self).__init__()
    
    def forward(self, risk, os, os_time, model = None, regularization = False, Lambda = 1e-05):
        # R_matrix
        batch_len = risk.shape[0]
        R_matrix = np.zeros([batch_len, batch_len], dtype=int)
        for i in range(batch_len):
            for j in range(batch_len):
                R_matrix[i,j] = os_time[j] >= os_time[i]
        R_matrix = torch.tensor(R_matrix, dtype = torch.float32)
        R_matrix = R_matrix.to('cuda')
        
        # exp_theta
        theta = risk
        exp_theta = torch.exp(theta)
    
        # negative_log_partial_likelihood
        loss = - torch.sum( (theta - torch.log(torch.sum( exp_theta*R_matrix ,dim=1)) ) * os.float() ) / torch.sum(os)
    
        # l1 regularization
        l1_reg = torch.zeros(1)
        if regularization == True:
            for param in model.parameters():
                l1_reg = l1_reg + torch.sum(torch.abs(param))
            return loss + Lambda * l1_reg
        
        return loss


# In[10]:


loss_fn = negative_log_partial_likelihood()
#print(loss_fn)


# # Optimizer

# In[11]:


params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
            
optimizer = optim.Adam(params_to_update, lr=1e-04)
#print(optimizer)


# # Training

# In[12]:


def train_loop(dataloader, model, loss_fn, optimizer, num_epochs = 10):
    
    since = time.time()
    
    c_index_history = []
    loss_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_c_index = 0.0
    
    model = model.to('cuda')
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        
        time.sleep(1)
        
        model = model.to('cuda')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                Dataloader = dataloader[0]
            else:
                model.eval()
                Dataloader = dataloader[1]   
                
            risk_all = None
            label_all = None
            running_loss = 0.0
            iteration = 0
        
            for inputs, labels in tqdm(Dataloader):
                iteration = iteration + 1
            
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            
                # zero the parameter gradients
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
            
                    if iteration == 1:
                        risk_all = outputs
                        label_all = labels
                    else:
                        risk_all = torch.cat([risk_all, outputs])
                        label_all = torch.cat([label_all, labels])
            
                    # loss
                    loss = loss_fn(risk = outputs, os = labels[:,0], os_time = labels[:,1])
            
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                running_loss = running_loss + loss.item() * torch.sum(labels[:,0]).item() 
            
            epoch_loss = running_loss / torch.sum(label_all[:,0]).item()
        
            OS_time = label_all[:,1].detach().cpu().numpy()
            HR = risk_all.detach().cpu().numpy()
            OS = label_all[:,0].detach().cpu().numpy()
        
            epoch_c_index = concordance_index(OS_time, -HR.reshape(-1), OS)
            print('{} Loss: {:.4f} C-index: {:.4f}'.format(phase, epoch_loss, epoch_c_index))
        
            if phase == 'val' and epoch_c_index > best_c_index:
                best_c_index = epoch_c_index
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                c_index_history.append(epoch_c_index)
                loss_history.append(epoch_loss)
                
            time.sleep(1)
            
        print('-' * 40)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val C-index: {:f}'.format(best_c_index))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    print('-' * 40)
    time.sleep(1)
    
    # test
    model.load_state_dict(best_model_wts)
    model.eval()
    
    risk_all = None
    label_all = None
    iteration = 0
    for inputs, labels in tqdm(dataloader[2]):
        iteration = iteration + 1
        
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        
        # zero the parameter gradients
        optimizer.zero_grad()
    
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        
            if iteration == 1:
                risk_all = outputs
                label_all = labels
            else:
                risk_all = torch.cat([risk_all, outputs])
                label_all = torch.cat([label_all, labels])
    
    OS_time = label_all[:,1].detach().cpu().numpy()
    HR = risk_all.detach().cpu().numpy()
    OS = label_all[:,0].detach().cpu().numpy()       
    epoch_c_index = concordance_index(OS_time, -HR.reshape(-1), OS)
    
    time.sleep(1)
    
    print('Test C-index: {:f}'.format(epoch_c_index))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_history, c_index_history


# ## model_ADI

# In[39]:


model = DeepConvSurv()
model.apply(init_weights)

ADI_Dataloader = split_dataset(ADI_Tile)

model_ADI, loss_history_ADI, c_index_history_ADI = train_loop(ADI_Dataloader, model, loss_fn, optimizer)

torch.save(model_ADI.state_dict(), './model_ADI_weights.pth')


# ## model_BACK

# In[40]:


model = DeepConvSurv()
model.apply(init_weights)

BACK_Dataloader = split_dataset(BACK_Tile)

model_BACK, loss_history_BACK, c_index_history_BACK = train_loop(BACK_Dataloader, model, loss_fn, optimizer)

torch.save(model_BACK.state_dict(), './model_BACK_weights.pth')


# ## model_DEB

# In[41]:


model = DeepConvSurv()
model.apply(init_weights)

DEB_Dataloader = split_dataset(DEB_Tile)

model_DEB, loss_history_DEB, c_index_history_DEB = train_loop(DEB_Dataloader, model, loss_fn, optimizer)

torch.save(model_DEB.state_dict(), './model_DEB_weights.pth')


# ## model_LYM

# In[42]:


model = DeepConvSurv()
model.apply(init_weights)

LYM_Dataloader = split_dataset(LYM_Tile)

model_LYM, loss_history_LYM, c_index_history_LYM = train_loop(LYM_Dataloader, model, loss_fn, optimizer)

torch.save(model_LYM.state_dict(), './model_LYM_weights.pth')


# ## model_MUC

# In[43]:


model = DeepConvSurv()
model.apply(init_weights)

MUC_Dataloader = split_dataset(MUC_Tile)

model_MUC, loss_history_MUC, c_index_history_MUC = train_loop(MUC_Dataloader, model, loss_fn, optimizer)

torch.save(model_MUC.state_dict(), './model_MUC_weights.pth')


# ## model_MUS

# In[44]:


model = DeepConvSurv()
model.apply(init_weights)

MUS_Dataloader = split_dataset(MUS_Tile)

model_MUS, loss_history_MUS, c_index_history_MUS = train_loop(MUS_Dataloader, model, loss_fn, optimizer)

torch.save(model_MUS.state_dict(), './model_MUS_weights.pth')


# ## model_NORM

# In[45]:


model = DeepConvSurv()
model.apply(init_weights)

NORM_Dataloader = split_dataset(NORM_Tile)

model_NORM, loss_history_NORM, c_index_history_NORM = train_loop(NORM_Dataloader, model, loss_fn, optimizer)

torch.save(model_NORM.state_dict(), './model_NORM_weights.pth')


# ## model_STR

# In[46]:


model = DeepConvSurv()
model.apply(init_weights)

STR_Dataloader = split_dataset(STR_Tile)

model_STR, loss_history_STR, c_index_history_STR = train_loop(STR_Dataloader, model, loss_fn, optimizer)

torch.save(model_STR.state_dict(), './model_STR_weights.pth')


# ## model_TUM

# In[47]:


model = DeepConvSurv()
model.apply(init_weights)

TUM_Dataloader = split_dataset(TUM_Tile)

model_TUM, loss_history_TUM, c_index_history_TUM = train_loop(TUM_Dataloader, model, loss_fn, optimizer)

torch.save(model_TUM.state_dict(), './model_TUM_weights.pth')


# # Extracting features

# In[48]:


model_list = [model_ADI, model_BACK, model_DEB, model_LYM, model_MUC, model_MUS, model_NORM, model_STR, model_TUM]

for i in range(len(model_list)):
    model_list[i] = model_list[i].to('cuda')
    model_list[i].eval()

tissue_labels = pd.read_csv('./tissue_labels10x.csv', index_col = 'tile')
labelcsv = pd.read_csv('./survival_COAD_survival.csv', index_col = 'sample')


# In[49]:


TCGA_COAD_RS_PATH = './randomsampling'
foldername = os.listdir(TCGA_COAD_RS_PATH)


# In[50]:


X = []
y = []
Pathology = []
TNL = []

for folder in tqdm(foldername):
    # Pathology
    Pathology.append(folder)
    
    # y
    OS = labelcsv.at[folder[0:15], 'OS']
    OS_time = labelcsv.at[folder[0:15], 'OS.time']
    y.append([OS, OS_time])
    
    # X
    tissue_list = [[], [], [], [], [], [], [], [], []]
    for file in os.listdir(os.path.join(TCGA_COAD_RS_PATH, folder)):
        t = tissue_labels.at[file, 'tissue']
        tissue_list[t].append(file)
    
    feature_list = []
    for i in range(len(tissue_list)):
        feature_list.append(torch.zeros((1, 32)).to('cuda'))
    
    tile_number_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    for i in range(len(tissue_list)):
        for j in range(len(tissue_list[i])):
            img_path = os.path.join(TCGA_COAD_RS_PATH, folder, tissue_list[i][j])
            image = Image.open(img_path)
            image_tensor = transform(image)
            image_tensor.unsqueeze_(0)
            image_tensor = image_tensor.to('cuda')
            
            model_list[i] = model_list[i].to('cuda')
            model_list[i].eval()
            
            with torch.no_grad():
                risk = model_list[i](image_tensor)
                feature = model_list[i].feature
            
            feature_list[i] = feature_list[i] + feature
            
        tile_number_list[i] = len(tissue_list[i])
        if len(tissue_list[i]) != 0:
            feature_list[i] = feature_list[i] / len(tissue_list[i])
        
    TNL.append(tile_number_list)
     
    FEATURE_LIST = []
    for i in range(len(feature_list)):
        for j in range(len(feature_list[i][0])):
            FEATURE_LIST.append(feature_list[i][0][j].item())
    
    X.append(FEATURE_LIST)

print(f'Pathology : {len(Pathology)}')
print(f'X : {len(X)}*{len(X[0])}')
print(f'y : {len(y)}*{len(y[0])}')
print(f'TNL : {len(TNL)}*{len(TNL[0])}')


# In[51]:


data = []
columns = []

for i in range(len(Pathology)):
    data.append([Pathology[i]] + y[i] + TNL[i] + X[i])

columns.append('pathology')

columns.append('OS')
columns.append('OS.time')

columns.append('ADI.tile')
columns.append('BACK.tile')
columns.append('DEB.tile')
columns.append('LYM.tile')
columns.append('MUC.tile')
columns.append('MUS.tile')
columns.append('NORM.tile')
columns.append('STR.tile')
columns.append('TUM.tile')

for tissue in ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']:
    for j in range(32):
        columns.append(tissue + '.feature' + str(j))
        
#print(f'{len(data)}*{len(columns)}')


# In[52]:


df = pd.DataFrame(data = data, columns = columns)

df.to_csv('./histopathological_features_10xmbv3.csv', index=False)


# In[ ]:




