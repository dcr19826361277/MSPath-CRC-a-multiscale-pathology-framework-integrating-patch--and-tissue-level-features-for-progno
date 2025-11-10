#!/usr/bin/env python
# coding: utf-8



import openslide

import os
import shutil
import time
import gc
from tqdm import tqdm

from collections import Counter
from random import sample
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torchvision import transforms, models


# In[10]:


TCGA_COAD_PATH = './TCGA_COAD'
foldername = os.listdir(TCGA_COAD_PATH)


# In[11]:


pathology_list = pd.read_csv('./pathology_list.csv', index_col = 'pathology')
pathology_list_index = list(pathology_list.index)
#print(len(pathology_list_index))


# # Sampling

# In[12]:


os.mkdir('./randomsampling40x')

for i in tqdm(range(0, len(foldername))):
    print(foldername[i])
    
    since = time.time()
    
    if os.path.isdir(os.path.join(TCGA_COAD_PATH, foldername[i])):
        filename = os.listdir(os.path.join(TCGA_COAD_PATH, foldername[i]))
        for j in range(0, len(filename)):
            if filename[j][-3:len(filename[j])] == 'svs':
                print(filename[j])
                
                if filename[j][0:23] not in pathology_list_index:
                    print('Discard')
                    continue
                
                os.mkdir(os.path.join('./randomsampling40x', filename[j][0:23]))
                
                slide = openslide.OpenSlide(os.path.join(TCGA_COAD_PATH, foldername[i], filename[j]))
                
                try:
                    magnification = int(slide.properties['aperio.AppMag'])
                except:
                    magnification = 20
                
                [W, H] = slide.level_dimensions[0]
                w = int(W*(20/magnification))
                h = int(H*(20/magnification))

                total_tile_number = (w//224)*(h//224)
                print(f'Slide {i}, {magnification}X, {total_tile_number} tiles')
                
                # Random sampling 5%
                tile_number_list = list(range(1, total_tile_number+1))
                tile_number_list = sample(tile_number_list, total_tile_number//20)
                
                # Save tile
                for k in range(0, len(tile_number_list)):
                    h_tile_number = h//224
                    
                    x = tile_number_list[k] // h_tile_number
                    y = tile_number_list[k] %  h_tile_number
                    
                    factor = int(magnification/20) # 20X:1, 40X:2
                    if y>0:
                        location = (x*224*factor, (y-1)*224*factor)
                        crop = slide.read_region(location = location, level = 0, size = (224*factor, 224*factor))
                        crop = crop.convert('RGB')
                        crop = crop.resize((224, 224))
                    elif y == 0:
                        location = ((x-1)*224*factor, (h_tile_number-1)*224*factor)
                        crop = slide.read_region(location = location, level = 0, size = (224*factor, 224*factor))
                        crop = crop.convert('RGB')
                        crop = crop.resize((224, 224))
                        
                    CROP_TILE_PATH = os.path.join('./randomsampling40x', filename[j][0:23], filename[j][0:23]+'_{:06}.tif')
                    crop.save(CROP_TILE_PATH.format(tile_number_list[k]))
                
                del slide
                gc.collect()
    
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('-' * 60)


# # Classification

# ## Load model

# In[13]:


model = models.resnet50(weights=True)

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 9)
)

model.load_state_dict(torch.load('./resnet50_weights.pth'))
model = model.to('cuda')
model.eval()


# ## Transform

# In[14]:


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ## Predict

# In[15]:


TCGA_COAD_RS_PATH = './randomsampling40x'
RSfoldername = os.listdir(TCGA_COAD_RS_PATH)


# In[16]:


tilenamelist = []
tissuelist = []

since = time.time()

for i in tqdm(range(0, len(RSfoldername))):
    print(RSfoldername[i])
    
    filename = os.listdir(os.path.join(TCGA_COAD_RS_PATH, RSfoldername[i]))
    for j in range(0, len(filename)):
        tilenamelist.append(filename[j])
        
        img_path = os.path.join(TCGA_COAD_RS_PATH, RSfoldername[i], filename[j])
        image = Image.open(img_path)
        image_tensor = preprocess(image)
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.to('cuda')
        
        model = model.to('cuda')
        model.eval()
        
        with torch.no_grad():
            output = model(image_tensor)
            _, pred = torch.max(output, 1)
        
        pred = pred.item()
        
        tissuelist.append(pred)
        
    print('-' * 60)
    
time_elapsed = time.time() - since
print('Total time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


# In[17]:


store = {
    'tile': tilenamelist,
    'tissue': tissuelist,
}

store_df = pd.DataFrame(store)
store_df.to_csv('./tissue_labels40x.csv', index = False)






