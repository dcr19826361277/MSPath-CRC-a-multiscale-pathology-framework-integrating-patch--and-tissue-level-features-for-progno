#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Import required libraries for survival analysis and deep learning
import lifelines
print(f"Lifelines version: {lifelines.__version__}")

# In[2]:
import os
import time
import copy
from tqdm import tqdm  # Progress bar
import numpy as np
import pandas as pd
from PIL import Image

# Sklearn for dataset splitting
from sklearn.model_selection import train_test_split

# PyTorch for deep learning
import torch
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset

# Survival analysis metrics
from lifelines.utils import concordance_index


# # Dataset Definition
# This module defines a custom Dataset class for loading histopathological tile images
# and corresponding survival labels (OS: Overall Survival, OS.time: Survival Time)

# In[3]:
class TileDataset(Dataset):
    """
    Custom Dataset class for loading histopathological tile images and survival labels
    
    Args:
        img_dir (str): Directory containing tile images
        tissue (int): Tissue type identifier (0-8 corresponding to ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM)
        transform (torchvision.transforms): Image transformations to apply
        target_transform: Transformations for target labels
    """
    def __init__(self, img_dir, tissue, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.tissue = tissue
        self.transform = transform
        self.target_transform = target_transform
        
        # Load tissue labels CSV (maps tiles to tissue types)
        tissue_labels = pd.read_csv('./tissue_labels10x.csv', index_col='tile')
        # Filter tiles for the specified tissue type
        condition = tissue_labels['tissue'] == tissue
        tissue_labels_tissue_index = list(tissue_labels[condition].index)
        
        # Load survival data CSV (maps samples to OS and OS.time)
        labelcsv = pd.read_csv('./survival_COAD_survival.csv', index_col='sample')
        
        imagelist = []  # List to store tile image filenames
        labellist = []  # List to store survival labels [OS, OS.time]
        
        # Populate image and label lists
        for file in tissue_labels_tissue_index:
            imagelist.append(file)
            
            # Extract survival labels (OS: event indicator, OS.time: survival time)
            OS = labelcsv.at[file[0:15], 'OS']          # Overall Survival event (1=event occurred, 0=censored)
            OS_time = labelcsv.at[file[0:15], 'OS.time']# Overall Survival time in days
            labellist.append(torch.tensor([OS, OS_time]))
        
        self.imagelist = imagelist
        self.labellist = labellist
        
    def __len__(self):
        """Return total number of tiles in the dataset"""
        return len(self.imagelist)
    
    def __getitem__(self, idx):
        """
        Get image and corresponding label by index
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, label_tensor) where label is [OS, OS.time]
        """
        # Construct image path (folder structure: first 23 chars of filename)
        img_path = os.path.join(self.img_dir, self.imagelist[idx][0:23], self.imagelist[idx])
        image = Image.open(img_path)
        label = self.labellist[idx]
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


# In[4]:
# Define image transformation pipeline (convert to tensor only)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Create dataset instances for each tissue type (0-8)
ADI_Tile  = TileDataset('./randomsampling', 0, transform=transform)  # Adipose tissue
BACK_Tile = TileDataset('./randomsampling', 1, transform=transform) # Background
DEB_Tile  = TileDataset('./randomsampling', 2, transform=transform) # Debris
LYM_Tile  = TileDataset('./randomsampling', 3, transform=transform) # Lymphoid tissue
MUC_Tile  = TileDataset('./randomsampling', 4, transform=transform) # Mucus
MUS_Tile  = TileDataset('./randomsampling', 5, transform=transform) # Muscle tissue
NORM_Tile = TileDataset('./randomsampling', 6, transform=transform) # Normal tissue
STR_Tile  = TileDataset('./randomsampling', 7, transform=transform) # Stroma
TUM_Tile  = TileDataset('./randomsampling', 8, transform=transform) # Tumor tissue


# ## Dataset Splitting Function
# Splits dataset into training (60%), validation (20%), and testing (20%) sets
# using stratified sampling to maintain class balance (based on OS event status)

# In[5]:
def split_dataset(dataset, BS=64):
    """
    Split dataset into train (60%), validation (20%), and test (20%) sets with stratification
    
    Args:
        dataset (TileDataset): Input dataset to split
        BS (int): Batch size for DataLoaders
        
    Returns:
        list: [train_dataloader, val_dataloader, test_dataloader]
    """
    # Extract OS labels for stratification (ensure balanced event/censored distribution)
    original_targets = [i[0] for i in dataset.labellist]
    
    # First split: 80% train+validation, 20% test
    train_val_index, test_index = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.2, 
        random_state=0, 
        stratify=original_targets
    )
    
    # Extract labels for train+validation set
    train_val_targets = []
    sort_train_val_index = sorted(train_val_index)
    for i in sort_train_val_index:
        train_val_targets.append(original_targets[i])
    
    # Second split: split train+validation into 75% train, 25% validation (final 60%/20%)
    train_index, val_index = train_test_split(
        sort_train_val_index, 
        test_size=0.25, 
        random_state=0, 
        stratify=train_val_targets
    )
    
    # Create Subset instances for each split
    train_dataset = Subset(dataset, train_index)
    val_dataset = Subset(dataset, val_index)
    test_dataset = Subset(dataset, test_index)
    
    # Create DataLoaders with specified batch size
    train_dataloader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BS, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    dataloader = [train_dataloader, val_dataloader, test_dataloader]
    
    return dataloader


# # Model Definition
# DeepConvSurv model using MobileNetV3-Large as backbone for survival prediction
# Predicts risk scores for survival analysis using histopathological tile images

# In[6]:
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class DeepConvSurv(nn.Module):
    """
    DeepConvSurv model for survival analysis with convolutional backbone
    
    Architecture:
        - MobileNetV3-Large (pretrained) as feature extractor
        - Fully connected layers to map features to risk scores
        - Output: single risk score per tile
    """
    def __init__(self):
        super(DeepConvSurv, self).__init__()
        # Load pretrained MobileNetV3-Large and remove final classification layer
        self.mobilenet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
        self.mobilenet = nn.Sequential(*list(self.mobilenet.children())[:-1])  # Remove last layer

        # Fully connected layers for risk score prediction
        self.fc1 = nn.Sequential(
            nn.Linear(960, 32),    # MobileNetV3-Large output dim = 960
            nn.ReLU(),             # Activation function
            nn.Dropout(0.5)        # Dropout for regularization
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 1)       # Output single risk score
        )

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x (torch.Tensor): Input image tensor (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Risk scores (batch_size, 1)
        """
        # Extract features using MobileNetV3
        x = self.mobilenet(x)
        # Flatten features
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
        x = self.fc1(x)
        self.feature = x.detach()  # Store intermediate features for later extraction
        risks = self.fc2(x)
        return risks

def init_weights(m):
    """
    Kaiming weight initialization for convolutional and linear layers
    Batch normalization layers initialized with constant values
    
    Args:
        m (nn.Module): Model layer to initialize
    """
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

# Initialize model and apply weight initialization
model = DeepConvSurv()
model.apply(init_weights)


# # Loss Function
# Negative Log Partial Likelihood (NLL) loss for Cox Proportional Hazards model
# Implements the loss function used in survival analysis for right-censored data

# In[9]:
class negative_log_partial_likelihood(nn.Module):
    """
    Negative Log Partial Likelihood loss for Cox Proportional Hazards model
    
    This loss function is specifically designed for survival analysis with right-censored data
    """
    def __init__(self):
        super(negative_log_partial_likelihood, self).__init__()
    
    def forward(self, risk, os, os_time, model=None, regularization=False, Lambda=1e-05):
        """
        Compute negative log partial likelihood loss
        
        Args:
            risk (torch.Tensor): Predicted risk scores (batch_size, 1)
            os (torch.Tensor): Overall Survival event indicator (1=event, 0=censored)
            os_time (torch.Tensor): Survival time for each sample
            model (nn.Module): Model for L1 regularization (optional)
            regularization (bool): Whether to apply L1 regularization
            Lambda (float): Regularization strength
            
        Returns:
            torch.Tensor: Computed loss value
        """
        # Create risk set matrix R: R[i,j] = 1 if os_time[j] >= os_time[i], else 0
        batch_len = risk.shape[0]
        R_matrix = np.zeros([batch_len, batch_len], dtype=int)
        for i in range(batch_len):
            for j in range(batch_len):
                R_matrix[i,j] = os_time[j] >= os_time[i]
        R_matrix = torch.tensor(R_matrix, dtype=torch.float32).to('cuda')
        
        # Compute exponential of risk scores (theta)
        theta = risk
        exp_theta = torch.exp(theta)
    
        # Calculate negative log partial likelihood
        # Loss = -1/N * sum( (theta_i - log(sum(exp(theta_j) for j in R_i))) * os_i )
        # where R_i is the risk set for sample i, N is number of events
        loss = - torch.sum( (theta - torch.log(torch.sum(exp_theta * R_matrix, dim=1))) * os.float() ) / torch.sum(os)
    
        # Add L1 regularization if enabled
        l1_reg = torch.zeros(1)
        if regularization:
            for param in model.parameters():
                l1_reg = l1_reg + torch.sum(torch.abs(param))
            return loss + Lambda * l1_reg
        
        return loss

# Initialize loss function
loss_fn = negative_log_partial_likelihood()


# # Optimizer Setup
# Adam optimizer with learning rate 1e-4 for trainable parameters

# In[11]:
# Collect trainable parameters (requires_grad=True)
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
            
# Initialize Adam optimizer with specified learning rate
optimizer = optim.Adam(params_to_update, lr=1e-04)


# # Training Loop
# Complete training pipeline with train/validation/test phases
# Tracks Concordance Index (C-index) as primary metric for survival analysis

# In[12]:
def train_loop(dataloader, model, loss_fn, optimizer, num_epochs=10):
    """
    Complete training loop with train, validation, and test phases
    
    Args:
        dataloader (list): [train_dataloader, val_dataloader, test_dataloader]
        model (nn.Module): Model to train
        loss_fn (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of training epochs
        
    Returns:
        tuple: (best_model, loss_history, c_index_history)
    """
    since = time.time()  # Track training time
    
    # Track training history
    c_index_history = []  # Validation C-index per epoch
    loss_history = []     # Validation loss per epoch
    
    # Store best model weights (based on validation C-index)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_c_index = 0.0
    
    model = model.to('cuda')  # Move model to GPU
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        
        time.sleep(1)  # Short delay for readability
        
        # Training and validation phases
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                Dataloader = dataloader[0]
            else:
                model.eval()   # Set model to evaluation mode
                Dataloader = dataloader[1]   
                
            # Initialize variables to accumulate predictions and labels
            risk_all = None
            label_all = None
            running_loss = 0.0
            iteration = 0
        
            # Iterate over data batches
            for inputs, labels in tqdm(Dataloader):
                iteration += 1
            
                # Move data to GPU
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
            
                # Zero parameter gradients
                optimizer.zero_grad()
                
                # Forward pass with gradient computation only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
            
                    # Accumulate predictions and labels
                    if iteration == 1:
                        risk_all = outputs
                        label_all = labels
                    else:
                        risk_all = torch.cat([risk_all, outputs])
                        label_all = torch.cat([label_all, labels])
            
                    # Compute loss
                    loss = loss_fn(risk=outputs, os=labels[:,0], os_time=labels[:,1])
            
                    # Backward pass and optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # Update running loss (weighted by number of events)
                running_loss += loss.item() * torch.sum(labels[:,0]).item() 
            
            # Calculate epoch loss (average over number of events)
            epoch_loss = running_loss / torch.sum(label_all[:,0]).item()
        
            # Prepare data for C-index calculation
            OS_time = label_all[:,1].detach().cpu().numpy()  # Survival times
            HR = risk_all.detach().cpu().numpy()            # Predicted hazard ratios
            OS = label_all[:,0].detach().cpu().numpy()      # Event indicators
        
            # Calculate Concordance Index (higher = better survival prediction)
            # C-index ranges from 0.5 (random) to 1.0 (perfect prediction)
            epoch_c_index = concordance_index(OS_time, -HR.reshape(-1), OS)
            print(f'{phase} Loss: {epoch_loss:.4f} C-index: {epoch_c_index:.4f}')
        
            # Update best model if validation C-index improves
            if phase == 'val' and epoch_c_index > best_c_index:
                best_c_index = epoch_c_index
                best_model_wts = copy.deepcopy(model.state_dict())
            
            # Record validation metrics
            if phase == 'val':
                c_index_history.append(epoch_c_index)
                loss_history.append(epoch_loss)
                
            time.sleep(1)  # Short delay for readability
            
        print('-' * 40)  # Separator line
    
    # Calculate total training time
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Val C-index: {best_c_index:.6f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    print('-' * 40)
    time.sleep(1)
    
    # Test phase with best model
    model.load_state_dict(best_model_wts)
    model.eval()
    
    # Accumulate test predictions
    risk_all = None
    label_all = None
    iteration = 0
    for inputs, labels in tqdm(dataloader[2]):
        iteration += 1
        
        # Move data to GPU
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')
        
        # Zero gradients (not necessary for evaluation, but kept for consistency)
        optimizer.zero_grad()
    
        # Forward pass without gradient computation
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
        
            # Accumulate predictions and labels
            if iteration == 1:
                risk_all = outputs
                label_all = labels
            else:
                risk_all = torch.cat([risk_all, outputs])
                label_all = torch.cat([label_all, labels])
    
    # Calculate test C-index
    OS_time = label_all[:,1].detach().cpu().numpy()
    HR = risk_all.detach().cpu().numpy()
    OS = label_all[:,0].detach().cpu().numpy()       
    epoch_c_index = concordance_index(OS_time, -HR.reshape(-1), OS)
    
    time.sleep(1)
    
    print(f'Test C-index: {epoch_c_index:.6f}')
    
    # Return best model and training history
    return model, loss_history, c_index_history


# ## Train Models for Each Tissue Type
# Train separate DeepConvSurv models for each of the 9 tissue types
# Save model weights after training

# In[39]:
# Train model for ADI (Adipose) tissue
model = DeepConvSurv()
model.apply(init_weights)
ADI_Dataloader = split_dataset(ADI_Tile)
model_ADI, loss_history_ADI, c_index_history_ADI = train_loop(ADI_Dataloader, model, loss_fn, optimizer)
torch.save(model_ADI.state_dict(), './model_ADI_weights.pth')

# In[40]:
# Train model for BACK (Background) tissue
model = DeepConvSurv()
model.apply(init_weights)
BACK_Dataloader = split_dataset(BACK_Tile)
model_BACK, loss_history_BACK, c_index_history_BACK = train_loop(BACK_Dataloader, model, loss_fn, optimizer)
torch.save(model_BACK.state_dict(), './model_BACK_weights.pth')

# In[41]:
# Train model for DEB (Debris) tissue
model = DeepConvSurv()
model.apply(init_weights)
DEB_Dataloader = split_dataset(DEB_Tile)
model_DEB, loss_history_DEB, c_index_history_DEB = train_loop(DEB_Dataloader, model, loss_fn, optimizer)
torch.save(model_DEB.state_dict(), './model_DEB_weights.pth')

# In[42]:
# Train model for LYM (Lymphoid) tissue
model = DeepConvSurv()
model.apply(init_weights)
LYM_Dataloader = split_dataset(LYM_Tile)
model_LYM, loss_history_LYM, c_index_history_LYM = train_loop(LYM_Dataloader, model, loss_fn, optimizer)
torch.save(model_LYM.state_dict(), './model_LYM_weights.pth')

# In[43]:
# Train model for MUC (Mucus) tissue
model = DeepConvSurv()
model.apply(init_weights)
MUC_Dataloader = split_dataset(MUC_Tile)
model_MUC, loss_history_MUC, c_index_history_MUC = train_loop(MUC_Dataloader, model, loss_fn, optimizer)
torch.save(model_MUC.state_dict(), './model_MUC_weights.pth')

# In[44]:
# Train model for MUS (Muscle) tissue
model = DeepConvSurv()
model.apply(init_weights)
MUS_Dataloader = split_dataset(MUS_Tile)
model_MUS, loss_history_MUS, c_index_history_MUS = train_loop(MUS_Dataloader, model, loss_fn, optimizer)
torch.save(model_MUS.state_dict(), './model_MUS_weights.pth')

# In[45]:
# Train model for NORM (Normal) tissue
model = DeepConvSurv()
model.apply(init_weights)
NORM_Dataloader = split_dataset(NORM_Tile)
model_NORM, loss_history_NORM, c_index_history_NORM = train_loop(NORM_Dataloader, model, loss_fn, optimizer)
torch.save(model_NORM.state_dict(), './model_NORM_weights.pth')

# In[46]:
# Train model for STR (Stroma) tissue
model = DeepConvSurv()
model.apply(init_weights)
STR_Dataloader = split_dataset(STR_Tile)
model_STR, loss_history_STR, c_index_history_STR = train_loop(STR_Dataloader, model, loss_fn, optimizer)
torch.save(model_STR.state_dict(), './model_STR_weights.pth')

# In[47]:
# Train model for TUM (Tumor) tissue
model = DeepConvSurv()
model.apply(init_weights)
TUM_Dataloader = split_dataset(TUM_Tile)
model_TUM, loss_history_TUM, c_index_history_TUM = train_loop(TUM_Dataloader, model, loss_fn, optimizer)
torch.save(model_TUM.state_dict(), './model_TUM_weights.pth')


# # Feature Extraction
# Extract intermediate features (32-dim) from trained models for each tissue type
# Aggregate features at the patient level (average over tiles)

# In[48]:
# Create list of trained models for each tissue type
model_list = [model_ADI, model_BACK, model_DEB, model_LYM, model_MUC, model_MUS, model_NORM, model_STR, model_TUM]

# Move all models to GPU and set to evaluation mode
for i in range(len(model_list)):
    model_list[i] = model_list[i].to('cuda')
    model_list[i].eval()

# Reload tissue labels and survival data
tissue_labels = pd.read_csv('./tissue_labels10x.csv', index_col='tile')
labelcsv = pd.read_csv('./survival_COAD_survival.csv', index_col='sample')

# In[49]:
# Define path to random sampling tile directory
TCGA_COAD_RS_PATH = './randomsampling'
foldername = os.listdir(TCGA_COAD_RS_PATH)  # List of patient folders

# In[50]:
# Initialize lists to store extracted features and metadata
X = []               # Aggregated features (9 tissues × 32 features = 288 dim)
y = []               # Survival labels [OS, OS.time]
Pathology = []       # Patient/Pathology IDs
TNL = []             # Tile Number List (count of tiles per tissue type)

# Process each patient folder
for folder in tqdm(foldername):
    # Store patient/pathology ID
    Pathology.append(folder)
    
    # Extract survival labels for the patient
    OS = labelcsv.at[folder[0:15], 'OS']
    OS_time = labelcsv.at[folder[0:15], 'OS.time']
    y.append([OS, OS_time])
    
    # Initialize list to store tiles per tissue type (9 tissue types)
    tissue_list = [[], [], [], [], [], [], [], [], []]
    # Categorize tiles by tissue type
    for file in os.listdir(os.path.join(TCGA_COAD_RS_PATH, folder)):
        t = tissue_labels.at[file, 'tissue']  # Get tissue type for tile
        tissue_list[t].append(file)
    
    # Initialize feature accumulators (32-dim per tissue type)
    feature_list = []
    for i in range(len(tissue_list)):
        feature_list.append(torch.zeros((1, 32)).to('cuda'))
    
    # Count tiles per tissue type
    tile_number_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Image transformation (same as training)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Extract and aggregate features for each tissue type
    for i in range(len(tissue_list)):
        # Process all tiles for current tissue type
        for j in range(len(tissue_list[i])):
            # Construct image path
            img_path = os.path.join(TCGA_COAD_RS_PATH, folder, tissue_list[i][j])
            image = Image.open(img_path)
            # Preprocess image
            image_tensor = transform(image)
            image_tensor.unsqueeze_(0)  # Add batch dimension
            image_tensor = image_tensor.to('cuda')
            
            # Extract features using corresponding tissue model
            model_list[i] = model_list[i].to('cuda')
            model_list[i].eval()
            
            with torch.no_grad():
                risk = model_list[i](image_tensor)  # Forward pass (unused)
                feature = model_list[i].feature    # Extract intermediate features (32-dim)
            
            # Accumulate features
            feature_list[i] = feature_list[i] + feature
        
        # Record number of tiles for the tissue type
        tile_number_list[i] = len(tissue_list[i])
        # Average features if there are tiles for the tissue type
        if len(tissue_list[i]) != 0:
            feature_list[i] = feature_list[i] / len(tissue_list[i])
        
    # Store tile counts for the patient
    TNL.append(tile_number_list)
     
    # Flatten feature list (9 tissues × 32 features)
    FEATURE_LIST = []
    for i in range(len(feature_list)):
        for j in range(len(feature_list[i][0])):
            FEATURE_LIST.append(feature_list[i][0][j].item())
    
    # Store flattened features
    X.append(FEATURE_LIST)

# Print dataset dimensions for verification
print(f'Pathology : {len(Pathology)} samples')
print(f'X : {len(X)} samples × {len(X[0])} features')
print(f'y : {len(y)} samples × {len(y[0])} labels')
print(f'TNL : {len(TNL)} samples × {len(TNL[0])} tissue tile counts')

# In[51]:
# Prepare data for CSV export
data = []
columns = []

# Combine all data into single list (Pathology + y + TNL + X)
for i in range(len(Pathology)):
    data.append([Pathology[i]] + y[i] + TNL[i] + X[i])

# Define column names
columns.append('pathology')                  # Patient/Pathology ID
columns.append('OS')                         # Overall Survival event
columns.append('OS.time')                    # Overall Survival time
# Tile count columns
columns.append('ADI.tile')
columns.append('BACK.tile')
columns.append('DEB.tile')
columns.append('LYM.tile')
columns.append('MUC.tile')
columns.append('MUS.tile')
columns.append('NORM.tile')
columns.append('STR.tile')
columns.append('TUM.tile')
# Feature columns (32 per tissue type)
for tissue in ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']:
    for j in range(32):
        columns.append(f'{tissue}.feature{j}')

# In[52]:
# Create DataFrame and save to CSV
df = pd.DataFrame(data=data, columns=columns)
df.to_csv('./histopathological_features_10xmbv3.csv', index=False)
