import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import time
import copy
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        lablelist = []
        for folder in os.listdir(img_dir):
            for file in os.listdir(os.path.join(img_dir, folder)):
                imagelist.append(file)
                lablelist.append(folder)
        self.imagelist = imagelist
        self.lablelist = lablelist

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.lablelist[idx], self.imagelist[idx])
        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.lablelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
target_transform = None

NCT_CRC_HE_100K = ImageDataset('./NCT-CRC-HE-100K', transform=transform, target_transform=target_transform)

NCT_CRC_HE_100K_NONORM = ImageDataset('D:/WSI-HSfeatures-main/NCT-CRC-HE-100K-NONORM', transform=transform,
                                      target_transform=target_transform)

# 合并数据集
combined_dataset = ConcatDataset([NCT_CRC_HE_100K, NCT_CRC_HE_100K_NONORM])

# 70% training, 15% validation, and 15% testing
labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}

targets = np.arange(len(combined_dataset))
for i in range(len(combined_dataset)):
    if i < len(NCT_CRC_HE_100K):
        targets[i] = labels_map[NCT_CRC_HE_100K.lablelist[i]]
    else:
        targets[i] = labels_map[NCT_CRC_HE_100K_NONORM.lablelist[i - len(NCT_CRC_HE_100K)]]

train_index, val_test_index = train_test_split(np.arange(len(targets)), test_size=0.3, random_state=0, shuffle=True,
                                               stratify=targets)

val_test_targets = []
sort_val_test_index = sorted(val_test_index)
for i in sort_val_test_index:
    val_test_targets.append(targets[i])

val_index, test_index = train_test_split(sort_val_test_index, test_size=0.5, random_state=0, shuffle=True,
                                         stratify=val_test_targets)

train_dataset = Subset(combined_dataset, train_index)
val_dataset = Subset(combined_dataset, val_index)
test_dataset = Subset(combined_dataset, test_index)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# print(len(train_dataloader.dataset))
# print(len(val_dataloader.dataset))
# print(len(test_dataloader.dataset))
# change ResNet50 FC layer

model = models.resnet50(weights=True)

for name, param in model.named_parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 9)
)
loss_fn = nn.CrossEntropyLoss()
#print(loss_fn)

params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.Adam(params_to_update, lr=0.0001)
# print(optimizer)

def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, num_epochs=50):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to('cuda')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        time.sleep(1)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.shape[0]
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # print("deep copy the model")
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            time.sleep(1)

        print('-' * 10)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model_ft, hist = train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer)

torch.save(model_ft.state_dict(), './resnet50_weights.pth')

import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split
from PIL import Image

import os
import time
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# For CAM visualization
from torchcam.methods import CAM, GradCAM, SmoothGradCAMpp  # Choose any CAM method
from torchcam.utils import overlay_mask

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


# ================== Dataset & Dataloader ==================
class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        imagelist = []
        lablelist = []
        for folder in os.listdir(img_dir):
            for file in os.listdir(os.path.join(img_dir, folder)):
                imagelist.append(file)
                lablelist.append(folder)
        self.imagelist = imagelist
        self.lablelist = lablelist

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.lablelist[idx], self.imagelist[idx])
        image = Image.open(img_path)
        labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
        label = labels_map[self.lablelist[idx]]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

target_transform = None

NCT_CRC_HE_100K = ImageDataset('./NCT-CRC-HE-100K', transform=transform, target_transform=target_transform)

# Split dataset (70% train, 15% val, 15% test)
labels_map = {"ADI": 0, "BACK": 1, "DEB": 2, "LYM": 3, "MUC": 4, "MUS": 5, "NORM": 6, "STR": 7, "TUM": 8}
targets = np.arange(len(NCT_CRC_HE_100K))
for i in range(len(NCT_CRC_HE_100K)):
    targets[i] = labels_map[NCT_CRC_HE_100K.lablelist[i]]

train_index, val_test_index = train_test_split(np.arange(len(targets)), test_size=0.3, random_state=0, shuffle=True,
                                               stratify=targets)

val_test_targets = []
sort_val_test_index = sorted(val_test_index)
for i in sort_val_test_index:
    val_test_targets.append(targets[i])

val_index, test_index = train_test_split(sort_val_test_index, test_size=0.5, random_state=0, shuffle=True,
                                         stratify=val_test_targets)

train_dataset = Subset(NCT_CRC_HE_100K, train_index)
val_dataset = Subset(NCT_CRC_HE_100K, val_index)
test_dataset = Subset(NCT_CRC_HE_100K, test_index)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# ================== Model Definition ==================
model = models.resnet50(weights=True)

# Freeze all layers except the final classifier
for name, param in model.named_parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(2048, 1024),
    nn.Dropout(0.5),
    nn.Linear(1024, 9)
)

loss_fn = nn.CrossEntropyLoss()

params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = optim.Adam(params_to_update, lr=0.0001)


# ================== Training Loop ==================
def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, num_epochs=1):
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model = model.to(device)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        time.sleep(1)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
            else:
                model.eval()
                dataloader = val_dataloader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            time.sleep(1)

        print('-' * 10)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Train the model
model_ft, hist = train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer)

# Save the trained model
torch.save(model_ft.state_dict(), './resnet50_weights.pth')

# ================== Testing with CAM Visualization ==================
model_ft = model_ft.to(device)
model_ft.eval()

# Initialize CAM extractor (choose any method: CAM, GradCAM, SmoothGradCAMpp)
cam_extractor = CAM(model_ft, 'layer4')  # 'layer4' is the last conv layer in ResNet50

running_loss = 0.0
running_corrects = 0

for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model_ft(inputs)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Generate CAM
        activation_map = cam_extractor(preds.item(), outputs)

        # Visualize CAM (for the first image in the batch)
        if len(activation_map) > 0:
            # Convert tensor to PIL image
            img = transforms.ToPILImage()(inputs.squeeze(0).cpu())
            # Overlay CAM on the original image
            result = overlay_mask(img, transforms.ToPILImage()(activation_map[0].squeeze(0).cpu()), alpha=0.5)

            # Display the result
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'Original Image\nTrue: {labels.item()}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title(f'CAM Overlay\nPredicted: {preds.item()}')
            plt.axis('off')

            plt.show()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

# Compute final test metrics
test_loss = running_loss / len(test_dataloader.dataset)
test_acc = running_corrects.double() / len(test_dataloader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
model_ft = model_ft.to(device)
model_ft.eval()

# Initialize CAM extractor (choose any method: CAM, GradCAM, SmoothGradCAMpp)
cam_extractor = CAM(model_ft, 'layer4')  # 'layer4' is the last conv layer in ResNet50

running_loss = 0.0
running_corrects = 0

for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model_ft(inputs)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Generate CAM
        activation_map = cam_extractor(preds.item(), outputs)

        # Visualize CAM (for the first image in the batch)
        if len(activation_map) > 0:
            # Convert tensor to PIL image
            img = transforms.ToPILImage()(inputs.squeeze(0).cpu())
            # Overlay CAM on the original image
            result = overlay_mask(img, transforms.ToPILImage()(activation_map[0].squeeze(0).cpu()), alpha=0.5)

            # Display the result
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'Original Image\nTrue: {labels.item()}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title(f'CAM Overlay\nPredicted: {preds.item()}')
            plt.axis('off')

            plt.show()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

# Compute final test metrics
test_loss = running_loss / len(test_dataloader.dataset)
test_acc = running_corrects.double() / len(test_dataloader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
model_ft = model_ft.to(device)
model_ft.eval()

# Initialize CAM extractor (choose any method: CAM, GradCAM, SmoothGradCAMpp)
cam_extractor = CAM(model_ft, 'layer4')  # 'layer4' is the last conv layer in ResNet50

running_loss = 0.0
running_corrects = 0

for inputs, labels in tqdm(test_dataloader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model_ft(inputs)
        loss = loss_fn(outputs, labels)
        _, preds = torch.max(outputs, 1)

        # Generate CAM
        activation_map = cam_extractor(preds.item(), outputs)

        # Visualize CAM (for the first image in the batch)
        if len(activation_map) > 0:
            # Convert tensor to PIL image
            img = transforms.ToPILImage()(inputs.squeeze(0).cpu())
            # Overlay CAM on the original image
            result = overlay_mask(img, transforms.ToPILImage()(activation_map[0].squeeze(0).cpu()), alpha=0.5)

            # Display the result
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img)
            plt.title(f'Original Image\nTrue: {labels.item()}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.title(f'CAM Overlay\nPredicted: {preds.item()}')
            plt.axis('off')

            plt.show()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

# Compute final test metrics
test_loss = running_loss / len(test_dataloader.dataset)
test_acc = running_corrects.double() / len(test_dataloader.dataset)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')