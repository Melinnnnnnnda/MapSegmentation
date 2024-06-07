import rasterio
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import Dataset, DataLoader,TensorDataset
import numpy as np
from skimage.util import view_as_windows
from PIL import Image
import argparse
import numpy as np
import torch
from torch.utils import data
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
import torch.nn.functional as F
import pdb
import matplotlib.pyplot as plt
import random


class HistoricalMapDataset(Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, start_idx, end_idx,dilation=False):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.w_size     = w_size
        self.dilation   = dilation
        self.image_patches    = np.array(generate_tiling(self.image_path, w_size=self.w_size))[start_idx:end_idx]
        self.gt_patches=[]
        for gt in large_gt_path:
            self.gt_patches.append(np.array(generate_tiling(gt,    w_size=self.w_size))[start_idx:end_idx]) 
        self.gt_patches = np.asarray(self.gt_patches)
        self.gt_patches = np.moveaxis(self.gt_patches,0, -1)
        print('Window_size: {}, Generate {} image patches and {} gt patches.'.format(w_size, len(self.image_patches), len(self.gt_patches)))

    def __len__(self):
        return len(self.image_patches) 

    def __getitem__(self, index):
        img    = self.image_patches[index]
        labels = self.gt_patches[index]

        img = img / 255.
        img = np.array(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()
        labels = np.array(labels, dtype=np.float32)
        labels = np.transpose(labels, (2, 0, 1))
        labels = torch.from_numpy(labels).float()
        
        
        if self.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            labels = binary_dilation(labels, structure=struct1).astype(np.uint8)

        
        return img, labels
     
def generate_tiling(in_img_path, w_size):
    # Generate tiling images
    in_img = np.array(Image.open(in_img_path))
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    if len(in_img.shape) == 2:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px)], 'edge')
        tiles = view_as_windows(img_pad, (win_size,win_size), step=pad_px)
    else:
        img_pad = np.pad(in_img, [(pad_px,pad_px), (pad_px,pad_px), (0,0)], 'edge')
        tiles = view_as_windows(img_pad, (win_size,win_size,3), step=pad_px)
    tiles_lst = []
    for row in range(tiles.shape[0]):
        for col in range(tiles.shape[1]):
            if len(in_img.shape) == 2:
                tt = tiles[row, col, ...].copy()
            else:
                tt = tiles[row, col, 0, ...].copy()
            tiles_lst.append(tt)


    return tiles_lst

def count_classes(train_gt_paths, class_names):
    # Count num of pixels for each land cover type
    counts = []
    
    for path in train_gt_paths:
        with rasterio.open(path) as src:
            data = src.read(1)  # Read the first channel
            count = (data == 1).sum()  # pixel=1 counts
            counts.append(count)

    # Draw
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, counts, color='skyblue')
    plt.xlabel('Land Cover Type')
    plt.ylabel('Number of Pixels')
    plt.title('Number of Pixels for Each Land Cover Type')
    plt.xticks(rotation=45)

    # Add value on each bar
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}', va='bottom', ha='center')
    
    plt.tight_layout()
    plt.show()

    return counts


# The following are methods for data augmentation
def random_flip(image, label):
    # Flip
    if random.random() < 0.5:
        # Horizontal flip
        image = torch.flip(image, [2])
        label = torch.flip(label, [1])
    if random.random() < 0.5:
        # Vertical flip
        image = torch.flip(image, [1])
        label = torch.flip(label, [0])
    return image, label

def random_rotation(image, label):
    # Rotate
    k = random.randint(0, 3)
    image = torch.rot90(image, k, dims=[1, 2])
    label = torch.rot90(label, k, dims=[0, 1])
    return image, label

def random_translation(image, label, max_shift=10):
    # Translate
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)
    image = torch.roll(image, shifts=(tx, ty), dims=(1, 2))
    label = torch.roll(label, shifts=(tx, ty), dims=(0, 1))
    return image, label

def add_random_noise(image, noise_level=0.1):
    # Add noise
    noise = torch.randn(image.size()) * noise_level
    image = image + noise
    image = torch.clip(image, 0, 1)
    return image


def augment_image(image, label, augmentation_count):
    # image: [Channel=3, H, W]
    # label: [H, W]
    # Returm: augmented image set，label set and number of non-zero pixels

    augmented_images = []
    augmented_labels = []
    num = 0

    for _ in range(augmentation_count):
        aug_img, aug_lbl = image.clone(), label.clone()
        aug_img, aug_lbl = random_flip(aug_img, aug_lbl)
        aug_img, aug_lbl = random_rotation(aug_img, aug_lbl)
        aug_img, aug_lbl = random_translation(aug_img, aug_lbl)
        aug_img = add_random_noise(aug_img)  # 仅对图像加噪点
        augmented_images.append(aug_img)
        augmented_labels.append(aug_lbl)
         # Calculate number of pixels=1 in label
        num += torch.sum(aug_lbl == 1).item()
        
    return augmented_images, augmented_labels, num

def augment_dataset(class_index, threshold, count, image, labels):
    # Augment each image to form an augmented dataset
    if (labels[class_index].sum() / total_pixels) > threshold:
        # Augemnt
        aug_images, aug_labels, num = augment_image(image, labels[class_index],count)
        for aug_image, aug_label in zip(aug_images, aug_labels):
            # For the same tensor size
            augmented_label = torch.zeros_like(labels)
            augmented_label[class_index] = aug_label
            augmented_dataset.append((aug_image, augmented_label))
            # Update class_count
        class_counts[class_index]+=num
    
def calculate_weights(counts):
    # Calculate the weights for each land cover type by the number of their non-zero pixels
    counts = torch.tensor(counts, dtype=torch.float32)
    total_pixels = counts.sum()
    # weight equals to total pixels / non-zero pixels
    pos_weights = total_pixels / counts
    return pos_weights

def plot(images):
    # Plot the augmentations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    titles = ['Original','Horizontal Flip', 'Rotation', 'Translation', 'Random Noise']

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.permute(1, 2, 0).numpy())  # Convert CHW to HWC for plotting
        # ax.imshow(lbl,cmap='gray')  # Overlay label
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

import random

def downsample(dataset, class_counts, target_class_count):
    # Decrease image patches to balance the dataset
    downsampled_dataset = []
    class_indices = {i: [] for i in range(len(class_counts))}
    downsampled_class_counts = np.zeros(len(class_counts), dtype=int)

    # Save the index for each land cover type
    for idx, (image, label) in enumerate(dataset):
        for i in range(len(class_counts)):
            if (label[i] != 0).sum() > 0:
                class_indices[i].append(idx)

    # Decrease images
    for class_index, indices in class_indices.items():
        if class_counts[class_index] > target_class_count:
            # Initialize
            sampled_indices = []
            current_pixels = 0
            # random shuffle
            random.shuffle(indices)

            for idx in indices:
                sample_pixels = (dataset[idx][1][class_index] != 0).sum().item()
                if current_pixels + sample_pixels > target_class_count:
                    break
                sampled_indices.append(idx)
                current_pixels += sample_pixels
            downsampled_class_counts[class_index] = current_pixels
        else:
            sampled_indices = indices
            downsampled_class_counts[class_index] = class_counts[class_index]

        downsampled_dataset.extend([dataset[i] for i in sampled_indices])


    return downsampled_dataset, downsampled_class_counts







w_size=256
batch_size=8
total_pixels = w_size * w_size

n_class=7
class_names = ['Buildings', 'Forests', 'Lakes', 'Rivers', 'Wetlands', 'Roads', 'Streams']
train_image_paths='./1223/1223.tif'
train_gt_paths=['./1223/buildings.tif','./1223/forests.tif','./1223/lakes.tif','./1223/rivers.tif','./1223/wetlands.tif','./1223/roads.tif', './1223/streams.tif']
test_image_paths='./1222/1222.tif'
test_gt_paths=['./1222/buildings.tif','./1222/forests.tif','./1222/lakes.tif','./1222/rivers.tif','./1222/wetlands.tif','./1222/roads.tif','./1222/streams.tif']



class_counts = count_classes(train_gt_paths, class_names)

# train:3000 validation:750  4:1
train_start_idx = 200
train_end_idx = 3200
val_start_idx = 3200
val_end_idx = 3950

train_dataset = HistoricalMapDataset(train_image_paths, train_gt_paths, w_size, train_start_idx, train_end_idx, dilation=False)


# Weight for each land cover type
pos_weights = calculate_weights(class_counts)




j=0
lbl=[]

augmented_dataset = []


for i in range(len(train_dataset)):
    image, labels = train_dataset[i]
    
    # Building
    augment_dataset(0, 0.01, 1, image, labels)
    # Lake
    augment_dataset(2, 0.01, 120, image, labels)
    # River
    augment_dataset(3, 0.04, 5, image, labels)
    # Wetland
    augment_dataset(4, 0.01, 590, image, labels)

    

# Combine original and augmented dataset
full_dataset = list(train_dataset) + augmented_dataset

# Set minimum class count as the count of stream
target_class_counts = class_counts[6]
# Downsample dataset
downsampled_dataset, downsampled_class_counts = downsample(full_dataset, class_counts, target_class_counts)

# # Plot each land cover type in the dataset
# plt.figure(figsize=(10, 6))
# bars = plt.bar(class_names, downsampled_class_counts, color='skyblue')
# plt.xlabel('Land Cover Type')
# plt.ylabel('Number of Pixels')
# plt.title('Number of Pixels for Each Land Cover Type')
# plt.xticks(rotation=45)

# for bar, count in zip(bars, downsampled_class_counts):
#     plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{count}', va='bottom', ha='center')

# plt.tight_layout()
# plt.show()


# Here we train with the full dataset instead of the downsampled one for better results
# trainloader = DataLoader(downsampled_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
trainloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
n_train = len(trainloader)

# Validation dataset
val_dataset = HistoricalMapDataset(train_image_paths, train_gt_paths, w_size, val_start_idx, val_end_idx, dilation=False)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# Test dataset
test_dataset = HistoricalMapDataset(test_image_paths,test_gt_paths,w_size,train_start_idx, train_end_idx,dilation=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
n_test = len(testloader)



### UNET
from unet import UNET
model = UNET(3, n_class)  #three channels, one-type feature segmentation first
print("UNET model success")


### FCN
from fcn import FCN8s
from fcn import VGGNet
# vgg_model = VGGNet(requires_grad=True)
# model=FCN8s(vgg_model,n_class)
# print("FCN model success")







import tqdm
from torch import optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Loss function for binary classification, here i use the augmented dataset, so no weight loss needed
criteria = nn.BCELoss(reduction='mean')
# To use weighted loss, change to the code below, and remeber to restore traindataset to the original one
# criteria = nn.BCELoss(reduction='mean',weight=pos_weights)  #image [B,H,W,C]




# Adam optimizer
base_lr = 1e-3
weight_decay = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)


epochs = 100 # Define the number of epochs  


model = model.to(device)
# pos_weights = pos_weights.to(device)

min_eval_loss = 100


SAVE_PATH = './model/unet.pth'
SAVE_PATH_BEST = './model/unet_best.pth'



# Training, loop over the dataset multiple times
for epoch in range(epochs):
    
    model.train()
    train_loss = 0
    
    with tqdm.tqdm(total=int(n_train*batch_size-1), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:  
        for i, data in enumerate(trainloader):

            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            
            outputs=torch.transpose(outputs,1,-1)
            labels=torch.transpose(labels,1,-1)
            
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            train_loss += loss.item()
            
            # Update the pbar
            pbar.update(batch_size)
         
            # Add loss (batch) value to tqdm
            pbar.set_postfix(**{'Loss': loss.item()})
    
    torch.save(model.state_dict(), SAVE_PATH)

    # Evaluation phase
    model.eval()

    eval_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            outputs=torch.transpose(outputs,1,-1)
            labels=torch.transpose(labels,1,-1)
            
            loss = criteria(outputs, labels)
            eval_loss += loss.item()
    
    # average loss, not total
    avg_eval_loss = eval_loss / len(testloader) 

    # Print statistics
    print(f'Epoch {epoch+1}, Eval Loss: {avg_eval_loss}')

    if avg_eval_loss < min_eval_loss:
        min_eval_loss = avg_eval_loss
        torch.save(model.state_dict(), SAVE_PATH_BEST)
        print(f'Epoch {epoch+1}, Eval Loss decreasing, saving model.')
    


print('Finished Training')



