import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.utils.data import Dataset, DataLoader
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
import torchvision.transforms as T
import matplotlib.pyplot as plt
from matplotlib import gridspec

class HistoricalMapDataset(Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, dilation=False):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.w_size     = w_size
        self.dilation   = dilation
        self.image_patches    = np.array(generate_tiling(self.image_path, w_size=self.w_size))[200:215]
        self.gt_patches=[]
        for gt in large_gt_path:
            self.gt_patches.append(np.array(generate_tiling(gt,    w_size=self.w_size))[200:215]) 
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

def generate_tiling_seg(image_path, w_size):
    # Generate tiling images
    win_size = w_size
    pad_px = win_size // 2

    # Read image
    in_img = np.array(Image.open(image_path))
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

    # Reshape tiles array to move the channels dimension before the spatial dimensions
    tiles_array = np.array(tiles_lst)
    if len(in_img.shape) == 3:
        tiles_array = tiles_array.transpose(0, 3, 1, 2)  # Rearrange dimensions: (n_patches, height, width, channels) -> (n_patches, channels, height, width)
    
    return tiles_array

class SegMapDataset(Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, dilation=False):
        self.image_path = large_image_path
        self.gt_path = large_gt_path
        self.w_size = w_size
        self.dilation = dilation
        self.image_patches = np.array(generate_tiling_seg(self.image_path, w_size=self.w_size))[312:315]
        self.gt_patches = np.array(generate_tiling_seg(self.gt_path, w_size=self.w_size))[312:315]

    def __len__(self):
        return len(self.image_patches)  # Make sure this is returning the correct length

    def __getitem__(self, index):
        # Ensure data is correctly shaped for Image.fromarray()
        img = self.image_patches[index]
        gt = self.gt_patches[index]

        img = img / 255.
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img).float()
    
        gt = torch.from_numpy(np.array([gt])).float()
        return img, gt
    


import matplotlib.pyplot as plt






w_size=256
batch_size=8

n_class=7
train_image_paths='./1223/1223.tif'
train_gt_paths=['./1223/buildings.tif','./1223/forests.tif','./1223/lakes.tif','./1223/rivers.tif','./1223/wetlands.tif','./1223/roads.tif', './1223/streams.tif']
test_image_paths='./1222/1222.tif'
test_gt_paths=['./1222/buildings.tif','./1222/forests.tif','./1222/lakes.tif','./1222/rivers.tif','./1222/wetlands.tif','./1222/roads.tif','./1222/streams.tif']



test_dataset = HistoricalMapDataset(test_image_paths,test_gt_paths,w_size,dilation=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
n_test = len(testloader)


import matplotlib.pyplot as plt










### UNET
from unet import UNET
model_unet = UNET(3, n_class)  #three channels, one-type feature segmentation first
# print("UNET model success")


### FCN
from fcn import FCN8s
from fcn import VGGNet
vgg_model = VGGNet(requires_grad=True)
model_fcn=FCN8s(vgg_model,n_class)
# print("FCN model success")

### Segformer
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn
model_seg = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model_seg.decode_head.classifier = nn.Sequential(
        nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid()
        )

import tqdm
from torch import optim
import torch.nn as nn

# Loss function for binary classification
criteria = nn.BCELoss(reduction='mean')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_unet = model_unet.to(device)
LOAD_PATH_unet = './model/unet.pth'
model_unet.load_state_dict(torch.load(LOAD_PATH_unet))

model_fcn = model_fcn.to(device)
LOAD_PATH_fcn = './model/fcn.pth'
model_fcn.load_state_dict(torch.load(LOAD_PATH_fcn))

model_seg=model_seg.to(device)
LOAD_PATH_seg='./model/segformer.pth'
model_seg.load_state_dict(torch.load(LOAD_PATH_seg))

# Visualize test set images
dataiter = iter(testloader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)


class_names = ['Buildings', 'Forests', 'Lakes', 'Rivers', 'Wetlands', 'Roads', 'Streams']

gt=[]
lb=labels[0]
for i in range(n_class):
        temp = lb[i]
        temp = temp.unsqueeze(-1)
        temp = temp.detach().cpu().numpy()  # Detach the tensor, move it to CPU, and convert to NumPy array
        gt.append(temp)


def GetClassesOutput(model):
    # Predict one batch for model FCN or UNet

    outputs = model(images)

    # Extract the results of the first image patch
    first = outputs[0]  # [7, 256, 256]
    classes = []
    for i in range(n_class):
        # original output is possibility, binarize it
        temp = (first[i]>0.5).int()
        temp = temp.unsqueeze(-1)
        temp = temp.detach().cpu().numpy()  # Detach the tensor, move it to CPU, and convert to NumPy array
        classes.append(temp)
    return classes

def GetClassesOutput_Seg(model):
    # Predict one batch for model Segformer

    outputs = model(images)['logits']

    # Extract the results of the first image patch
    first = outputs[0]  # [7, 256, 256]
    
    classes = []
    for i in range(n_class):
        temp = first[i]

        temp = temp.unsqueeze(0)  # Add a batch dimension (N = 1)
        temp = temp.unsqueeze(0)  # Add a channel dimension (C = 1)
        temp = nn.functional.interpolate(temp,
                                size=(256, 256), # (height, width)
                                mode='bilinear',
                                align_corners=False)
        temp = temp.squeeze(0)  # Remove the batch dimension
        temp = temp.squeeze(0)  # Remove the channel dimension if necessary

        # original output is possibility, binarize it
        temp = (temp>0.5).int()
        temp = temp.unsqueeze(-1)
        temp = temp.detach().cpu().numpy()  # Detach the tensor, move it to CPU, and convert to NumPy array
        classes.append(temp)
    return classes

classes_unet=GetClassesOutput(model_unet)
classes_fcn=GetClassesOutput(model_fcn)
classes_seg=GetClassesOutput_Seg(model_seg)


import matplotlib.pyplot as plt

org_img = images[0].permute(1, 2, 0)
org_img = org_img.detach().cpu().numpy() 

# Setting the size of the entire figure
fig, axs = plt.subplots(5, n_class+1, figsize=(18, 12))  # 4 rows, n_class + 1 columns

for i in range(n_class+1):
    axs[0, i].axis('off')
# Displaying the original image at the top center
axs[0, 4].imshow(org_img)
axs[0, 4].set_title('Original')
axs[0, 4].axis('off')

# Displaying each class output for FCN, UNet, and Segformer
for i in range(n_class):
    # Ground truth
    axs[1, i+1].imshow(gt[i], cmap='gray')
    axs[1, i+1].set_title(class_names[i])
    axs[1, i+1].axis('off')

    # FCN outputs
    axs[2, i+1].imshow(classes_fcn[i], cmap='gray')
    axs[2, i+1].axis('off')

    # UNet outputs
    axs[3, i+1].imshow(classes_unet[i], cmap='gray')
    axs[3, i+1].axis('off')

    # Segformer outputs
    axs[4, i+1].imshow(classes_seg[i], cmap='gray')
    axs[4, i+1].axis('off')

ModelLabels=['Ground truth','FCN','UNet','Segformer']
for i in range(1,5):
    axs[i, 0].text(-0.1, 0.5, ModelLabels[i-1], transform=axs[i, 0].transAxes, fontsize=14, va='center', ha='right', rotation=0)
    axs[i, 0].axis('off')
# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()







print(1)







