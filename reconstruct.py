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
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.transform import from_origin

def reconstruct_from_patches(patches_images, patch_size, step_size, image_size_2d, image_dtype):
    '''
    Adjust to take patch images directly.
    patch_size is the size of the tiles
    step_size should be patch_size//2
    image_size_2d is the size of the original image
    image_dtype is the data type of the target image
    '''
    i_h, i_w = np.array(image_size_2d[:2]) + (patch_size, patch_size) # assuming image_size_2d contains only height and width
    num_classes = patches_images.shape[-1]  # the last dimension is the number of classes
    p_h = p_w = patch_size

    # Create an image for each class
    imgs = np.zeros((num_classes, i_h + p_h//2, i_w + p_w//2), dtype=image_dtype)
    
    numrows = (i_h) // step_size - 1
    numcols = (i_w) // step_size - 1
    expected_patches = numrows * numcols
    
    if len(patches_images) != expected_patches:
        raise ValueError(f"Expected {expected_patches} patches, got {len(patches_images)}")

    patch_offset = step_size // 2
    patch_inner = p_h - step_size
    for row in range(numrows):
        for col in range(numcols):
            idx = row * numcols + col
            patch = patches_images[idx]
            # Remove the padding from the patch for each class
            for c in range(num_classes):
                patch_roi = patch[patch_offset:-patch_offset, patch_offset:-patch_offset, c]
                imgs[c, row * step_size:row * step_size + patch_inner,
                    col * step_size:col * step_size + patch_inner] = patch_roi

    # Remove the extra padding to match the original image size
    final_images = imgs[:, step_size // 2:-(patch_size + step_size // 2),
                        step_size // 2:-(patch_size + step_size // 2)]
 
    return final_images


class HistoricalMapDataset(Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, dilation=False):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.w_size     = w_size
        self.dilation   = dilation
        self.image_patches    = np.array(generate_tiling(self.image_path, w_size=self.w_size))
        self.gt_patches=[]
        for gt in large_gt_path:
            self.gt_patches.append(np.array(generate_tiling(gt,    w_size=self.w_size))) 

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

w_size=256
batch_size=8

n_class=7
train_image_paths='./1223/1223.tif'
train_gt_paths=['./1223/buildings.tif','./1223/forests.tif','./1223/lakes.tif','./1223/rivers.tif','./1223/wetlands.tif','./1223/roads.tif', './1223/streams.tif']
test_image_paths='./1222/1222.tif'
test_gt_paths=['./1222/buildings.tif','./1222/forests.tif','./1222/lakes.tif','./1222/rivers.tif','./1222/wetlands.tif','./1222/roads.tif','./1222/streams.tif']



# train_dataset = HistoricalMapDataset(train_image_paths,train_gt_paths,w_size,dilation=False) 
# trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
# n_train = len(trainloader)

test_dataset = HistoricalMapDataset(test_image_paths,test_gt_paths,w_size,dilation=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
n_test = len(testloader)



print("load success")


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


LOAD_PATH_unet = './model/unet.pth'



model.load_state_dict(torch.load(LOAD_PATH_unet))


import tqdm
from torch import optim
import torch.nn as nn

# Adam optimizer
base_lr = 1e-4
weight_decay = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)


epochs = 100 # Define the number of epochs  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_reconstruct=[]

with torch.no_grad():  # Temporarily set all the requires_grad flag to false
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        # Threshold the output
        predicted = (outputs>0.5).int()
        # Store in cpu in case of memory explosion
        test_reconstruct.append(predicted.detach().cpu().numpy())


# Concatenate along the batch dimension N, C, H, W
test_reconstruct = np.concatenate(test_reconstruct, axis=0)  
# Change the sequence of channel
test_reconstruct = test_reconstruct.transpose(0, 2, 3, 1) 


in_img = np.array(Image.open(test_image_paths))
step_size = w_size // 2


# reconstruct
reconstruct_img = reconstruct_from_patches(test_reconstruct, w_size, step_size, in_img.shape[:2], np.int32)

class_names = ['Buildings', 'Forests', 'Lakes', 'Rivers', 'Wetlands', 'Roads', 'Streams']


# Save image with geotransform and projection info
with rasterio.open('./1222/forests.tif') as src:
    transform = src.transform
    crs = src.crs

output_dir = './reconstruct'
os.makedirs(output_dir, exist_ok=True)

for i, name in enumerate(class_names):
    image = reconstruct_img[i]
    output_path = os.path.join(output_dir, f'{name}.tif')
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=image.shape[0],
        width=image.shape[1],
        count=1,
        dtype=image.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(image, 1)

print("Reconstructed images saved successfully.")






  
