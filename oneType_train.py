
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
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor, TrainingArguments, Trainer
import torch.nn as nn
CUDA_LAUNCH_BLOCKING=1


class HistoricalMapDataset(Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, start_idx, end_idx, dilation=False):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.w_size     = w_size
        self.dilation   = dilation
        self.image_patches    = np.array(generate_tiling(self.image_path, w_size=self.w_size))[start_idx:end_idx]
        self.gt_patches       = np.array(generate_tiling(self.gt_path,    w_size=self.w_size))[start_idx:end_idx]
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
        
        
        if self.dilation:
            struct1 = ndimage.generate_binary_structure(2, 2)
            labels = binary_dilation(labels, structure=struct1).astype(np.uint8)

        labels = torch.from_numpy(np.array([labels])).float()
        return img, labels
    
def generate_tiling(image_path, w_size):
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


    return tiles_lst

w_size=256
batch_size=8
n_class=1

train_raster_path = './1223/1223.tif'
train_shp_path = './1223/roads.tif'
test_raster_path = './1222/1222.tif'
test_shp_path = './1222/roads.tif'

# train:3000 validation:750  4:1
train_start_idx = 200
train_end_idx = 3200
val_start_idx = 3200
val_end_idx = 3950

train_dataset = HistoricalMapDataset(train_raster_path,train_shp_path,w_size,train_start_idx,train_end_idx,dilation=False) 
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) 
n_train = len(trainloader)

val_dataset = HistoricalMapDataset(train_raster_path,train_shp_path, w_size, val_start_idx, val_end_idx, dilation=False)
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

test_dataset = HistoricalMapDataset(test_raster_path,test_shp_path,w_size,train_start_idx,train_end_idx,dilation=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
n_test = len(testloader)



print("load success")




### UNET
from unet import UNET
# model = UNET(3, 1)  #three channels, one-type feature segmentation first
# print("UNET model success")


### FCN
from fcn import FCN8s
from fcn import VGGNet
# vgg_model = VGGNet(requires_grad=True)
# model=FCN8s(vgg_model,n_class)
# print("FCN model success")

### Segformer
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.decode_head.classifier = nn.Sequential(
        nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid()
        )




import tqdm
from torch import optim
import torch.nn as nn

# Loss function for binary classification
criteria = nn.BCELoss(reduction='mean')

# Adam optimizer
base_lr = 1e-3
weight_decay = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)  
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)


epochs = 100  # Define the number of epochs  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

min_eval_loss = 100
# Saving retrain model
SAVE_PATH = './model/segformer_roads.pth'
SAVE_PATH_BEST = './model/segformer_roads_best.pth'


# Loop over the dataset multiple times
for epoch in range(epochs):

    model.train()
    train_loss = 0
    with tqdm.tqdm(total=int(n_train*batch_size-1), desc=f'Epoch {epoch + 1}/{epochs}', unit='img', bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}') as pbar:  
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)['logits']
            outputs = nn.functional.interpolate(outputs,
                        size=(256, 256), # (height, width)
                        mode='bilinear',
                        align_corners=False)
            
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
            outputs = model(images)['logits']
            outputs = nn.functional.interpolate(outputs,
            size=(256, 256), # (height, width)
            mode='bilinear',
            align_corners=False)
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


torch.save(model.state_dict(), SAVE_PATH)





