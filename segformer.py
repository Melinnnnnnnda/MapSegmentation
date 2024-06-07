
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
import torch
import matplotlib.pyplot as plt
from torch import optim
import torch.nn as nn

class HistoricalMapDataset(Dataset):
    def __init__(self, large_image_path, large_gt_path, w_size, dilation=False):
        self.image_path = large_image_path
        self.gt_path    = large_gt_path
        self.w_size     = w_size
        self.dilation   = dilation
        self.image_patches    = np.array(generate_tiling(self.image_path, w_size=self.w_size))[200:3200]
        self.gt_patches=[]
        for gt in large_gt_path:
            self.gt_patches.append(np.array(generate_tiling(gt,    w_size=self.w_size))[200:3200]) 
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

train_dataset = HistoricalMapDataset(train_image_paths,train_gt_paths,w_size,dilation=False) 
trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) # WARNING: SHUFFLE MUST BE TRUE TO PREVENT HUGE OVERFIT
n_train = len(trainloader)

test_dataset = HistoricalMapDataset(test_image_paths,test_gt_paths,w_size,dilation=False)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
n_test = len(testloader)


    


# Prepare the processor and the model
processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model.decode_head.classifier = nn.Sequential(
        nn.Conv2d(256, n_class, kernel_size=(1, 1), stride=(1, 1)),
        nn.Sigmoid()
        )

print("Segformer model success")


criteria = nn.BCELoss(reduction='mean')

# Adam optimizer
base_lr = 1e-3
weight_decay = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-5, verbose=True)

epochs = 100  # Define the number of epochs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)







SAVE_PATH = './model/segformer.pth'
SAVE_PATH_BEST = './model/segformer_best.pth'


import matplotlib.pyplot as plt


# Training Function
def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, device):

    min_eval_loss = 100
    
    for epoch in range(epochs):
        # 1个epoch保存一次
        train_loss = 0.0
        
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)['logits']
            outputs = nn.functional.interpolate(outputs,
                                    size=(256, 256), # (height, width)
                                    mode='bilinear',
                                    align_corners=False)
        
            # Compute the loss
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        # Print statistics
        print(f'Epoch {epoch+1}, Train Loss: {train_loss / len(train_dataloader)}')

        # torch.save(model.state_dict(), SAVE_PATH)

        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)['logits']
                outputs = nn.functional.interpolate(outputs,
                                    size=(256, 256), # (height, width)
                                    mode='bilinear',
                                    align_corners=False)
                loss = criterion(outputs, labels)
                eval_loss += loss.item()

        # average loss, not total
        avg_eval_loss = eval_loss / len(testloader) 
        # Print statistics
        print(f'Epoch {epoch+1}, Eval Loss: {avg_eval_loss}')

        if avg_eval_loss < min_eval_loss:
            min_eval_loss = avg_eval_loss
            torch.save(model.state_dict(), SAVE_PATH_BEST)
            print(f'Epoch {epoch+1}, Eval Loss decreasing, saving model.')

# Train the model
train_model(model, trainloader, testloader, optimizer=optimizer, criterion=criteria, device=device)


