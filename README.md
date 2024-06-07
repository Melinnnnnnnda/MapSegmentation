# MapSegmentation
This repository performs historical map segmentation based on three models: FCN, UNet, and Segformer. The land cover types include: buildings, forests, rivers, lakes, streams, wetlands and roads. Models will assign each pixel one or more land cover types.  

You can start with **main.py**, which performs the training of model UNet or FCN. If your model is Segformer, you can use **segformer.py** instead. After training, the model parameters will be stored under folder *model*. You can then use **reconstruct.py** to restore the whole image from the model parameters. **Draw.py** allows you to compare the performance of the three models.

The codes above can identify seven land cover types. But if you are only interested in one specific feature type, e.g.: buildings. You can use **oneType_train.py** to train, and **oneType_reconstruct.py** to restore the whole image.

## main.py
It trains UNet or FCN on map segmentation of seven land cover types (buildings, forests, rivers, lakes, streams, wetlands and roads). It outputs model parameters.

Training :arrow_right: folder *1223*

Testing :arrow_right: folder *1222*

⚠️ **Note: The images couldn't be uploaded due to space limits of GitHub, so please download the images from Google Drive before you run the codes:**

https://drive.google.com/drive/folders/1qXMpPyXZK_t8Nc3d4_9DuFEqsrywhR-p?usp=sharing

Output :arrow_right: folder *model*

## segformer.py
It trains Segformer on map segmentation. The training, testing, and outputs are similar to main.py.

## fcn.py, unet.py, and unet_parts.py
They store network structures for FCN and UNet, no need to run them.

## reconstruct.py
It restores the whole image based on seven land cover types: buildings, forests, rivers, lakes, streams, wetlands and roads. For each type it generates a binary prediction (0: no feature, 1: feature exists) of the original raster (image to be segmented).

Input model: :arrow_right: a model in folder *model*

(You can also find a demo model in Google Drive: https://drive.google.com/drive/folders/1qXMpPyXZK_t8Nc3d4_9DuFEqsrywhR-p?usp=sharing)

Input raster: :arrow_right: image 1222.tif in folder *1222*

Output: :arrow_right: folder *reconstruct*

## draw.py
It visualizes mode performance. It compares original image patches with the predictions by three models on seven land cover types.

Input model: :arrow_right: three models in folder *model*

Input raster: :arrow_right: image 1222.tif in folder *1222*

Output: :arrow_right: 4*7 image patches on different model and land cover types

## oneType_reconstruct.py and oneType_train.py
They are similar to the previous main.py and reconstruct.py. The only difference is that these two only trains and identifies only one land cover type, e.g: buildings.
