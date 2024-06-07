# MapSegmentation
This repository performs historical map segmentation based on three models: FCN, UNet, and Segformer. The land cover types include: buildings, forests, rivers, lakes, streams, wetlands and roads. Models will assign each pixel one or more land cover types.  

You can start with **main.py**, which performs the training of model UNet or FCN. If your model is Segformer, you can use **segformer.py** instead. After training, the model parameters will be stored under folder *model*. You can then use **reconstruct.py** to restore the whole image from the model parameters. **Draw.py** allows you to compare the performance of the three models.

## main.py
It trains UNet or FCN on map segmentation. It outputs model parameters.

Training :arrow_right: folder *1223*

Testing: folder *1222*

⚠️ **Note: The images couldn't be uploaded due to space limits of GitHub, so please download the images from Google Drive before you run the codes:**

https://drive.google.com/drive/folders/1qXMpPyXZK_t8Nc3d4_9DuFEqsrywhR-p?usp=sharing

Output: folder *model*
