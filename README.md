# MapSegmentation
This repository performs historical map segmentation based on three models: FCN, UNet, and Segformer. The land cover types include: buildings, forests, rivers, lakes, streams, wetlands and roads. Models will assign each pixel one or more land cover types.  

You can start with **main.py**, which performs the training of model UNet or FCN. If your model is Segformer, you can use **segformer.py** instead. After training, the model parameters will be stored under folder *model*. You can then use **reconstruct.py** to restore the whole image from the model parameters. **Draw.py** allows you to compare the performance of the three models.

## main.py
It trains UNet or FCN on map segmentation. 

Training: folder *1223*

Testing: folder *1222*
