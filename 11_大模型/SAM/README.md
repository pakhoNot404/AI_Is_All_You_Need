# SAM

Paper：https://arxiv.org/pdf/2304.02643v1

Code：https://github.com/facebookresearch/segment-anything

___


## Overview
SAM模型大致上分成3个模块，一个标准的vit构成的image encoder、一个prompt encoder和一个mask decoder。整体结构图如下所示

![SAM](<../../Images/SAM overview.png>)


## Image Encoder
Image encoder用于对输入图像进行特征提取，并获取编码。基本结构是ViT。ViT结构图如下

![ViT](<../../Images/ViT overview.png>)



## Prompt Encoder


## Mask Decoder