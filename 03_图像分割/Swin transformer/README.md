# Swin Transformer

Paper：https://arxiv.org/abs/2103.14030

Code：https://github.com/microsoft/Swin-Transformer
      https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/swin_transformer

___

![swin transformer](<../../Images/Swin transformer.png>)

主要流程：

1. 首先经过Patch Partition，-> [B, H/4，W/4，48]
2. 然后经过四个stage，每个stage会下采样两倍，通道数上升2倍

## Patch Partition & Linear Embedding







