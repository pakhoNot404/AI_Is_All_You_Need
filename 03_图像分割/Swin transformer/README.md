# Swin Transformer

Paper：https://arxiv.org/abs/2103.14030

Code：https://github.com/microsoft/Swin-Transformer
      https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/swin_transformer

___

![swin transformer](<../../Images/Swin transformer.png>)

主要流程：

1. 首先经过Patch Partition和Linear Embedding层将输入转为 -> [B, H/4，W/4，48]；
2. 然后经过四个stage，每个stage会下采样两倍，通道数上升2倍；
3. 每个stage里面都有一个Patch Merging和若干个Swin Transformer Block；
4. 每个Swin Transformer Block里面都包含一个W-MSA和SW-MSA；
5. 对于分类任务，会在Stage4后面接Layer Norm、全局池化和全连接层得到最终的输出；对于分割任务而言，可以直接加各种分割头。

## Patch Partition & Linear Embedding

## Patch Merging

主要作用是下采样，特征图尺寸变为原来的一半，通道翻倍。

![Parch Meaging][./Images]





