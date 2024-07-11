# Vision Transformer

Paper：https://arxiv.org/abs/2010.11929v2

Code：https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

___
 
 主要流程：

 - 将输入图片分成若干个 16*16 的小patches
 - 将每个patches都输入到Embedding层（Linear Projection of Flattened Patches），输出得到每个patch对应的向量，也被称为Token
 - 在这一系列Token的前面加一个class token（参考的BERT），这个class token与其他Token的维度相同
 - 在上面的基础上加上一个位置编码 Position Embedding
 - 将前面的输出送进Transformer Encoder （Encoder Blocks堆叠L次）
 - 将class token的输入取出来

 网络结构图：

 ![ViT](<../../Images/ViT overview.png>)
 





