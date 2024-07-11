# Vision Transformer

Paper：https://arxiv.org/abs/2010.11929v2

Code：https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

___
 
 主要流程：

 1. 将输入图片分成若干个 16*16 的小patches
 2. 将每个patches都输入到Embedding层（Linear Projection of Flattened Patches），输出得到每个patch对应的向量，也被称为Token
 3. 在这一系列Token的前面加一个class token（参考的BERT），这个class token与其他Token的维度相同
 4. 在上面的基础上加上一个位置编码 Position Embedding
 5. 将前面的输出送进Transformer Encoder （Encoder Blocks堆叠L次）
 6. **分类任务**的话，接下来将class token的结果取出来，输入到MLP Head中即可；**分割任务**则删除类标记，直接将Transformer的输出映射回每个patch，并对每个patch进行解码，生成对应的分割掩码，不需要经过MLP Head。

 网络结构图：

 ![ViT](<../../Images/ViT overview.png>)


 ## Embedding层

> 对于标准的Transformer模块，要求输入的事token（向量）序列，即二维矩阵[num_token, token_dim]
功能：实现流程中的1，2步。
- 在代码实现中直接通过一个卷积层来实现，卷积核大小为16×16，stride为16，卷积核个数为768（ViT-B）
- [224, 224, 3] -> [14, 14, 768] -> [196, 768]
- 在输入Transformer Encoder之前需要加上[class] token 以及Position Embedding，都是可训练参数
- 拼接[class] token: Cat([1, 768], [196, 768]) -> [197, 768]
- 叠加Position Embedding：[197, 768] -> [197, 768] 论文中实验证明，位置编码不是很重要，默认使用1D的位置编码

## Transformer Encoder

Transformer Encoder 将Encoder Block 重复堆叠L次。接下来详细介绍Encoder Block：

- [ ] Layer Norm
- [ ] Multi-Head Attention
- [ ] Dropout
- [ ] MLP Block

### Layer Norm






![encoder block](<../../Images/Transformer Encoder Block.png>)

Encoder Block流程图截屏自B站@霹雳吧啦Wz，此处也向他的系列教学视频表示感谢~


 





