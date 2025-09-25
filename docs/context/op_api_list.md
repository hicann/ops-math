# 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

算子接口列表如下：

| 接口名                                                       | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [aclnnAbs](../../math/abs/docs/aclnnAbs.md)                  | 实现张量的绝对值运算。                                       |
| [aclnnAddLora](../../math/add_lora/docs/aclnnAddLora.md)     | 为神经网络添加LoRA（Low-Rank Adaptation）层功能，通过低秩分解减少参数数量。 |
| [aclnnCast](../../math/cast/docs/aclnnCast.md)               | 实现张量数据类型转换。                                       |
| [aclnnGroupedBiasAddGrad](../../math/grouped_bias_add_grad/docs/aclnnGroupedBiasAddGrad.md) | 分组偏置加法（GroupedBiasAdd）的反向传播。                   |
| [aclnnHansDecode](../../math/hans_decode/docs/aclnnHansDecode.md) | 对压缩后的张量基于PDF进行解码，同时基于mantissa（尾数）重组恢复张量。 |
| [aclnnHansEncode](../../math/hans_encode/docs/aclnnHansEncode.md) | 对张量的指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。 |
| [aclnnIsFinite](../../math/is_finite/docs/aclnnIsFinite.md)  | 判断张量中哪些元素是有限数值，即不是inf、-inf或nan。         |
| [aclnnIsInf](../../math/is_inf/docs/aclnnIsInf.md)           | 判断张量中哪些元素是无限大值，即是inf、-inf。                |
| [aclnnLinspace](../../math/lin_space/docs/aclnnLinspace.md)  | 生成一个等间隔数值序列。                                     |
| [aclnnSinkhorn](../math/sinkhorn/docs/aclnnSinkhorn.md) | 计算Sinkhorn距离，可以用于MoE模型中的专家路由。|