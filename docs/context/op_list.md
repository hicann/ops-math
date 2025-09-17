# 算子清单

> - 算子目录：每个目录承载每个算子所有的交付件，包括代码实现、example、文档等。
> - 算子IR（Intermediate Representation）：表示算子原型，描述了算子输入、输出、属性等信息，包括数据类型、shape、数据格式等。算子清单中有部分算子定义了IR，表明可通过IR构图方式调用算子。

算子清单如下：

| 算子分类 | 算子目录                                                     | 算子IR                                                       | 说明                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| math     | [abs](../../math/abs/README.md)                              | [Abs](../../math/abs/op_graph/abs_proto.h)                   | 实现张量的绝对值运算。                                       |
| math     | [add_lora](../../math/add_lora/README.md)                    | [AddLora](../../math/add_lora/op_graph/add_lora_proto.h)     | 为神经网络添加LoRA（Low-Rank Adaptation）层功能，通过低秩分解减少参数数量。 |
| math     | [angle_v2](../../math/angle_v2/README.md)                    | [AngleV2](../../math/angle_v2/op_graph/angle_v2_proto.h)     | 计算给定输入张量的element_wise angle（以弧度为单位）。       |
| math     | [cast](../../math/cast/README.md)                            | [Cast](../../math/cast/op_graph/cast_proto.h)                | 实现张量数据类型转换。                                       |
| math     | [diag_v2](../../math/diag_v2/README.md)                      | [DiagV2](../../math/diag_v2/op_graph/diag_v2_proto.h)        | 用于提取对角线元素或构造一个对角矩阵。                       |
| math     | [grouped_bias_add_grad](../../math/grouped_bias_add_grad/README.md) | [GroupedBiasAddGrad](../../math/grouped_bias_add_grad/op_graph/grouped_bias_add_grad_proto.h) | 分组偏置加法（GroupedBiasAdd）的反向传播。                   |
| math     | hans_decode                                                  | [HansDecode](../../math/hans_decode/op_graph/hans_decode_proto.h) | 对压缩后的张量基于PDF进行解码，同时基于mantissa重组恢复张量。 |
| math     | hans_encode                                                  | [HansEncode](../../math/hans_encode/op_graph/hans_encode_proto.h) | 对输入张量指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。 |
| math     | [histogram_v2](../../math/histogram_v2/README.md)            | [HistogramV2](../../math/histogram_v2/op_graph/histogram_v2_proto.h) | 计算张量值分布的函数。                                       |
| math     | [is_finite](../../math/is_finite/README.md)                  | [IsFinite](../../math/is_finite/op_graph/is_finite_proto.h)  | 判断张量中哪些元素是有限数值，即不是inf、-inf或nan。         |
| math     | [is_inf](../../math/is_inf/README.md)                        | [IsInf](../../math/is_inf/op_graph/is_inf_proto.h)           | 判断张量中哪些元素是无限大值，即是inf、-inf。                |
| math     | [lin_space](../../math/lin_space/README.md)                  | [LinSpace](../../math/lin_space/op_graph/lin_space_proto.h)  | 生成一个等间隔数值序列。                                     |
| math     | right_shift                                                  | [RightShift](../../math/right_shift/op_graph/right_shift_proto.h) | 对张量执行按位右移操作。                                     |
