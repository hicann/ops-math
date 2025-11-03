# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](dir_structure.md)。
> - **算子执行位置**：大部分算子运行在AI Core上，少部分算子运行在AI CPU上。默认情况下，项目中提到的算子一般指AI Core算子。
> - 关于AI Core和AI CPU详细介绍请参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。

项目提供的所有算子分类和算子列表如下：

|  算子分类  |   算子目录   |    算子执行位置   |     说明     |
|---------|--------------|-------------------|-----------|
| math   | [add_lora](../math/add_lora/README.md)     | AI Core     |  将输入x根据输入索引indices，分别和对应的weightA，weightB相乘，然后将结果累加到输入y上并输出。    |
| math   | [angle_v2](../math/angle_v2/README.md)        | AI Core  |  为输入张量的每一个元素取角度（单位：弧度）。 |
| math   | [diag_v2](../math/diag_v2/README.md)          | AI Core  |  根据输入的二维张量，提取由diagonal指定的对角线元素。 |
| math   | [grouped_bias_add_grad](../math/grouped_bias_add_grad/README.md)        | AI Core | 分组偏置加法（GroupedBiasAdd）的反向计算。 |
| math   | [hans_decode](../math/hans_decode/README.md)          | AI Core | 对压缩后的张量基于PDF进行解码，同时基于mantissa重组恢复张量。 |
| math   | [hans_encode](../math/hans_encode/README.md)       | AI Core  | 对输入张量指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。  |
| math   | [histogram_v2](../math/histogram_v2/README.md)        | AI Core | 计算张量直方图。 |
| math   | [is_finite](../math/is_finite/README.md)               | AI Core | 判断输入张量哪些元素是有限数值，即不是inf、-inf或nan。 |
| math   | [is_inf](../math/is_inf/README.md)         | AI Core   |  判断张量中哪些元素是无限大值，即为inf、-inf。  |
| math   | [lin_space](../math/lin_space/README.md)            | AI Core   |   生成一个等间隔数值序列。创建一个大小为steps的1维向量，其值从start起始到stop结束（包含）线性均匀分布。 |
| math   | [mul_addn](../math/mul_addn/README.md)    | AI Core             | 实现N>=2个mul和addn融合计算，减少搬运时间和内存的占用。       |
| math   | [non_finite_check](../math/non_finite_check/README.md)     | AI Core       | 检测输入tensor_list中是否存在非有限数值（NaN、Inf、-Inf）。      |
| math   | [pows](../math/pows/README.md)                | AI Core | 对input中的每个元素应用指数为exponent的幂运算。 |
| math   | [rfft1_d](../math/rfft1_d/README.md)      | AI Core      | 对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。           |
| math   | [ring_attention_update](../math/ring_attention_update/README.md)   | AI Core    | RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。     |
| math   | [segsum](../math/segsum/README.md)              | AI Core | 进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。|
| math   | [sinkhorn](../math/sinkhorn/README.md)         | AI Core   | 计算Sinkhorn距离，可以用于MoE模型中的专家路由。      |
| math   | [stft](../math/stft/README.md)      | AI Core    | 计算输入在滑动窗口内的傅里叶变换。       |
| math   | [transform_bias_rescale_qkv](../math/transform_bias_rescale_qkv/README.md) | AI Core | 一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口，用于调整这些向量的偏置（Bias）和缩放（Rescale）因子。 |
| conversion   | [circular_pad](../conversion/circular_pad/README.md)       | AI Core   |  使用输入循环填充输入tensor的最后两维。                  |
| conversion   | [circular_pad_grad](../conversion/circular_pad_grad/README.md)   | AI Core   |  circular_pad的反向传播。                       |
| conversion   | [coalesce_sparse](../conversion/coalesce_sparse/README.md)        | AI Core   | 实现对Coo_Tensor优化的方法coalesce()方法。           |
| conversion   | [diag_flat](../conversion/diag_flat/README.md)      | AI Core      | 创建一个以输入数组为对角线元素的平铺对角矩阵。        |
| conversion   | [feeds_repeat](../conversion/feeds_repeat/README.md)      | AI Core   | 对于输入feeds，根据输入feeds_repeat_times，将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小。     |
| conversion   | [fill_diagonal_v2](../conversion/fill_diagonal_v2/README.md)    | AI Core | 将指定值填充到矩阵的主对角线上。             |
| conversion   | [masked_select_v3](../conversion/masked_select_v3/README.md)    | AI Core    | 根据mask是否为True，选出input中对应位置的值，input和mask满足广播规则，结果为一维Tensor。   |
| conversion   | [pad_v3_grad_replicate](../conversion/pad_v3_grad_replicate/README.md)     | AI Core   | padv3 2D的反向传播。                 |
| conversion   | [pad_v3_grad_replication](../conversion/pad_v3_grad_replication/README.md)      | AI Core   | padv3 3D的反向传播。     |
| conversion   | [pad_v4_grad](../conversion/pad_v4_grad/README.md)        | AI Core      | pad之后的输入的反向传播。   |
| conversion   | [reflection_pad3d_grad](../conversion/reflection_pad3d_grad/README.md)     | AI Core    | 计算aclnnReflectionPad3d api的反向传播。             |
| conversion   | [stack_ball_query](../conversion/stack_ball_query/README.md)       | AI Core   | Stack Ball Query 是KNN的替代方案，用于查找点p1指定半径范围内的所有点(在实现中设置了K的上限)。          |
| conversion   | [strided_slice_assign_v2](../conversion/strided_slice_assign_v2/README.md) | AI Core    | StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。   |
| conversion   | [transpose_v2](../conversion/transpose_v2/README.md)       | AI Core     | 实现张量的维度置换（Permutation）操作，按照指定的顺序重新排列输入张量的维度。        |
| conversion   | [unfold_grad](../conversion/unfold_grad/README.md)       | AI Core     | 实现Unfold算子的反向功能，计算相应的梯度。       |
| math   | [abs](../math/abs/README.md)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [accumulate_nv2](../math/accumulate_nv2)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [acos](../math/acos)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [acosh](../math/acosh)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [add](../math/add)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [addcdiv](../math/addcdiv)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [addcmul](../math/addcmul)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [addr](../math/addr)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [arg_max_v2](../math/arg_max_v2)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [arg_max_with_value](../math/arg_max_with_value)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [arg_min](../math/arg_min)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [arg_min_with_value](../math/arg_min_with_value)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [asin](../math/asin)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [asinh](../math/asinh)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [atan](../math/atan)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [atan2](../math/atan2)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [atanh](../math/atanh)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [axpy](../math/axpy)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [axpy_v2](../math/axpy_v2)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [bincount](../math/bincount)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [bitwise_and](../math/bitwise_and)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [bitwise_not](../math/bitwise_not)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [bitwise_or](../math/bitwise_or)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [bitwise_xor](../math/bitwise_xor)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cast](../math/cast/README.md)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [ceil](../math/ceil)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [complex](../math/complex)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cos](../math/cos)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cosh](../math/cosh)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cummax](../math/cummax)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cummin](../math/cummin)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cumprod](../math/cumprod)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cumsum](../math/cumsum)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [cumsum_cube](../math/cumsum_cube)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [div](../math/div)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [dot](../math/dot)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [equal](../math/equal)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [erf](../math/erf)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [erfc](../math/erfc)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [exp](../math/exp)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [expand](../math/expand)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [expm1](../math/expm1)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [eye](../math/eye)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [floor](../math/floor)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [floor_div](../math/floor_div)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [floor_mod](../math/floor_mod)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [gcd](../math/gcd)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [ger](../math/ger)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [greater](../math/greater)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [greater_equal](../math/greater_equal)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [invert](../math/invert)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [is_close](../math/is_close)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [is_neg_inf](../math/is_neg_inf)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [is_pos_inf](../math/is_pos_inf)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [kl_div_v2](../math/kl_div_v2)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [lerp](../math/lerp)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [less](../math/less)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [less_equal](../math/less_equal)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [log](../math/log)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [log1p](../math/log1p)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [log_add_exp](../math/log_add_exp)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [logical_and](../math/logical_and)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [logical_not](../math/logical_not)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [logical_or](../math/logical_or)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [masked_scale](../math/masked_scale)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [maximum](../math/maximum)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [minimum](../math/minimum)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [mod](../math/mod)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [mul](../math/mul)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [muls](../math/muls)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [nan_to_num](../math/nan_to_num)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [neg](../math/neg)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [not_equal](../math/not_equal)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [one_hot](../math/one_hot)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [ones_like](../math/ones_like)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [pdist](../math/pdist)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [pow](../math/pow)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [range](../math/range)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [real](../math/real)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [real_div](../math/real_div)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reciprocal](../math/reciprocal)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_all](../math/reduce_all)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_any](../math/reduce_any)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_log_sum](../math/reduce_log_sum)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_log_sum_exp](../math/reduce_log_sum_exp)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_max](../math/reduce_max)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_mean](../math/reduce_mean)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_min](../math/reduce_min)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_nansum](../math/reduce_nansum)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_prod](../math/reduce_prod)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_std_v2](../math/reduce_std_v2)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_std_v2_update](../math/reduce_std_v2_update)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_std_with_mean](../math/reduce_std_with_mean)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_sum_op](../math/reduce_sum_op)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [reduce_var](../math/reduce_var)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [right_shift](../math/right_shift/README.md)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [round](../math/round)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [rsqrt](../math/rsqrt)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [scale](../math/scale)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [search_sorted](../math/search_sorted/README.md)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [select](../math/select)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sign](../math/sign)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sign_bits_pack](../math/sign_bits_pack)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sign_bits_unpack](../math/sign_bits_unpack)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [silent_check](../math/silent_check)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sin](../math/sin)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sinh](../math/sinh)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sort](../math/sort)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sqrt](../math/sqrt)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [square](../math/square)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [sub](../math/sub)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [tan](../math/tan)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [tanh](../math/tanh)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [tanh_grad](../math/tanh_grad)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [tensor_equal](../math/tensor_equal)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [tile](../math/tile)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [triangular_solve](../math/triangular_solve)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [trunc](../math/trunc)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [x_log_y](../math/x_log_y)    | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| math   | [zero_op](../math/zero_op)     | AI Core     | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。    |
| conversion   | [as_strided](../conversion/as_strided)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [broadcast_to](../conversion/broadcast_to)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [clip_by_value_v2](../conversion/clip_by_value_v2)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [concat](../conversion/concat/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [contiguous](../conversion/contiguous)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [fill](../conversion/fill/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [flatten](../conversion/flatten/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [im2col](../conversion/im2col/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [masked_fill](../conversion/masked_fill/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [matmul_v2_compress_dequant](../conversion/matmul_v2_compress_dequant/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [mirror_pad](../conversion/mirror_pad)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [npu_format_cast](../conversion/npu_format_cast)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [pack](../conversion/pack/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [pad](../conversion/pad)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [pad_v3](../conversion/pad_v3)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [pad_v3_grad](../conversion/pad_v3_grad/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [reshape](../conversion/reshape)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [roll](../conversion/roll)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [slice](../conversion/slice)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [split_v](../conversion/split_v/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [squeeze](../conversion/squeeze)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [strided_slice](../conversion/strided_slice)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [strided_slice_v3](../conversion/strided_slice_v3)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [tensor_move](../conversion/tensor_move)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [trans_data](../conversion/trans_data/README.md)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [transpose](../conversion/transpose)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [unsqueeze](../conversion/unsqueeze)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
| conversion   | [view_copy](../conversion/view_copy)       | AI Core   | 该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考[贡献指南](../CONTRIBUTING.md)。                |
