# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[目录结构](../../README.md#目录结构)。
> - **算子执行位置**：大部分算子运行在AI Core上，少部分算子运行在AI CPU上。默认情况下，项目中提到的算子一般指AI Core算子。
> - 关于AI Core和AI CPU详细介绍请参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。

项目提供的所有算子分类和算子列表如下：

|  算子分类  |   算子目录   |    算子执行位置   |     说明     |
|---------|--------------|-------------------|-----------|
| math   | [add_lora](../../math/add_lora/README.md)     | AI Core     |  将输入x根据输入索引indices，分别和对应的weightA，weightB相乘，然后将结果累加到输入y上并输出。    |
| math   | [angle_v2](../../math/angle_v2/README.md)        | AI Core  |  为输入张量的每一个元素取角度（单位：弧度）。 |
| math   | [diag_v2](../../math/diag_v2/README.md)          | AI Core  |  根据输入的二维张量，提取由diagonal指定的对角线元素。 |
| math   | [hans_decode](../../math/hans_decode/README.md)          | AI Core | 对压缩后的张量基于PDF进行解码，同时基于mantissa重组恢复张量。 |
| math   | [hans_encode](../../math/hans_encode/README.md)       | AI Core  | 对输入张量指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。  |
| math   | [histogram_v2](../../math/histogram_v2/README.md)        | AI Core | 计算张量直方图。 |
| math   | [grouped_bias_add_grad](../../math/grouped_bias_add_grad/README.md)        | AI Core | 分组偏置加法（GroupedBiasAdd）的反向计算。 |
| math   | [is_finite](../../math/is_finite/README.md)               | AI Core | 判断张量中哪些元素是无限大值，即为inf、-inf。 |
| math   | [is_inf](../../math/is_inf/README.md)         | AI Core   |  判断张量中哪些元素是无限大值，即为inf、-inf。  |
| math   | [lin_space](../../math/lin_space/README.md)            | AI Core   |   生成一个等间隔数值序列。创建一个大小为steps的1维向量，其值从start起始到stop结束（包含）线性均匀分布。 |
| math   | [mul_addn](../../math/mul_addn/README.md)    | AI Core             | 实现N>=2个mul和addn融合计算，减少搬运时间和内存的占用。       |
| math   | [non_finite_check](../../math/non_finite_check/README.md)     | AI Core       | 检测输入tensor_list中是否存在非有限数值（NaN、Inf、-Inf）。      |
| math   | [pows](../../math/pows/README.md)                | AI Core | 对input中的每个元素应用指数为exponent的幂运算。 |
| math   | [rfft1_d](../../math/rfft1_d/README.md)      | AI Core      | 对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。           |
| math   | [ring_attention_update](../../math/ring_attention_update/README.md)   | AI Core    | RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。     |
| math   | [segsum](../../math/segsum/README.md)              | AI Core | 进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。|
| math   | [sinkhorn](../../math/sinkhorn/README.md)         | AI Core   | 计算Sinkhorn距离，可以用于MoE模型中的专家路由。      |
| math   | [stft](../../math/stft/README.md)      | AI Core    | 计算输入在滑动窗口内的傅里叶变换。       |
| math   | [transform_bias_rescale_qkv](../../math/transform_bias_rescale_qkv/README.md) | AI Core | 一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口，用于调整这些向量的偏置（Bias）和缩放（Rescale）因子。 |
| conversion   | [circular_pad](../../conversion/circular_pad/README.md)       | AI Core   |  使用输入循环填充输入tensor的最后两维。                  |
| conversion   | [circular_pad_grad](../../conversion/circular_pad_grad/README.md)   | AI Core   |  circular_pad的反向传播。                       |
| conversion   | [coalesce_sparse](../../conversion/coalesce_sparse/README.md)        | AI Core   | 实现对Coo_Tensor优化的方法coalesce()方法。           |
| conversion   | [diag_flat](../../conversion/diag_flat/README.md)      | AI Core      | 创建一个以输入数组为对角线元素的平铺对角矩阵。        |
| conversion   | [feeds_repeat](../../conversion/feeds_repeat/README.md)      | AI Core   | 对于输入feeds，根据输入feeds_repeat_times，将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小。     |
| conversion   | [fill_diagonal_v2](../../conversion/fill_diagonal_v2/README.md)    | AI Core | 将指定值填充到矩阵的主对角线上。             |
| conversion   | [masked_select_v3](../../conversion/masked_select_v3/README.md)    | AI Core    | 根据mask是否为True，选出input中对应位置的值，input和mask满足广播规则，结果为一维Tensor。   |
| conversion   | [pad_v3_grad_replicate](../../conversion/pad_v3_grad_replicate/README.md)     | AI Core   | padv3 2D的反向传播。                 |
| conversion   | [pad_v3_grad_replication](../../conversion/pad_v3_grad_replication/README.md)      | AI Core   | padv3 3D的反向传播。     |
| conversion   | [pad_v4_grad](../../conversion/pad_v4_grad/README.md)        | AI Core      | pad之后的输入的反向传播。   |
| conversion   | [reflection_pad3d_grad](../../conversion/reflection_pad3d_grad/README.md)     | AI Core    | 计算aclnnReflectionPad3d api的反向传播。             |
| conversion   | [stack_ball_query](../../conversion/stack_ball_query/README.md)       | AI Core   | Stack Ball Query 是KNN的替代方案，用于查找点p1指定半径范围内的所有点(在实现中设置了K的上限)。          |
| conversion   | [strided_slice_assign_v2](../../conversion/strided_slice_assign_v2/README.md) | AI Core    | StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。   |
| conversion   | [transpose_v2](../../conversion/transpose_v2/README.md)       | AI Core     | 实现张量的维度置换（Permutation）操作，按照指定的顺序重新排列输入张量的维度。        |
| conversion   | [unfold_grad](../../conversion/unfold_grad/README.md)       | AI Core     | 实现Unfold算子的反向功能，计算相应的梯度。       |
