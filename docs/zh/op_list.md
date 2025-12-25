# 算子列表

> 说明：
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./context/dir_structure.md)。
> - **算子执行硬件单元**：大部分算子运行在AI Core，少部分算子运行在AI CPU。默认情况下，项目中提到的算子一般指AI Core算子。关于AI Core和AI CPU详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中“概念原理和术语 > 硬件架构与数据处理原理”。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，一般以aclnn为前缀，全量接口参见[aclnn列表](op_api_list.md)。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th>aclnn调用</th>
    <th>图模式调用</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
    <th>op_api</th>
    <th>op_graph</th>
  </tr></thead>
<tbody>
  <tr>
    <td>math</td>
    <td><a href="../../math/add_lora/README.md">add_lora</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将输入x根据输入索引indices，分别和对应的weightA，weightB相乘，然后将结果累加到输入y上并输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/angle_v2/README.md">angle_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>为输入张量的每一个元素取角度（单位：弧度）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/diag_v2/README.md">diag_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>根据输入的二维张量，提取由diagonal指定的对角线元素。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/grouped_bias_add_grad/README.md">grouped_bias_add_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>分组偏置加法（GroupedBiasAdd）的反向计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hans_decode/README.md">hans_decode</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对压缩后的张量基于PDF进行解码，同时基于mantissa重组恢复张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/hans_encode/README.md">hans_encode</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对输入张量指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/histogram_v2/README.md">histogram_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算张量直方图。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_finite/README.md">is_finite</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断输入张量哪些元素是有限数值，即不是inf、-inf或nan。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_inf/README.md">is_inf</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>判断张量中哪些元素是无限大值，即为inf、-inf。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lin_space/README.md">lin_space</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>生成一个等间隔数值序列。创建一个大小为steps的1维向量，其值从start起始到stop结束（包含）线性均匀分布。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul_addn/README.md">mul_addn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>实现N>=2个mul和addn融合计算，减少搬运时间和内存的占用。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/non_finite_check/README.md">non_finite_check</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>检测输入tensor_list中是否存在非有限数值（NaN、Inf、-Inf）。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pows/README.md">pows</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对input中的每个元素应用指数为exponent的幂运算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rfft1_d/README.md">rfft1_d</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ring_attention_update/README.md">ring_attention_update</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/segsum/README.md">segsum</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinkhorn/README.md">sinkhorn</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算Sinkhorn距离，可以用于MoE模型中的专家路由。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/stft/README.md">stft</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI Core</td>
    <td>计算输入在滑动窗口内的傅里叶变换。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/transform_bias_rescale_qkv/README.md">transform_bias_rescale_qkv</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口，用于调整这些向量的偏置（Bias）和缩放（Rescale）因子。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/circular_pad/README.md">circular_pad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>使用输入循环填充输入tensor的最后两维。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/circular_pad_grad/README.md">circular_pad_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>circular_pad的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/coalesce_sparse/README.md">coalesce_sparse</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现对Coo_Tensor优化的方法coalesce()方法。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/diag_flat/README.md">diag_flat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>创建一个以输入数组为对角线元素的平铺对角矩阵。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/feeds_repeat/README.md">feeds_repeat</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>对于输入feeds，根据输入feeds_repeat_times，将对应的feeds的第0维上的数据复制对应的次数，并将输出y的第0维padding到output_feeds_size的大小。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill_diagonal_v2/README.md">fill_diagonal_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>将指定值填充到矩阵的主对角线上。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_select_v3/README.md">masked_select_v3</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>根据mask是否为True，选出input中对应位置的值，input和mask满足广播规则，结果为一维Tensor。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad_replicate/README.md">pad_v3_grad_replicate</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>padv3 2D的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad_replication/README.md">pad_v3_grad_replication</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>padv3 3D的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v4_grad/README.md">pad_v4_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>pad之后的输入的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/reflection_pad3d_grad/README.md">reflection_pad3d_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>计算aclnnReflectionPad3d api的反向传播。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/stack_ball_query/README.md">stack_ball_query</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>Stack Ball Query 是KNN的替代方案，用于查找点p1指定半径范围内的所有点（在实现中设置了K的上限）。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_assign_v2/README.md">strided_slice_assign_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/transpose_v2/README.md">transpose_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现张量的维度置换（Permutation）操作，按照指定的顺序重新排列输入张量的维度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unfold_grad/README.md">unfold_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>实现Unfold算子的反向功能，计算相应的梯度。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat_v2/README.md">concat_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子沿指定维度将多个输入张量进行拼接。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/clip_by_value_v2/README.md">clip_by_value_v2</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子将输入的所有元素限制在[clipValueMin,clipValueMax]范围内。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/invert_permutation/README.md">invert_permutation</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/abs/README.md">abs</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/accumulate_nv2">accumulate_nv2</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acos/README.md">acos</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入的每个元素进行反余弦操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/acosh">acosh</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/add/README.md">add</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于完成加法计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/square/README.md">square</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入Tensor逐元素计算平方值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addcdiv">addcdiv</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addcmul">addcmul</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/addr">addr</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_max_v2">arg_max_v2</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_max_with_value">arg_max_with_value</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_min">arg_min</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/arg_min_with_value">arg_min_with_value</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asin">asin</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/asinh">asinh</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan/README.md">atan</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入的每个元素进行反正切操作后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atan2">atan2</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/atanh">atanh</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/axpy">axpy</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/axpy_v2">axpy_v2</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bincount">bincount</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_and">bitwise_and</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_not">bitwise_not</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_or">bitwise_or</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/bitwise_xor">bitwise_xor</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cast/README.md">cast</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ceil/README.md">ceil</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于返回输入Tensor中每个元素向上取整的结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/complex">complex</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cos">cos</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cosh">cosh</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cummax">cummax</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cummin">cummin</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumprod">cumprod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumsum/README.md">cumsum</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入张量self的元素，按照指定维度dim依次进行累加。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cumsum_cube">cumsum_cube</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/div">div</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/dot">dot</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/equal/README.md">equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子计算两个Tensor是否有相同的大小和元素。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/erf">erf</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/erfc">erfc</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/exp">exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/exp_segsum_grad/README.md">exp_segsum_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>AI Core</td>
    <td>Segsum的反向计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expand">expand</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/expm1">expm1</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/eye">eye</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor">floor</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor_div">floor_div</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/floor_mod">floor_mod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/gcd">gcd</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ger">ger</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/greater/README.md">greater</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于判断输入Tensor中的每个元素是否大于other Scalar的值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/greater_equal">greater_equal</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/invert">invert</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_close">is_close</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_neg_inf">is_neg_inf</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/is_pos_inf">is_pos_inf</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/kl_div_v2">kl_div_v2</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/lerp">lerp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/less/README.md">less</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于判断输入self中的每个元素是否小于输入other的值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/less_equal">less_equal</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log/README.md">log</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入张量x的元素，逐元素进行对数计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log1p">log1p</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/log_add_exp">log_add_exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_and">logical_and</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_not">logical_not</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/logical_or">logical_or</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/maximum">maximum</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/minimum">minimum</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mod">mod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/mul">mul</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/muls">muls</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/nan_to_num">nan_to_num</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/neg">neg</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/not_equal/README.md">not_equal</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子逐元素比较两个输入张量是否不相等。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/one_hot">one_hot</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/ones_like">ones_like</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pdist">pdist</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/pow">pow</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/range">range</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/real">real</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/real_div/README.md">real_div</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子按元素逐个返回x1/x2的结果。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reciprocal">reciprocal</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_all">reduce_all</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_any">reduce_any</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_log_sum">reduce_log_sum</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_log_sum_exp">reduce_log_sum_exp</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_max">reduce_max</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_mean/README.md">reduce_mean</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对一个多维向量按照指定的维度求平均值。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_min">reduce_min</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_nansum">reduce_nansum</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_prod">reduce_prod</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_v2">reduce_std_v2</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_v2_update">reduce_std_v2_update</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_std_with_mean">reduce_std_with_mean</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_sum_op">reduce_sum_op</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/reduce_var">reduce_var</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/right_shift/README.md">right_shift</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子按元素计算输入张量 x 与 y 的按位右移操作。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/round/README.md">round</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子将输入张量的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/rsqrt">rsqrt</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/scale">scale</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/search_sorted/README.md">search_sorted</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于在一个已排序的张量 sorted_sequence 中查找给定张量 values 应该插入的位置。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/select">select</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign">sign</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign_bits_pack">sign_bits_pack</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sign_bits_unpack">sign_bits_unpack</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/silent_check">silent_check</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sin">sin</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sinh/README.md">sinh</a></td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入的每个元素进行余弦后输出。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sort">sort</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sqrt">sqrt</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sub">sub</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tan">tan</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tanh">tanh</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tanh_grad">tanh_grad</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tensor_equal">tensor_equal</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/tile">tile</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/triangular_solve">triangular_solve</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/trunc">trunc</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/x_log_y">x_log_y</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/zero_op">zero_op</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/as_strided">as_strided</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/broadcast_to">broadcast_to</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/concat/README.md">concat</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/contiguous">contiguous</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/fill/README.md">fill</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/flatten/README.md">flatten</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/im2col/README.md">im2col</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/masked_fill/README.md">masked_fill</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/mirror_pad">mirror_pad</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/npu_format_cast">npu_format_cast</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pack/README.md">pack</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad">pad</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3">pad_v3</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/pad_v3_grad/README.md">pad_v3_grad</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/reshape">reshape</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/roll">roll</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/slice">slice</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/split_v/README.md">split_v</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/squeeze">squeeze</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice">strided_slice</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/strided_slice_v3">strided_slice_v3</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/tensor_move">tensor_move</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/trans_data/README.md">trans_data</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/transpose">transpose</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/unsqueeze">unsqueeze</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>conversion</td>
    <td><a href="../../conversion/view_copy">view_copy</a></td>
    <td>×</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI Core</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/inv_grad/README.md">inv_grad</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/cross/README.md">cross</a></td>
    <td>√</td>
    <td>√</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子对输入Tensor完成linear_cross运算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/sqrt_grad/README.md">sqrt_grad</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子用于完成梯度平方根计算。</td>
  </tr>
  <tr>
    <td>math</td>
    <td><a href="../../math/squared_difference">squared_difference</a></td>
    <td>√</td>
    <td>×</td>
    <td>×</td>
    <td>√</td>
    <td>AI CPU</td>
    <td>该算子暂无Ascend C代码实现，欢迎开发者补充贡献，贡献方式参考<a href="../../CONTRIBUTING.md">贡献指南</a>。</td>
  </tr>
</tbody>
</table>