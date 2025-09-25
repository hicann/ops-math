# 算子接口（aclnn）

为方便调用算子，提供一套基于C的API（以aclnn为前缀API），无需提供IR（Intermediate Representation）定义，方便高效构建模型与应用开发，该方式被称为“单算子API调用”，简称aclnn调用。

算子接口列表如下：

| 接口名                                                       | 说明                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [aclnnAbs](../../math/abs/docs/aclnnAbs.md)                  | 实现张量的绝对值运算。                                       |
| [aclnnAcos&aclnnInplaceAcos](../../math/acos/docs/aclnnAcos&aclnnInplaceAcos.md) | 对输入矩阵的每个元素进行反余弦操作后输出。|
| [aclnnAcosh&aclnnInplaceAcosh](../../math/acosh/docs/aclnnAcosh&aclnnInplaceAcosh.md) | 对输入Tensor中的每个元素进行反双曲余弦操作后输出。|
| [aclnnAdd&aclnnInplaceAdd](../../math/add/docs/aclnnAdd&aclnnInplaceAdd.md)     | 完成加法计算。 |
| [aclnnAddcdiv&aclnnInplaceAddcdiv](../../math/addcdiv/docs/aclnnAdd&aclnnAddcdiv&aclnnInplaceAddcdiv.md)     | 完成加法计算。 |
| [aclnnAddcmul&aclnnInplaceAddcmul](../../math/addcmul/docs/aclnnAdd&aclnnAddcdiv&aclnnAddcmul&aclnnInplaceAddcmul.md)     | 执行 tensor1 与 tensor2 的逐元素乘法，将结果乘以标量值value并与输入self/selfRef做逐元素加法。 |
| [aclnnAddLora](../../math/add_lora/docs/aclnnAddLora.md)     | 为神经网络添加LoRA（Low-Rank Adaptation）层功能，通过低秩分解减少参数数量。 |
| [aclnnAddr&aclnnInplaceAddr](../../math/addr/docs/aclnnAddr&aclnnInplaceAddr.md)| 求一维向量vec1和vec2的外积得到一个二维矩阵，并将外积结果矩阵乘一个系数后和自身乘系数相加后输出。|
| [aclnnAll](../../math/reduce_all/docs/aclnnAll.md)|对于给定维度dim中的每一维，如果输入Tensor中该维度对应的所有元素计算为True，则返回True，否则返回False。|
| [aclnnAmax](../../math/reduce_max/docs/aclnnAmax.md)|返回张量在指定维度(dim)上每个切片的最大值。|
| [aclnnAmin](../../math/reduce_min/docs/aclnnAmin.md)|返回张量在指定维度(dim)上每个切片的最小值。|
| [aclnnAminmax](../../math/reduce_min/docs/aclnnAminmax.md)|返回输入张量在指定维度上每行的最小值和最大值。|
| [aclnnAminmaxAll](../../math/reduce_min/docs/aclnnAminmaxAll.md)|返回输入张量在所有维度上的最小值和最大值。|
| [aclnnAminmaxDim](../../math/reduce_min/docs/aclnnAminmaxDim.md)|返回输入张量在指定维度上的最小值和最大值。|
| [aclnnAny](../../math/reduce_any/docs/aclnnAny.md)|对于给定维度dim中的每一维，如果输入Tensor中该维度对应的任意元素计算为True，则返回True，否则返回False。|
| [aclnnArange](../../math/range/docs/aclnnArange.md)|从start起始到end结束按照step的间隔获取值，并保存到输出的1维张量，其中数据范围为[start,end)。|
| [aclnnArgMax](../../math/arg_max_v2/docs/aclnnArgMax.md)|返回张量在指定维度（dim）上的最大值的索引，并保存到out张量中。|
| [aclnnArgMin](../../math/arg_min/docs/aclnnArgMin.md)|返回tensor中指定轴的最小值索引，并保存到out中。|
| [aclnnAsin&aclnnInplaceAsin](../../math/asin/docs/aclnnAsin&aclnnInplaceAsin.md)|对输入矩阵的每个元素进行反正弦操作后输出。|
| [aclnnAsinh&aclnnInplaceAsinh](../../math/asinh/docs/aclnnAsinh&aclnnInplaceAsinh.md)|对输入Tensor中的每个元素进行反双曲正弦操作后输出。|
| [aclnnAtan&aclnnInplaceAtan](../../math/atan/docs/aclnnAtan&aclnnInplaceAtan.md)|对输入矩阵的每个元素进行反正切操作后输出。|
| [aclnnAtan2&aclnnInplaceAtan2](../../math/atan2/docs/aclnnAtan2&aclnnInplaceAtan2.md)|对输入张量self和other进行逐元素的反正切运算。|
| [aclnnAtanh&aclnnInplaceAtanh](../../math/atanh/docs/aclnnAtanh&aclnnInplaceAtanh.md)|对输入Tensor中的每个元素进行反双曲正切操作后输出。|
| [aclnnBatchNormStats](../../math/reduce_std_with_mean/docs/aclnnBatchNormStats.md)|计算单卡输入数据的均值和标准差的倒数。|
| [aclnnBincount](../../math/bincount/docs/aclnnAtanh&aclnnBincount.md)|计算非负整数数组中每个数的频率。|
| [aclnnBitwiseNot](../../math/bitwise_not/docs/aclnnBitwiseNot.md)|输入为BOOL型tensor时，进行逻辑非运算；输入为INT型时进行按位非运算。|
| [aclnnBitwiseAndScalar](../../math/bitwise_and/docs/aclnnAtanh&aclnnBitwiseAndScalar.md)|计算输入tensor中每个元素和输入标量的按位与结果。|
| [aclnnBitwiseAndTensor](../../math/bitwise_and/docs/aclnnAtanh&aclnnBitwiseAndTensor.md)|输入为BOOL型tensor时，进行逻辑与运算；输入为INT型时，进行位与运算。|
| [aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar](../../math/bitwise_or/docs/aclnnBitwiseOrScalar&aclnnInplaceBitwiseOrScalar.md)|计算输入张量self中每个元素和输入标量other的按位或。|
| [aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor](../../math/bitwise_or/docs/aclnnBitwiseOrTensor&aclnnInplaceBitwiseOrTensor.md)|计算张量self中每个元素与other张量中对应位置的元素的按位或。|
| [aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar](../../math/bitwise_xor/docs/aclnnBitwiseXorScalar&aclnnInplaceBitwiseXorScalar.md)|计算输入张量self中每个元素和输入标量other的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑异或。|
| [aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor](../../math/bitwise_xor/docs/aclnnBitwiseXorTensor&aclnnInplaceBitwiseXorTensor.md)|计算输入张量self中每个元素与输入张量other中对应位置元素的按位异或，输入self和other必须是整数或布尔类型，对于布尔类型，计算逻辑异或。|
| [aclnnCast](../../math/cast/docs/aclnnCast.md)               | 实现张量数据类型转换。                                       |
| [aclnnCat](../../conversion/concat/docs/aclnnCat.md)|ircular_pad2d的反向传播, 前向计算参考aclnnCircularPad2d。|
| [aclnnCeil&aclnnInplaceCeil](../../math/ceil/docs/aclnnCeil&aclnnInplaceCeil.md)|返回输入tensor中每个元素向上取整的结果。|
| [aclnnChannelShuffle](../../conversion/transpose/docs/aclnnChannelShuffle.md)|将(*, C, H, W)张量的channels分成g个组，然后将每个通道组中的通道进行随机重排，最后将所有通道合并输出，同时保持最终输出张量的shape和输入张量保持一致。|
| [aclnnCircularPad2d](../../conversion/circular_pad/docs/aclnnCircularPad2d.md)|使用输入循环填充输入tensor的最后两维。|
| [aclnnCircularPad3d](../../conversion/circular_pad/docs/aclnnCircularPad3d.md)|使用输入循环填充输入tensor的最后三维。|
| [aclnnCircularPad2dBackward](../../conversion/circular_pad_grad/docs/aclnnCircularPad2dBackward.md)|circular_pad2d的反向传播, 前向计算参考aclnnCircularPad2d。|
| [aclnnCircularPad3dBackward](../../conversion/circular_pad_grad/docs/aclnnCircularPad3dBackward.md)|aclnnCircularPad3d的反向传播。|
| [aclnnClamp](../../conversion/clip_by_value_v2/docs/aclnnClamp.md)|将输入的所有元素限制在[min,max]范围内，如果min为None，则没有下限，如果max为None，则没有上限。|
| [aclnnComplex](../../math/complex/docs/aclnnComplex.md)|输入两个Shape和Dtype一致的Tensor：real和imag。|
| [aclnnConstantPadNd](../../conversion/pad_v3/docs/aclnnConstantPadNd.md)|对输入的张量self，以pad参数为基准进行数据填充，填充值为value。|
| [aclnnContiguous](../../conversion/contiguous/docs/aclnnContiguous.md)|见算子文档。|
| [aclnnCos&aclnnInplaceCos](../../math/cos/docs/aclnnCos&aclnnInplaceCos.md)|对输入矩阵的每个元素进行余弦操作后输出。|
| [aclnnCosh&aclnnInplaceCosh](../../math/cosh/docs/aclnnCosh&aclnnInplaceCosh.md)|双曲函数，根据公式返回一个新的tensor。结果的形状与输入tensor相同。|
| [aclnnCummax](../../math/cummax/docs/aclnnCummax.md)| 计算self中的累积最大值，并返回最大值以及对应的索引。|
| [aclnnCummin](../../math/cummin/docs/aclnnCummin.md)|计算self中的累积最小值，并返回最小值以及对应的索引。|
| [aclnnCumprod&aclnnInplaceCumprod](../../math/cumprod/docs/aclnnCumprod&aclnnInplaceCumprod.md)|新增aclnnCumprod接口，cumprod函数用于计算输入张量在指定维度上的累积乘积。|
| [aclnnCumsum](../../math/cumsum/docs/aclnnCumsum.md)|对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。|
| [aclnnCumsumV2](../../math/cumsum/docs/aclnnCumsumV2.md)|对输入张量self的元素，按照指定维度dim依次进行累加，并将结果保存到输出张量out中。|
| [aclnnDiagFlat](../../conversion/diag_flat/docs/aclnnDiagFlat.md)|生成对角线张量。如果输入self为一维张量，则返回二维张量，self里元素为对角线值；如果输入self是二维及以上张量，则先进行扁平化（化简为一维张量），再转化为第一种场景处理。|
| [aclnnDiv&aclnnInplaceDiv](../../math/div/docs/aclnnDiv&aclnnInplaceDiv.md)|完成除法计算。|
| [aclnnDot](../../math/dot/docs/aclnnDot.md)|计算两个一维张量的点积结果。|
| [aclnnEqScalar&aclnnInplaceEqScalar](../../math/equal/docs/aclnnEqScalar&aclnnInplaceEqScalar.md)|计算self中的元素的值与other的值是否相等，将self每个元素与other的值的比较结果写入out中。|
| [aclnnEqTensor&aclnnInplaceEqTensor](../../math/equal/docs/aclnnEqTensor&aclnnInplaceEqTensor.md)|计算两个Tensor中的元素是否相等，返回一个Tensor，self=other的为True(1.)，否则为False(0.)。|
| [aclnnEqual](../../math/equal/docs/aclnnEqual.md)|计算两个Tensor是否有相同的大小和元素，返回一个Bool类型。|
| [aclnnErf&aclnnInplaceErf](../../math/erf/docs/aclnnErf&aclnnInplaceErf.md)|返回输入Tensor中每个元素对应的误差函数的值。|
| [aclnnErfc&aclnnInplaceErfc](../../math/erfc/docs/aclnnErfc&aclnnInplaceErfc.md)| 返回输入Tensor中每个元素对应的误差互补函数的值。|
| [aclnnExp&aclnnInplaceExp](../../math/exp/docs/aclnnExp&aclnnInplaceExp.md)|返回一个新的张量，该张量的每个元素都是输入张量对应元素的指数。|
| [aclnnExpand](../../math/expand/docs/aclnnExpand.md)| 将输入张量self广播成指定size的张量。|
| [aclnnExpm1&aclnnInplaceExpm1](../../math/expm1/docs/aclnnExpm1&aclnnInplaceExpm1.md)|以输入的self为指数，计算自然常数e的幂，并对指数计算结果进行减1计算。|
| [aclnnExp2&aclnnInplaceExp2](../../math/pow/docs/aclnnExp2&aclnnInplaceExp2.md)|self每个元素作为基数2的幂完成计算。|
| [aclnnExpSegsum](../../math/segsum/docs/aclnnExpSegsum.md)|进行分段和计算。生成对角线为0的半可分矩阵，且上三角为-inf。|
| [aclnnEye](../../math/eye/docs/aclnnEye.md)|返回一个二维张量，该张量的对角线上元素值为1，其余元素值为0。|完成除法计算，对余数向下取整。|
| [aclnnFlatten](../../conversion/flatten/docs/aclnnFlatten.md)|将输入Tensor，基于给定的axis，扁平化为一个2D的Tensor。|
| [aclnnFloor&aclnnInplaceFloor](../../math/floor/docs/aclnnFloor&aclnnInplaceFloor.md)|返回输入Tensor中每个元素向下取整，并将结果回填到输入Tensor中。|
| [aclnnFloorDivide&aclnnInplaceFloorDivide](../../math/floor_div/docs/aclnnFloorDivide&aclnnInplaceFloorDivide.md)|
| [aclnnFloorDivides&aclnnInplaceFloorDivides](../../math/floor_div/docs/aclnnFloorDivides&aclnnInplaceFloorDivides.md)|完成除法计算，对余数向下取整。|
| [aclnnFmodScalar&aclnnInplaceFmodScalar](../../math/mod/docs/aclnnFmodScalar&aclnnInplaceFmodScalar.md)|返回 self除以other的余数。|
| [aclnnFmodTensor&aclnnInplaceFmodTensor](../../math/mod/docs/aclnnFmodTensor&aclnnInplaceFmodTensor.md)|返回self除以other的余数。|
| [aclnnFrac&aclnnInplaceFrac](../../math/sub/docs/aclnnFrac&aclnnInplaceFrac.md)|计算输入Tensor中每个元素的小数部分后输出。|
| [aclnnGcd](../../math/gcd/docs/aclnnGcd.md)|对给定的self和other计算element-wise维度的最大公约数，其中self和other都需要为整数。|
| [aclnnGer](../../math/ger/docs/aclnnGer.md)|实现self和vec2的内积。|
| [aclnnGeScalar&aclnnInplaceGeScalar](../../math/greater_equal/docs/aclnnGeScalar&aclnnInplaceGeScalar.md)|判断输入Tensor中的每个元素是否大于等于other Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于等于判断是否成立。|
| [aclnnGlobalAveragePool](../../math/reduce_mean/docs/aclnnGlobalAveragePool.md)| 传入一个输入张量X，并在同一通道中的值上应用平均池化。|
| [aclnnGlobalMaxPool](../../math/reduce_max/docs/aclnnGlobalMaxPool.md)|输入一个张量，并对同一通道中的值取最大值。|
| [aclnnGtScalar&aclnnInplaceGtScalar](../../math/greater/docs/aclnnGtScalar&aclnnInplaceGtScalar.md)|判断输入Tensor中的每个元素是否大于other Scalar的值，返回一个Bool类型的Tensor，对应输入Tensor中每个位置的大于判断是否成立。|
| [aclnnGroupedBiasAddGrad](../../math/grouped_bias_add_grad/docs/aclnnGroupedBiasAddGrad.md) | 分组偏置加法（GroupedBiasAdd）的反向传播。                   |
| [aclnnHansDecode](../../math/hans_decode/docs/aclnnHansDecode.md) | 对压缩后的张量基于PDF进行解码，同时基于mantissa（尾数）重组恢复张量。 |
| [aclnnHansEncode](../../math/hans_encode/docs/aclnnHansEncode.md) | 对张量的指数位所在字节实现PDF统计，按PDF分布统计进行无损压缩。 |
| [aclnnHardtanh&aclnnInplaceHardtanh](../../conversion/clip_by_value_v2/docs/aclnnHardtanh&aclnnInplaceHardtanh.md)|将输入的所有元素限制在[clipValueMin,clipValueMax]范围内，若元素大于clipValueMax则限制为clipValueMax，若元素小于clipValueMin则限制为clipValueMin，否则等于元素本身。|
| [aclnnIm2col](../../conversion/im2col/docs/aclnnIm2col.md)|图像到列，滑动局部窗口数据转为列向量，拼接为大张量。从批处理输入张量中提取滑动窗口。|
| [aclnnInplaceCopy](../../conversion/view_copy/docs/aclnnInplaceCopy.md)|将src中的元素复制到selfRef张量中并返回selfRef。|
| [aclnnInplaceFillDiagonal](../../conversion/fill_diagonal_v2/docs/aclnnInplaceFillDiagonal.md)|以fillValue填充tensor对角线。|
| [aclnnInplaceFillScalar](../../conversion/fill/docs/aclnnInplaceFillScalar.md)|对tensor填充指定标量。|
| [aclnnInplaceFillTensor](../../conversion/fill/docs/aclnnInplaceFillTensor.md)|对selfRef张量填充value， value是张量。|
| [aclnnInplaceMaskedFillScalar](../../conversion/masked_fill/docs/aclnnInplaceMaskedFillScalar.md)|用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。|
| [aclnnInplaceMaskedFillTensor](../../conversion/masked_fill/docs/aclnnInplaceMaskedFillTensor.md)|用value填充selfRef里面与mask矩阵中值为true的位置相对应的元素。|
| [aclnnInplaceOne](../../math/ones_liek/docs/aclnnInplaceOne.md)|返回形状和类型相同的张量，所有元素都设置为1。|
| [aclnnInplaceZero](../../math/zero_op/docs/aclnnInplaceZero.md)|将selfRef张量填充为全零。|
| [aclnnIsClose](../../math/is_close/docs/aclnnIsClose.md)|返回一个带有布尔元素的新张量，判断给定的self和other是否彼此接近，如果值接近，则返回True，否则返回False。|
| [aclnnIsFinite](../../math/is_finite/docs/aclnnIsFinite.md)  | 判断张量中哪些元素是有限数值，即不是inf、-inf或nan。         |
| [aclnnIsInf](../../math/is_inf/docs/aclnnIsInf.md)           | 判断张量中哪些元素是无限大值，即是inf、-inf。                |
| [aclnnIsInScalarTensor](../../math/equal/docs/aclnnIsInScalarTensor.md)|检查element中的元素是否等于testElement。|
| [aclnnIsNegInf](../../math/is_neg_inf/docs/aclnnIsNegInf.md)|判断输入张量的元素是否为负无穷。|
| [aclnnIsPosInf](../../math/is_pos_inf/docs/aclnnIsPosInf.md)|判断输入张量的元素是否为正无穷。|
| [aclnnKlDiv](../../math/kl_div_v2/docs/aclnnKlDiv.md)|计算KL散度。|
| [aclnnLerp&aclnnInplaceLerp](../../math/lerp/docs/aclnnLerp&aclnnInplaceLerp.md)|根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。|
| [aclnnLerps&aclnnInplaceLerps](../../math/lerp/docs/aclnnLerps&aclnnInplaceLerps.md)|根据给定的权重，在起始和结束Tensor之间进行线性插值，返回插值后的Tensor。|
| [aclnnLeScalar&aclnnInplaceLeScalar](../../math/less_equal/docs/aclnnLeScalar&aclnnInplaceLeScalar.md)|判断输入self中的元素值是否小于等于other的值，并将self的每个元素的值与other值的比较结果写入out中。|
| [aclnnLeTensor&aclnnInplaceLeTensor](../../math/less_equal/docs/aclnnLeTensor&aclnnInplaceLeTensor.md)|计算self中的元素值是否小于等于other的值，将self每个元素与other值的比较结果写入out中。|
| [aclnnLinspace](../../math/lin_space/docs/aclnnLinspace.md)  | 生成一个等间隔数值序列。                                     |
| [aclnnLog&aclnnInplaceLog](../../math/log/docs/aclnnLog&aclnnInplaceLog.md)  | 完成自然对数的计算                                    |
| [aclnnLog2&aclnnInplaceLog2](../../math/log/docs/aclnnLog2&aclnnInplaceLog2.md)  | 完成以2为底的对数计算          |
| [aclnnLog10&aclnnInplaceLog10](../../math/log/docs/aclnnLog10&aclnnInplaceLog10.md)  | 完成输入以10为底的对数计算          |
| [aclnnLog1p&aclnnInplaceLog1p](../../math/log1p/docs/aclnnLog1p&aclnnInplaceLog1p.md)  | 对输入Tensor完成log1p运算。          |
| [aclnnLogicalAnd&aclnnInplaceLogicalAnd](../../math/logical_and/docs/aclnnLogicalAnd&aclnnInplaceLogicalAnd.md)|完成给定输入张量元素的逻辑与运算。0被视为False，非0被视为True。|
| [aclnnLogicalNot&aclnnInplaceLogicalNot](../../math/logical_not/docs/aclnnLogicalNot&aclnnInplaceLogicalNot.md)|计算给定输入Tensor的逐元素逻辑非。如果未指定输出类型，输出Tensor是bool类型。|
| [aclnnLogicalOr&aclnnInplaceLogicalOr](../../math/logical_or/docs/aclnnLogicalOr&aclnnInplaceLogicalOr.md)|完成给定输入张量元素的逻辑或运算。当两个输入张量为非bool类型时，0被视为False，非0被视为True。|
| [aclnnLogSumExp](../../math/reduce_log_sum_exp/docs/aclnnLogSumExp.md)|返回输入tensor指定维度上的指数之和的对数。|
| [aclnnLtScalar&aclnnInplaceLtScalar](../../math/less/docs/aclnnLtScalar&aclnnInplaceLtScalar.md)|判断输入self中的每个元素是否小于输入other的值，返回一个Bool类型的Tensor。|
| [aclnnLtTensor&aclnnInplaceLtTensor](../../math/less/docs/aclnnLtTensor&aclnnInplaceLtTensor.md)|判断输入self中的每个元素是否小于输入other中的元素，返回一个Bool类型的Tensor。|
| [aclnnMaskedSelect](../../conversion/masked_select_v3/docs/aclnnMaskedSelect.md)|根据一个布尔掩码张量（mask）中的值选择输入张量（self）中的元素作为输出，形成一个新的一维张量。|
| [aclnnMatmulCompressDequant](../../conversion/matmul_v2_compress_dequant/docs/aclnnMatmulCompressDequant.md)|进行l@r矩阵乘计算时，可先通过msModelSlim工具对r矩阵进行无损压缩，减少r矩阵的内存占用大小，然后通过本接口完成无损解压缩，矩阵乘，反量化计算。|
| [aclnnMaxDim](../../math/arg_max_with_value/docs/aclnnMaxDim.md)|返回Tensor指定维度的最大值及其索引位置。|
| [aclnnMaskedScale](../../math/masked_scale/docs/aclnnMaskedScale.md)|完成elementwise计算。|
| [aclnnMax](../../math/reduce_max/docs/aclnnMax.md)|返回Tensor所有元素中的最大值。|
| [aclnnMaximum](../../math/maximum/docs/aclnnMaximum.md)|计算两个张量中每个元素的最大值，并返回一个新的张量。|
| [aclnnMaxN](../../math/maximum/docs/aclnnMaxN.md)|对输入tensor列表中每个输入tensor对应元素求max。|
| [aclnnMaxV2](../../math/reduce_max/docs/aclnnMaxV2.md)|按指定维度对输入tensor求元素最大值。|
| [aclnnMean](../../math/reduce_mean/docs/aclnnMean.md)|按指定维度对Tensor求均值。|
| [aclnnMeanV2](../../math/reduce_mean/docs/aclnnMeanV2.md)|按指定维度对Tensor求均值。|
| [aclnnMin](../../math/reduce_min/docs/aclnnMin.md)|返回Tensor所有元素中的最小值。|
| [aclnnMinDim](../../math/arg_min_with_value/docs/aclnnMinDim.md)|返回self中指定维度的最小值及其索引位置。|
| [aclnnMinimum](../../math/minimum/docs/aclnnMinimum.md)|计算两个张量中每个元素的最小值，并返回一个新的张量。|
| [aclnnMinN](../../math/minimum/docs/aclnnMinN.md)|对输入tensor列表中每个输入tensor对应元素求min。|
| [aclnnMul&aclnnInplaceMul](../../math/mul/docs/aclnnMul&aclnnInplaceMul.md)|完成tensor与tensor间的乘法计算。|
| [aclnnMuls&aclnnInplaceMuls](../../math/muls/docs/aclnnMuls&aclnnInplaceMuls.md)|完成tensor与scalar间的乘法计算。|
| [aclnnNanToNum&aclnnInplaceNanToNum](../../math/nan_to_num/docs/aclnnNanToNum&aclnnInplaceNanToNum.md)|将输入中的NaN、正无穷大和负无穷大值分别替换为nan、posinf、neginf指定的值。|
| [aclnnNeg&aclnnInplaceNeg](../../math/neg/docs/aclnnNeg&aclnnInplaceNeg.md)|对输入的每个元素完成相反数计算。|
| [aclnnNeScalar&aclnnInplaceNeScalar](../../math/not_equal/docs/aclnnNeScalar&aclnnInplaceNeScalar.md)|计算selfRef中的元素的值与other的值是否不相等。|
| [aclnnNeTensor&aclnnInplaceNeTensor](../../math/not_equal/docs/aclnnNeTensor&aclnnInplaceNeTensor.md)|计算self（selfRef）中的元素的值与other的值是否不相等。|
| [aclnnNpuFormatCast](../../conversion/npu_format_cast/docs/aclnnNpuFormatCast.md)|完成ND←→NZ的转换功能。|
| [aclnnOneHot](../../math/one_hot/docs/aclnnOneHot.md)|[aclnnOneHot](../../math/one_hot/docs/aclnnOneHot.md)|
| [aclnnPdist](../../math/pdist/docs/aclnnPdist.md)|计算输入self中每对行向量的p范数距离。|
| [aclnnPdistForward](../../math/pdist/docs/aclnnPdistForward.md)|计算输入self中每对行向量的p范数距离。|
| [aclnnPermute](../../conversion/transpose/docs/aclnnPermute.md)|对tensor的任意维度进行调换。如输入self是shape为[2, 3, 5]的tensor，dims为(2, 0, 1)，则输出是shape为[5, 2, 3]的tensor。|
| [aclnnPowScalarTensor](../../math/pow/docs/aclnnPowScalarTensor.md)|exponent每个元素作为input对应元素的幂完成计算。|
| [aclnnPowTensorScalar&aclnnInplacePowTensorScalar](../../math/pow/docs/aclnnPowTensorScalar&aclnnInplacePowTensorScalar.md)|exponent每个元素作为input对应元素的幂完成计算。|
| [aclnnPowTensorTensor&aclnnInplacePowTensorTensor](../../math/pow/docs/aclnnPowTensorTensor&aclnnInplacePowTensorTensor.md)|exponent每个元素作为input对应元素的幂完成计算。|
| [aclnnProd](../../math/reduce_prod/docs/aclnnProd.md)|回输入tensor中所有元素的乘积。|
| [aclnnProdDim](../../math/reduce_prod/docs/aclnnProdDim.md)|返回输入tensor给定维度上每行的乘积。|
| [aclnnRange](../../math/range/docs/aclnnRange.md)|从start起始到end结束按照step的间隔取值。|
| [aclnnReal](../../math/real/docs/aclnnReal.md)|为输入张量的每一个元素取实数部分。|
| [aclnnReciprocal&aclnnInplaceReciprocal](../../math/reciprocal/docs/aclnnReciprocal&aclnnInplaceReciprocal.md)|返回一个具有每个输入元素倒数的新张量。|
| [aclnnReduceLogSum](../../math/reduce_log_sum/docs/aclnnReduceLogSum.md)|返回给定维度中输入张量每行的和再取对数。|
| [aclnnReduceNansum](../../math/reduce_nansum/docs/aclnnReduceNansum.md)|将tensor中NaN处理为0后，返回输入tensor给定维度上的和。|
| [aclnnReduceSum](../../math/reduce_sum_op/docs/aclnnReduceSum.md)|返回给定维度中输入张量每行的和。|
| [aclnnReflectionPad1d](../../conversion/mirror_pad/docs/aclnnReflectionPad1d.md)|使用输入边界的反射填充输入tensor。|
| [aclnnReflectionPad2d](../../conversion/mirror_pad/docs/aclnnReflectionPad2d.md)|使用输入边界的反射填充输入tensor。|
| [aclnnReflectionPad3d](../../conversion/mirror_pad/docs/aclnnReflectionPad3d.md)| 3D反射填充。|
| [aclnnRemainderScalarTensor](../../math/floor_mod/docs/aclnnRemainderScalarTensor.md)|将scalar self进行broadcast成和tensor other一样shape的tensor以后，其中的每个元素都转换为除以other的对应元素以后得到的余数。|
| [aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar](../../math/floor_mod/docs/aclnnRemainderTensorScalar&aclnnInplaceRemainderTensorScalar.md)|将tensor self中的每个元素都转换为除以scalar other以后得到的余数。|
| [aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor](../../math/floor_mod/docs/aclnnRemainderTensorTensor&aclnnInplaceRemainderTensorTensor.md)|将tensor self和tensor other进行broadcast成一致的shape后，其中的每个元素都转换为除以other的对应元素以后得到的余数。|
| [aclnnRepeat](../../math/tile/docs/aclnnRepeat.md)|对输入tensor沿着repeats中对每个维度指定的复制次数进行复制。|
| [aclnnReplicationPad1d](../../conversion/pad_v3/docs/aclnnReplicationPad1d.md)|使用输入边界填充输入tensor的最后一维。|
| [aclnnReplicationPad2d](../../conversion/pad_v3/docs/aclnnReplicationPad2d.md)|使用输入边界填充输入tensor的最后两维。|
| [aclnnReplicationPad3d](../../conversion/pad_v3/docs/aclnnReplicationPad3d.md)|使用输入边界填充输入tensor的最后三维。|
| [aclnnReplicationPad1dBackward](../../conversion/pad_v3_grad_replicate/docs/aclnnReplicationPad1dBackward.md)|replication_pad1d的反向传播。|
| [aclnnReplicationPad2dBackward](../../conversion/pad_v3_grad_replicate/docs/aclnnReplicationPad2dBackward.md)|replication_pad2d的反向传播, 前向计算参考aclnnReplicationPad2d。|
| [aclnnReplicationPad3dBackward](../../conversion/pad_v3_grad_replication/docs/aclnnReplicationPad3dBackward.md)|计算aclnnReplicationPad3d的反向传播。|
| [aclnnReflectionPad1dBackward](../../conversion/pad_v4_grad/docs/aclnnReflectionPad1dBackward.md)|reflection_pad1d的反向传播，前向计算参考aclnnReflectionPad1d。|
| [aclnnReflectionPad2dBackward](../../conversion/pad_v4_grad/docs/aclnnReflectionPad2dBackward.md)|reflection_pad2d的反向传播，前向计算参考aclnnReflectionPad2d。|
| [aclnnReflectionPad3dBackward](../../conversion/reflection_pad3d_grad/docs/aclnnReflectionPad3dBackward.md)|计算aclnnReflectionPad3d api的反向传播。|
| [aclnnRingAttentionUpdate](../../math/ring_attention_update/docs/aclnnRingAttentionUpdate.md)|RingAttentionUpdate算子功能是将两次FlashAttention的输出根据其不同的softmax的max和sum更新。|
| [aclnnRoll](../../conversion/roll/docs/aclnnRoll.md)|沿给定尺寸和维度移动Tensor中的数据。|
| [aclnnRound&aclnnInplaceRound](../../math/round/docs/aclnnRound&aclnnInplaceRound.md)|将输入的值舍入到最接近的整数，若该值与两个整数距离一样则向偶数取整。|
| [aclnnRoundDecimals&aclnnInplaceRoundDecimals](../../math/round/docs/aclnnRoundDecimals&aclnnInplaceRoundDecimals.md)|将输入Tensor的元素四舍五入到指定的位数。|
| [aclnnRsqrt&aclnnInplaceRsqrt](../../math/rsqrt/docs/aclnnRsqrt&aclnnInplaceRsqrt.md)|求input(Tensor)每个元素的平方根的倒数。|
| [aclnnRsub](../../math/sub/docs/aclnnRsub.md)|完成减法计算。|
| [aclnnScale](../../math/scale/docs/aclnnScale.md)|参见算子文档。|
| [aclnnSearchSorted](../../math/search_sorted/docs/aclnnSearchSorted.md)|在一个已排序的张量（sortedSequence）中查找给定tensor值（self）应该插入的位置。|
| [aclnnSign](../../math/sign/docs/aclnnSign.md)|对输入的tensor逐元素进行Sign符号函数的运算并输出结果tensor。|
| [aclnnSignBitsPack](../../math/sign_bits_pack/docs/aclnnSignBitsPack.md)|将float16类型或者float32类型的1位Adam打包为uint8。|
| [aclnnSignBitsUnpack](../../math/sign_bits_unpack/docs/aclnnSignBitsUnpack.md)|将uint8类型1位Adam拆包为float32或者float16。|
| [aclnnSilentCheck](../../math/silent_check/docs/aclnnSilentCheck.md)|SilentCheckV2算子功能主要根据输入特征值（val），与绝对阈值、相对阈值比较，来识别是否触发静默检测故障。|
| [aclnnSin&aclnnInplaceSin](../../math/sin/docs/aclnnSin&aclnnInplaceSin.md)|对输入Tensor完成sin运算。|
| [aclnnSinc&aclnnInplaceSinc](../../math/sin/docs/aclnnSinc&aclnnInplaceSinc.md)|对输入Tensor完成sinc运算。|
| [aclnnSinh&aclnnInplaceSinh](../../math/sinh/docs/aclnnSinh&aclnnInplaceSinh.md)|对输入Tensor完成sinh运算。|
| [aclnnSinkhorn](../../math/sinkhorn/docs/aclnnSinkhorn.md) | 计算Sinkhorn距离，可以用于MoE模型中的专家路由。|
| [aclnnSlice](../../conversion/strided_slice_v3/docs/aclnnSlice.md)|在指定维度dim上，根据给定的范围[start,end]和步长step，从输入张量self中提取子张量out。|
| [aclnnSliceV2](../../conversion/strided_slice_v3/docs/aclnnSliceV2.md)|根据给定的维度axes、范围[starts,ends]和步长steps，从输入张量self中提取张量out。|
| [aclnnSort](../../math/sort/docs/aclnnSort.md)|将输入tensor中的元素根据指定维度进行升序/降序， 并且返回对应的index值。|
| [aclnnSplitTensor](../../conversion/split_v/docs/aclnnSplitTensor.md)|将输入self沿dim轴按照splitSections大小均匀切分。|
| [aclnnSplitWithSize](../../conversion/split_v/docs/aclnnSplitWithSize.md)|将输入self沿dim轴切分至splitSize中每个元素的大小。|
| [aclnnSqrt&aclnnInplaceSqrt](../../math/sqrt/docs/aclnnSqrt&aclnnInplaceSqrt.md)|完成非负数平方根计算，负数情况返回nan。|
| [aclnnSquare](../../math/square/docs/aclnnSquare.md)|为输入张量的每一个元素计算平方值。|
| [aclnnStack](../../conversion/pack/docs/aclnnStack.md)| 沿着新维度连接张量序列。|
| [aclnnStackBallQuery](../../conversion/stack_ball_query/docs/aclnnStackBallQuery.md)|见算子文档。|
| [aclnnStd](../../math/reduce_std_v2/docs/aclnnStd.md)|计算指定维度(dim)的标准差，这个dim可以是单个维度，维度列表或者None。|
| [aclnnStdMeanCorrection](../../math/reduce_std_with_mean/docs/aclnnStdMeanCorrection.md)|计算样本标准差和均值。|
| [aclnnStridedSliceAssignV2](../../conversion/strided_slice_assign_v2/docs/aclnnStridedSliceAssignV2.md)|StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。|
| [aclnnSub&aclnnInplaceSub](../../math/sub/docs/aclnnSub&aclnnInplaceSub.md)|完成减法计算，被减数按alpha进行缩放。|
| [aclnnSum](../../math/accumulate_nv2/docs/aclnnSum.md) | 返回输入tensors列表中每个输入tensor依次做add求和。|
| [aclnnSWhere](../../math/select/docs/aclnnSWhere.md)|根据条件选取self或other中元素并返回（支持广播）。|
| [aclnnTan&aclnnInplaceTan](../../math/tan/docs/aclnnTan&aclnnInplaceTan.md)|它计算输入张量self中每个元素的正切值，并将结果存储在输出张量out中。|
| [aclnnTanh&aclnnInplaceTanh](../../math/tanh/docs/aclnnTanh&aclnnInplaceTanh.md)|激活函数。返回与输入tensor shape相同的tensor，对输入tensor进行elementwise的计算。|
| [aclnnTanhBackward](../../math/tanh_grad/docs/aclnnTanhBackward.md)|aclnnTanh的反向。|
| [aclnnTransConvolutionWeight](../../conversion/trans_data/docs/aclnnTransConvolutionWeight.md)|需要和aclnnCalculateConvolutionWeightSize接口配套使用，用于创建一个对于Convolution算子计算性能亲和的weight Tensor。|
| [aclnnTransformBiasRescaleQkv](../../math/transform_bias_rescale_qkv/docs/aclnnTransformBiasRescaleQkv.md)|TransformBiasRescaleQkv 算子是一个用于处理多头注意力机制中查询（Query）、键（Key）、值（Value）向量的接口。|
| [aclnnTransMatmulWeight](../../conversion/trans_data/docs/aclnnTransMatmulWeight.md)|需要和aclnnCalculateMatmulWeightSize、aclnnCalculateMatmulWeightSizeV2接口配套使用，用于创建一个对于Matmul算子计算性能亲和的weight Tensor。|
| [aclnnTriangularSolve](../../math/triangular_solve/docs/aclnnTriangularSolve.md)|求解一个具有方形上或下三角形可逆矩阵A和多个右侧b的方程组。|
| [aclnnTrunc&aclnnInplaceTrunc](../../math/trunc/docs/aclnnTrunc&aclnnInplaceTrunc.md)|对输入Tensor完成trunc运算（将数字的小数部分截去，返回整数部分）。|
| [aclnnUnfoldGrad](../../conversion/unfold_grad/docs/aclnnUnfoldGrad.md)|实现Unfold算子的反向功能，计算相应的梯度。|
| [aclnnVar](../../math/reduce_var/docs/aclnnVar.md)|返回输入Tensor指定维度的值求得的方差。|
| [aclnnVarMean](../../math/reduce_var/docs/aclnnVarMean.md)| 返回输入Tensor指定维度的值求得的均值及方差。|
| [aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther](../../math/x_log_y/docs/aclnnXLogYScalarOther&aclnnInplaceXLogYScalarOther.md)|计算self * log(other)的结果。|
| [aclnnXLogYScalarSelf](../../math/x_log_y/docs/aclnnXLogYScalarSelf.md)|计算self * log(other)的结果。|
| [aclnnXLogYTensor&aclnnInplaceXLogYTensor](../../math/x_log_y/docs/aclnnXLogYTensor&aclnnInplaceXLogYTensor.md)|计算self * log(other)的结果。|
| [aclRfft1D](../../math/rfft1_d/docs/aclRfft1D.md)|对输入张量self进行RFFT（傅里叶变换）计算，输出是一个包含非负频率的复数张量。|
| [aclStft](../../math/stft/docs/aclStft.md)|计算输入在滑动窗口内的傅里叶变换。|
