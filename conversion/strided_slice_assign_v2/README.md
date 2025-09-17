# StridedSliceAssignV2
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：StridedSliceAssign是一种张量切片赋值操作，它可以将张量inputValue的内容，赋值给目标张量varRef中的指定位置。

## 参数说明

| 参数名       | 输入/输出/属性 | 描述                                                         | 数据类型                                             | 数据格式 |
| ------------ | -------------- | ------------------------------------------------------------ | ---------------------------------------------------- | -------- |
| varRef       | 输入\|输出张量 | 输入的tensor。                                               | FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8 | ND       |
| inputValue   | 输入张量       | 替换切片的tensor，数据类型需与varRef保持一致，shape需要与varRef计算得出的切片shape保持一致，综合约束请见[约束说明](#约束说明)。 | FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8 | ND       |
| begin        | 输入数组       | 切片位置的起始索引。                                         | INT64                                                | -        |
| end          | 输入数组       | 切片位置的终止索引。                                         | INT64                                                | -        |
| strides      | 输入数组       | 切片的步长。                                                 | INT64                                                | -        |
| axesOptional | 输入数组       | 可选参数，切片的轴。                                         | INT64                                                | -        |

## 约束说明

inputValue的shape第i维的计算公式为：$inputValueShape[i] = \lceil\frac{end[i] - begin[i]}{strides[i]} \rceil$，其中$\lceil x\rceil$ 表示对 $x$向上取整。$end$ 和 $begin$ 为经过特殊值调整后的取值，调整方式为：当 $end[i] < 0$ 时，$end[i]=varShape[i] + end[i]$ ，若仍有$end[i] < 0$，则 $end[i] = 0$ ，当 $end[i] > varShape[i]$ 时， $end[i] = varShape[i]$ 。$begin$ 同理。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_strided_slice_assign_v2](tests/ut/op_kernel/test_strided_slice_assign_v2.cpp) | 通过[aclnnStridedSliceAssignV2](docs/aclnnStridedSliceAssignV2.md)接口方式调用StridedSliceAssignV2算子。 |