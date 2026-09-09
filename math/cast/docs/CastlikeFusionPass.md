# CastlikeFusionPass

## 融合模式

该融合将图中的CastLike算子替换为Cast算子：CastLike的输入为x（待转换的tensor）和y（目标类型参考tensor），输出为output；融合时从y的数据类型中提取目标类型dst_type，将CastLike替换为仅接收x输入、携带dst_type属性的Cast算子，输出shape与x相同。融合前后图结构如下。

![](../../../docs/zh/figures/CastlikeFusionPass.png)

## 使用约束

- 结构约束：
  - 图中存在CastLike算子，输入为x、y，输出为output。
  - y为类型参考tensor，仅参与目标类型的推导，融合后不再使用。
- 数据类型约束：
  - x支持FLOAT16、FLOAT、DOUBLE、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX32、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。
  - y的数据类型即转换目标类型dst_type，需为Cast算子支持的输出类型。
- 数据格式和shape约束：
  - 输入数据格式为ND。
  - x的shape为0D\~8D，替换后Cast的输出shape与x相同。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR/Ascend 950DT
<!-- end id1 -->
