# ConcatToConcatDFusionPass

## 融合模式

该融合将图中的Concat、ConcatV2、ConcatV2D算子替换为ConcatD算子，`concat_dim`的来源根据算子类型不同而不同：

- Concat算子：`concat_dim`从第0个输入（常量）读取。
- ConcatV2算子：`concat_dim`从最后一个输入（常量）读取。
- ConcatV2D算子：`concat_dim`从`concat_dim`属性读取。

融合前后图结构如下：

![](../../../docs/zh/figures/ConcatToConcatDFusionPass.png)

替换后ConcatD输出的dtype、shape、format与原算子输出保持一致。

## 使用约束

- 图中存在Concat/ConcatV2/ConcatV2D算子。
- Concat/ConcatV2算子的`concat_dim`输入必须为常量，且数据类型为int32或int64；ConcatV2D算子需携带`concat_dim`属性。

## 支持的型号

<!-- npu="950" id2 -->
Ascend 950PR&950DT系列产品
<!-- end id2 -->
