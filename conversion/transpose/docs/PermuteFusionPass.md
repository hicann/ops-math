# PermuteFusionPass

## 融合模式

该融合将图中的Permute算子替换为TransposeD或Transpose算子，具体替换方式取决于运行平台：

<!-- npu="A3,910b,910,310p,310b" id16 -->
**场景一**：

<!-- npu="A3,910b,910,310p,310b" id14 -->
将Permute替换为TransposeD算子，Permute的维度排列属性作为TransposeD的`perm`属性传入。
支持平台如下：
<!-- end id14 -->
<!-- npu="A3" id7 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id7 -->
<!-- npu="910b" id8 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id8 -->
<!-- npu="310b" id9 -->
Atlas 200I/500 A2 推理产品
<!-- end id9 -->
<!-- npu="310p" id10 -->
Atlas 推理系列产品
<!-- end id10 -->
<!-- npu="910" id11 -->
Atlas 训练系列产品
<!-- end id11 -->

融合前后图结构如下：

![](../../../docs/zh/figures/PermuteFusionPass_1.png)
<!-- end id16 -->

<!-- npu="950" id15 -->
**场景二**：

<!-- npu="950" id13 -->
将Permute替换为Transpose算子，Permute的维度排列属性以Const节点的形式作为Transpose的`perm`输入传入。
支持平台如下：
<!-- end id13 -->

<!-- npu="950" id12 -->
Ascend 950PR/Ascend 950DT
<!-- end id12 -->


融合前后图结构如下：

![](../../../docs/zh/figures/PermuteFusionPass_2.png)
<!-- end id15 -->

Permute算子的维度排列顺序优先从`perm`属性读取，若不存在则回退读取`order`属性。替换后输出的shape与dtype与原Permute输出一致。

## 使用约束

- 结构约束：
  - 图中存在Permute算子，输入为x，输出为y。
  - Permute算子需携带`perm`或`order`属性，用于指定维度排列顺序；属性取值需为x维度的合法全排列，不可重复、不可遗漏。
- 版本约束：GE编译器版本不低于 9.1.0 时融合生效。

## 支持的型号

<!-- npu="310b" id1 -->
Atlas 200I/500 A2 推理产品
<!-- end id1 -->

<!-- npu="310p" id2 -->
Atlas 推理系列产品
<!-- end id2 -->

<!-- npu="910" id3 -->
Atlas 训练系列产品
<!-- end id3 -->

<!-- npu="910b" id4 -->
Atlas A2 训练系列产品/Atlas A2 推理系列产品
<!-- end id4 -->

<!-- npu="A3" id5 -->
Atlas A3 训练系列产品/Atlas A3 推理系列产品
<!-- end id5 -->

<!-- npu="950" id6 -->
Ascend 950PR/Ascend 950DT
<!-- end id6 -->
