# RaggedBinCount

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：RaggedBinCount按 `splits` 划分的ragged row，统计每个bin的出现次数或权重和。设
  `B = numel(splits) - 1`、`M = size[0]`，第 `b` 行对应 `values[splits[b]:splits[b + 1]]`，输出shape为 `[B, M]`。

  对满足 `0 <= values[i] < M` 的元素：`binary_output=false` 且 `weights` 为空时输出对应位置累加 `1.0`；
  `binary_output=false` 且 `weights` 非空时累加 `weights[i]`；`binary_output=true` 时出现过的bin输出 `1.0`
  并忽略权重值。大于等于 `M` 的bin会被忽略，负bin属于非法输入。

  使用场景：对变长序列批次（ragged batch）逐行做直方图统计，如按样本统计词表命中次数或类别权重和。

- 计算公式：

  非二值（`binary_output = false`）：

$$output_{b,m} = \sum_{i=splits_b}^{splits_{b+1}-1} \mathbb{1}[values_i = m] \cdot w_i$$

  其中 $w_i = weights_i$（`weights` 非空）或 $w_i = 1.0$（`weights` 为空tensor `[0]`）。

  二值（`binary_output = true`），忽略 `weights`：

$$output_{b,m} = \mathbb{1}\left[\exists\, i \in [splits_b,\ splits_{b+1}) ,\ values_i = m\right]$$

  其中 $b \in [0, B)$、$m \in [0, M)$，$\mathbb{1}[\cdot]$ 为指示函数（成立取1，否则取0）。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| splits | 输入 | 严格1D row-splits tensor，元素数至少为2 | INT64 | ND |
| values | 输入 | 1D或2D bin index tensor，按连续元素顺序展平 | INT32、INT64 | ND |
| size | 输入 | 1D且元素个数为1（即shape为`[1]`），`size[0]` 为非负bin数量，dtype与values相同 | INT32、INT64 | ND |
| weights | 输入 | 权重tensor，必传。元素个数为0（空Tensor）时表示不使用权重，等价于全1；否则元素个数必须与values相同，按values展平后的顺序逐元素对应 | FLOAT（FP32） | ND |
| binary_output | 属性 | 是否输出二值结果，默认false | BOOL | - |
| output | 输出 | 严格2D，shape为`[numel(splits)-1, size[0]]` | FLOAT（FP32） | ND |

## 约束说明

- 本目录仅注册 `ascend950`，Kernel位于 `arch35`，不改变A2/A3既有实现。
- 各输入的数据类型必须整组匹配下表之一，不允许跨行组合（`size` 的dtype必须与 `values` 相同）：

| splits | values | size | weights | output | 数据格式 |
| --- | --- | --- | --- | --- | --- |
| INT64 | INT32 | INT32 | FLOAT（FP32） | FLOAT（FP32） | ND |
| INT64 | INT64 | INT64 | FLOAT（FP32） | FLOAT（FP32） | ND |

- `splits[0]` 必须为0，`splits` 必须单调不降，`splits[-1]` 必须等于values元素数。
- `weights` 只按**元素个数**校验，不校验shape与维数：元素个数为0即表示不使用权重，非0时必须与`values`的元素个数相同（维数可以不同，按展平顺序对应）。此规则与canndev一致。
- 空Tensor支持情况，按轴分别说明：
  - `size[0]` 为0：输出为空Tensor `[numel(splits)-1, 0]`，不下发有效计算，直接返回成功。
  - `values` 元素数为0：此时 `splits` 各元素必须全为0，输出各位置均为 `+0.0`。
  - `weights` 元素个数为0（如shape为`[0]`、`[0,3]`、`[2,0]`）：表示无权重，等价于全1，均合法。
  - `splits` **不支持**空Tensor，元素数必须至少为2，否则校验失败返回错误。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|--------|------|
| 图模式调用 | [test_geir_ragged_bin_count](./examples/test_geir_ragged_bin_count.cpp) | 通过[算子IR](./op_graph/ragged_bin_count_proto.h)构图方式调用RaggedBinCount算子。 |

该算子不提供同名aclnn接口（`CMakeLists.txt` 显式配置 `ACLNNTYPE aclnn_exclude`），仅支持上表的图模式调用。
