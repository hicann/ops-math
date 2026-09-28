# ParallelConcat

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×    |
| <term>Atlas推理系列产品</term>                             |    ×    |
| <term>Atlas训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：沿第一维拼接N个首维大小为1的同形张量。动态输入values中每个张量shape为[1, d_1..d_k]，输出output_data shape为[N, d_1..d_k]，第i行内容为第i个输入的展平内容。数据为逐比特搬运，NaN/Inf/±0等任意位模式保持不变。
- 计算公式：

  $$
  \text{output\_data}[i, d_1, \ldots, d_k] = \text{values}_i[0, d_1, \ldots, d_k], \quad i = 0, 1, \ldots, N-1
  $$

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|-----|-----------|----|---------|------|
| values | 输入 | 动态输入，待拼接的张量列表，共N个张量，每个张量shape为[1, d_1..d_k]，即公式中`values_i`。首维大小必须为1，所有张量shape与数据类型必须相同。 | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL | ND |
| output_data | 输出 | 拼接结果张量，shape为[N, d_1..d_k]，即公式中`output_data`。数据类型与values相同。 | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64、BOOL | ND |
| shape | 属性 | 输出张量的显式shape，即[N, d_1..d_k]。必选属性。 | LIST_INT | - |
| N | 属性 | 动态输入values的个数，N≥1且N==shape[0]。必选属性。 | INT | - |

## 约束说明

- 每个输入张量首维大小必须为1，rank范围[1,8]（rank为0的标量输入非法）。
- 所有输入张量的shape与数据类型必须相同，且attr shape[1:]与输入shape[1:]一致。
- 输出output_data的shape必须等于attr shape。
- 支持空张量场景（任一d_m为0时输出shape为[N, 0]）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| GE图模式 | [test_geir_parallel_concat](examples/arch35/test_geir_parallel_concat.cpp) | 通过[算子IR](op_graph/parallel_concat_proto.h)构图方式调用ParallelConcat算子。 |
