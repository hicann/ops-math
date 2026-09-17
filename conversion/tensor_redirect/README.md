# TensorRedirect

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：将输入张量`x`的值传递到输出张量`output_x`，输出与输入的数据类型和shape一致。本算子用于图优化或引用重定向场景，不做类型转换、广播或数据布局变换。

- 计算公式：

$$
(\mathrm{output\_x})_i=x_i
$$

其中 $i$ 遍历 $x$ 的全部元素。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>需要传递的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_x</td>
      <td>输出</td>
      <td>传递结果张量。数据类型、shape和数据格式均与x一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
      <td>ND</td>
    </tr>
  </tbody></table>

### 产品差异说明

<table><thead>
  <tr>
    <th>产品</th>
    <th>数据类型</th>
    <th>静态shape能力</th>
    <th>动态shape能力</th>
    <th>shape/rank及空Tensor限制</th>
  </tr></thead>
<tbody>
  <tr>
    <td><term>Ascend 950PR/Ascend 950DT</term></td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
    <td>输入ND→输出ND</td>
    <td>输入ND→输出ND；支持固定rank动态shape和dynamic rank</td>
    <td>rank取值范围为[1, 8]；支持空Tensor</td>
  </tr>
  <tr>
    <td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term><br>
      <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term><br>
      <term>Atlas 200I/500 A2 推理产品</term><br>
      <term>Atlas 推理系列产品</term><br>
      <term>Atlas 训练系列产品</term></td>
    <td>FLOAT、FLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64</td>
    <td>输入ND→输出ND</td>
    <td>不支持动态shape</td>
    <td>rank取值范围为[1, 8]；不支持空Tensor</td>
  </tr>
</tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | [test_geir_tensor_redirect](./examples/test_geir_tensor_redirect.cpp) | 通过[算子IR](./op_graph/tensor_redirect_proto.h)构图方式调用TensorRedirect算子。 |

本算子不提供同名aclnn接口，仅支持上表的GE图模式调用。
