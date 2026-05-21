# Acosh

> 反双曲余弦（Inverse Hyperbolic Cosine）算子，对应 PyTorch `torch.acosh` / `torch.acosh_`。
> 在线 aclnn API 文档：[docs/aclnnAcosh&aclnnInplaceAcosh.md](docs/aclnnAcosh&aclnnInplaceAcosh.md)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：对输入Tensor中的每个元素进行反双曲余弦操作后输出。

- 计算公式：

  $$
  out=cosh^{-1}(self)
  $$

## 参数说明

<table style="table-layout: fixed; width: 1000px"><colgroup>
<col style="width: 50px">
<col style="width: 70px">
<col style="width: 200px">
<col style="width: 100px">
<col style="width: 50px">
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
    <td>self</td>
    <td>输入</td>
    <td>公式中的输入 self，dtype 与 out 一致，shape 任意（0~8 维），支持空 Tensor，支持非连续 Tensor。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>公式中的输出 out，shape 与 self 完全一致；非原地接口专有。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selfRef</td>
    <td>输入/输出</td>
    <td>原地接口专有，既作输入又作输出的张量，计算结果原地写回。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

- 参数 `self` 与 `out` 的 dtype 必须完全一致；shape 维度数 ∈ [0, 8]。
- 仅支持 ND 数据格式；不支持 NCHW / NHWC / 5HD 等私有格式。

## 约束说明


## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用（非原地）</td>
    <td><a href="./examples/test_aclnn_acosh.cpp">examples/test_aclnn_acosh.cpp</a></td>
    <td rowspan="2">参考 ops-math 仓库根 build.sh 完成算子编译与算子包安装，再用 examples 验证（NPU 真机）。</td>
  </tr>
  <tr>
    <td>aclnn 调用（原地）</td>
    <td><a href="./examples/test_aclnn_inplace_acosh.cpp">examples/test_aclnn_inplace_acosh.cpp</a></td>
  </tr>
</tbody></table>

