# ReduceStdWithMean

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | √ |
| <term>Atlas推理系列产品</term> | √ |
| <term>Atlas训练系列产品</term> | √ |

> Ascend 950PR/950DT 由本自定义算子包提供 kernel 实现（arch35）；其余产品由 CANN 内置算子库支持。Ascend 950PR/950DT、Atlas A3、Atlas A2 支持 float16 / float32 / bfloat16；其余产品支持 float16 / float32。

## 功能说明

给定外部预计算的均值 `mean`，沿指定维度 `dim` 计算输入 `x` 的标准差（`invert=false`）或其倒数（`invert=true`）。是 BatchNorm/LayerNorm/GroupNorm 归一化层统计量计算的关键子步骤，以及 PyTorch `torch.std`/`torch.std_mean`/`torch.batch_norm_stats` 在 NPU 上的底层实现。

标准差模式（`invert=false`，`correction` 生效）：

$$
y = \sqrt{\frac{\sum_{j=0}^{N-1}(x_j - mean_j)^2}{\max(0,\; N - correction)}}
$$

标准差倒数模式（`invert=true`，固定除以 N、加 epsilon，忽略 correction）：

$$
y = \frac{1}{\sqrt{\dfrac{1}{N}\sum_{j=0}^{N-1}(x_j - mean_j)^2 + \epsilon}}
$$

其中 $N=\prod_{d \in dim}\text{shape}[d]$ 为归约维度元素总数，$correction$ 为 Bessel 校正因子（0=总体、1=样本），$\epsilon$ 为数值稳定小量。

ReduceStdWithMean 为 CANN 内部 L0 算子，被 `aclnnStd`、`aclnnStdMeanCorrection`、`aclnnBatchNormStats` 三个上层 aclnn 接口组合调用，不直接暴露独立 aclnn 接口。

| 上层接口 | 功能 | correction | invert | eps |
|:---|:---|:---|:---|:---|
| aclnnStd | 独立标准差 | 用户传入 | false | — |
| aclnnStdMeanCorrection | 同时输出 (std, mean) | 用户传入 | false | 0.001（固定） |
| aclnnBatchNormStats | 输出 (mean, invstd) | 0（固定） | true | 用户传入 |

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出/属性</th>
<th>描述</th>
<th>数据类型</th>
<th>数据格式</th>
</tr>
</thead>
<tbody>
<tr>
<td>x</td>
<td>输入</td>
<td>待计算标准差的张量。</td>
<td>FLOAT16、FLOAT、BFLOAT16</td>
<td>ND</td>
</tr>
<tr>
<td>mean</td>
<td>输入</td>
<td>外部预计算均值（已广播到 x 形状），dtype 与 x 一致。</td>
<td>与 x 一致</td>
<td>ND</td>
</tr>
<tr>
<td>y</td>
<td>输出</td>
<td>输出张量，dtype 与 x 一致；shape 由 dim + keepdim 推导。</td>
<td>与 x 一致</td>
<td>ND</td>
</tr>
<tr>
<td>dim</td>
<td>可选属性</td>
<td>归约维度，范围 [-rank(x), rank(x)-1]，空表示全部维度。默认 {}。</td>
<td>ListInt</td>
<td>-</td>
</tr>
<tr>
<td>unbiased</td>
<td>可选属性</td>
<td>Bessel 校正（legacy 字段）。默认 true。</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>keepdim</td>
<td>可选属性</td>
<td>是否保留归约维为 size 1。默认 false。</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>invert</td>
<td>可选属性</td>
<td>true 返回 1/sqrt(var+eps)，false 返回 sqrt(var)。默认 false。</td>
<td>Bool</td>
<td>-</td>
</tr>
<tr>
<td>epsilon</td>
<td>可选属性</td>
<td>数值稳定项（仅 invert=true 使用）。默认 0.001。</td>
<td>Float</td>
<td>-</td>
</tr>
<tr>
<td>correction</td>
<td>可选属性</td>
<td>校正因子：0=总体，1=样本。默认 1。</td>
<td>Int</td>
<td>-</td>
</tr>
</tbody>
</table>

## 约束说明

- x 与 mean 数据类型须一致，仅支持 float16 / float32 / bfloat16；输出 dtype 与 x 一致。
- x 的 shape 维度不超过 8 维。
- mean 的 shape 须与 x 完全一致（mean 须已广播到 x 形状）。
- dim 取值范围 [-rank(x), rank(x)-1]，空表示全部维度归约，且不能包含重复轴（负数轴归一化后判重）。
- 输出 y 的 shape 须与 dim + keepdim 推导结果一致。
- correction 仅支持非负整数；invert=true 时忽略 correction，固定除以 N。
- 空张量（N=0）输出 NaN；N<=correction（invert=false）输出 +Inf 或 NaN。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
|---|---|---|
| GE图模式 | [test_geir_reduce_std_with_mean.cpp](examples/test_geir_reduce_std_with_mean.cpp) | 通过 GE IR 构图调用，算子 IR 定义见 [op_host/reduce_std_with_mean_def.cpp](op_host/reduce_std_with_mean_def.cpp) |
| aclnn API | [test_aclnn_batch_norm_stats.cpp](examples/test_aclnn_batch_norm_stats.cpp) | 通过 [aclnnBatchNormStats](docs/aclnnBatchNormStats.md) 接口方式调用ReduceStdWithMean算子。 |
| aclnn API | [test_aclnn_std_mean_correction.cpp](examples/test_aclnn_std_mean_correction.cpp) | 通过 [aclnnStdMeanCorrection](docs/aclnnStdMeanCorrection.md) 接口方式调用ReduceStdWithMean算子。 |
