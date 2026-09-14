# KLDiv

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：计算KL散度损失（Kullback-Leibler Divergence）。

- 壳算子定位：KLDiv为壳路由算子，注册名KLDiv与旧GE图的op type精确匹配，用于旧图迁移兼容。如有新计算需求，请使用KLDivV2代替；本算子计算内核纯转发KLDivV2的log_target=False DAG，恒按log_target=False语义计算。

- 计算公式（恒log_target=False）：

$$
loss_i=\begin{cases}
0 & \text{if } target_i=0 \\
NaN & \text{if } target_i<0 \\
target_i \times \left(\log(target_i)-x_i\right) & \text{else}
\end{cases}
$$

- 输出y由reduction决定：

$$
y=\begin{cases}
loss & \text{if } reduction=\text{"none"} \\
\sum_i loss_i & \text{if } reduction=\text{"sum"} \\
\frac{\sum_i loss_i}{numel} & \text{if } reduction=\text{"mean"} \\
\frac{\sum_i loss_i}{dim_0} & \text{if } reduction=\text{"batchmean"}
\end{cases}
$$

- reduction="none"时y与x同shape；reduction为"mean"、"sum"、"batchmean"时y为标量（shape为空）。

## 参数说明

<table style="table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 90px">
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
      <td>公式中的输入张量x（log概率）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>target</td>
      <td>输入</td>
      <td>公式中的输入张量target（概率），与x同shape同dtype。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>reduction</td>
      <td>必选属性</td>
      <td><ul><li>指定输出的归约方式。</li><li>取值仅支持none、mean、sum、batchmean。</li><li>无默认值，构造图时必须显式设置。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量y。reduction="none"时与x同shape，其余取值时为标量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- 本算子原型未声明log_target属性（v1原型即无此属性），恒按log_target=False语义计算；旧图中携带的log_target属性会被作为未知属性忽略。

## 约束说明

- x与target必须同shape、同dtype。
- reduction为必选属性，取值仅支持none、mean、sum、batchmean。
- 原型纸面声明支持DT_DOUBLE，但tiling阶段dtype白名单拒绝double，实际不支持DT_DOUBLE。
- target等于0的位置对应输出为0；target小于0的位置输出为NaN（值域约束，tiling/infershape不校验，调用方需自行保证输入语义合法）。
- FLOAT16、BFLOAT16输入在kernel内部提升为FLOAT32计算，结果按round-to-nearest-even回落原dtype。
- 输出shape：reduction="none"时y与x同shape；reduction为"mean"、"sum"、"batchmean"时y为标量。
- 本算子为壳路由算子，无aclnn接口：旧图迁移走atc图编译路径；PyTorch适配请使用ops-math kl_div_v2目录的aclnnKlDiv接口。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_kl_div.cpp">test_geir_kl_div</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>

- 图模式示例通过<a href="./op_graph/kl_div_proto.h">算子IR</a>以op::KLDiv构图方式调用本算子。
