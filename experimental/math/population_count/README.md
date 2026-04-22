# PopulationCount

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：逐元素统计输入张量中每个 16 比特整数的二进制表示中 bit=1 的个数（population count，亦称 Hamming weight / popcount）。
- 计算公式：

  $$
  y_i = \text{popcount}_{16}(x_i)
  $$

  其中 $x$ 为输入整数张量，按无符号 16 位宽解释（INT16 负数按二进制补码逐比特计数），$y$ 为输出张量。每个元素的取值范围为 $[0, 16]$。

- 算法实现：基于 16-bit SWAR（SIMD Within A Register）的 4 步归约 + 高字节清零 + Cast 至 UINT8：
  1. $u = u - ((u \gg 1) \\& \text{0x5555})$
  2. $u = (u \\& \text{0x3333}) + ((u \gg 2) \\& \text{0x3333})$
  3. $u = (u + (u \gg 4)) \\& \text{0x0F0F}$
  4. $u = u + (u \gg 8)$
  5. $u = u \\& \text{0x00FF}$（将高字节清零，保证 Cast 不饱和）
  6. $y = \text{Cast}(u, \text{uint8}, \text{CAST\\_NONE})$

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
<col style="width: 200px">
<col style="width: 170px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td>x</td>
    <td>输入</td>
    <td>待统计的整数张量，对应公式中 x。INT16 负数按二进制补码（16 位无符号位模式）逐比特计数。</td>
    <td>INT16、UINT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>输出</td>
    <td>每个元素对应位计数结果，对应公式中 y。shape 与 x 一致，dtype 固定为 UINT8，取值范围 [0, 16]。</td>
    <td>UINT8</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- aclnnPopulationCount 默认确定性实现。
- 仅支持 <term>Ascend 950PR/Ascend 950DT</term>（arch35 / DAV_3510），其他芯片不支持。
- x 的 dtype 必须为 INT16 或 UINT16；y 的 dtype 必须为 UINT8。
- x、y 的 shape 必须完全相同，不支持广播（broadcast）。
- 支持 0-8 维 Tensor，0 维表示标量（scalar），此时 y 也必须为 0 维。
- 支持空 Tensor（元素个数为 0），直接返回空结果。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="examples/arch35/test_aclnn_population_count.cpp">test_aclnn_population_count</a></td>
    <td>参见 aclnnPopulationCount 接口文档了解接口定义与参数规格。调用前需完成自定义算子包的编译与安装（bash build.sh --pkg --experimental --soc=ascend950 --ops=population_count）。</td>
  </tr>
</tbody></table>

## 参考资源

- 接口：`aclnnPopulationCountGetWorkspaceSize` / `aclnnPopulationCount`
- Target：Ascend 950PR/Ascend 950DT（arch35 / DAV_3510）
