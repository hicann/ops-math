# Im2col

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：图像到列，滑动局部窗口数据转为列向量，拼接为大张量。从批处理输入张量中提取滑动窗口。考虑一个形状为（N, C, H, W）或 (C, H, W) 的批处理输入张量，其中N是批处理维度， C是通道维度， 而 H, W 表示图像大小，此操作将输入的空间维度内的每个滑动kernel_size大小的块展平为（N, C $\times \prod$（kernel_size）, L）的3-D 或 （C $\times \prod$（kernel_szie）, L）的2-D 的 output张量的列（即最后一维），而L是这些块的总数。
- 计算公式：
  $L = \prod_{d} \lfloor \frac{spatial_size[d] + 2 \times padding[d] - dilation[d] \times （kernel_size[d] -1） -1}{stride[d]} + 1$ \rfloor, 其中spatial_size由上述输入张量的H,W构成。

## 参数说明

<table style="undefined;table-layout: fixed; width: 966px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 290px">
<col style="width: 264px">
<col style="width: 102px">
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
    <td>输入张量</td>
    <td>输入张量，shape为3维或4维。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>kernelSize</td>
    <td>输入数组</td>
    <td>卷积核的大小，size为2。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>dilation</td>
    <td>输入数组</td>
    <td>膨胀参数，size为2。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>padding</td>
    <td>输入数组</td>
    <td>填充大小，size为2，padding[0]表示H方向，padding[1]表示W方向。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>stride</td>
    <td>输入数组</td>
    <td>步长，size为2。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>输出张量，shape根据参数推导得出。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入张量的维度必须是3维或4维。
- kernelSize、dilation、padding、stride的size必须为2。
- kernelSize、dilation、stride的值必须大于0。
- padding的值不能小于0。

## 调用说明

| 调用方式  | 样例代码                                              | 说明                                                         |
| --------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_im2col](./examples/test_aclnn_im2col.cpp) | 通过[aclnnIm2col](docs/aclnnIm2col.md)接口方式调用Im2col算子。 |
