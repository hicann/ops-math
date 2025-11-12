# MatmulV2CompressDequant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

- **算子功能**：进行矩阵乘计算时，可先通过msModelSlim工具对右矩阵进行无损压缩，减少内存占用，然后通过本接口完成无损解压缩、矩阵乘和反量化计算。
- **计算公式**：
  
  ```
  x2_unzip = unzip(x2, compressIndex, compressInfo)
  result = (x1 @ x2_unzip + bias) * deqScale
  ```
  其中x2表示右矩阵经过msModelSlim工具压缩后的一维数据，x2_unzip是接口内部进行无损解压缩后的数据（与原始右矩阵数据一致）。

## 参数说明约束说明

<table style="undefined;table-layout: fixed; width: 869px"><colgroup>
<col style="width: 144px">
<col style="width: 166px">
<col style="width: 343px">
<col style="width: 114px">
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
    <td>x1</td>
    <td>输入张量</td>
    <td>矩阵乘的左输入，2维张量。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入张量</td>
    <td>压缩后的矩阵乘右输入，1维张量。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>compressIndex</td>
    <td>输入张量</td>
    <td>矩阵乘右输入的压缩索引表，1维张量。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>bias</td>
    <td>输入张量</td>
    <td>偏置项，支持空指针传入。</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>deqScale</td>
    <td>输入张量</td>
    <td>反量化参数，数据类型为UINT64。</td>
    <td>UINT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>offsetW</td>
    <td>输入张量</td>
    <td>矩阵乘右输入的偏移量，当前仅支持空指针传入。</td>
    <td>INT8</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>offsetX</td>
    <td>输入属性</td>
    <td>矩阵乘左输入的偏移量，当前仅支持0。</td>
    <td>INT32</td>
    <td>-</td>
  </tr>
  <tr>
    <td>compressInfo</td>
    <td>输入数组</td>
    <td>压缩数据相关信息，包括压缩块信息和原始shape等。</td>
    <td>INT64</td>
    <td>-</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出张量</td>
    <td>计算结果输出。</td>
    <td>FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

- x1和x2_unzip的Reduce维度大小必须相等。
- 所有输入张量不支持非连续的Tensor。
- deqScale需要将原始float类型参数转换为UINT64数据格式。
- 当前offsetW仅支持空指针，offsetX仅支持0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_aclnn_matmul_compress_dequant](./examples/test_aclnn_matmul_compress_dequant.cpp) | 通过[aclnnMatmulCompressDequant](docs/aclnnMatmulCompressDequant.md)接口方式调用MatmulCompressDequant算子。 |
