# MatrixSetDiag

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：将输入tensor的对角线元素替换为对角线tensor的值。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>input</td>
      <td>输入</td>
      <td>待进行替换的原始tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、UINT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、BOOL、COMPLEX64、DOUBLE</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diagonal</td>
      <td>输入</td>
      <td>对角线tensor。</td>
      <td>与input的数据类型保持一致。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>输入</td>
      <td>对角线的取值范围。</td>
      <td>INT32。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>进行替换后的tensor。</td>
      <td>与input的数据类型保持一致。</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- input的维度最少为2维，最大不超过8维。
- k是数据类型INT32的标量或模长为2的向量。
- 当k为标量且K[0]=K[1]时，diagonal的维度比input的维度小1维，且最后一维的值为input最后两维的较小值；当k不是标量且K[0]!=K[1]时，diagonal的维度等于input的维度。
- 当k为标量且K[0]=K[1]时，diagonal的维度除最后一维外其他维度要和input的维度一一对应相等；当k不是标量且K[0]!=K[1]时，diagonal的维度除最后两维外其他维度要和input的维度一一对应相等。

## 调用说明

| 调用方式 | 调用样例                                                                      | 说明                                                                    |
|--------------|---------------------------------------------------------------------------|-----------------------------------------------------------------------|
| 图模式调用 | [test_geir_matrix_set_diag_v2](examples/test_geir_matrix_set_diag_v2.cpp) | 通过[算子IR](op_graph/matrix_set_diag_v2_proto.h)构图方式调用MatrixSetDiagV2算子。 |
