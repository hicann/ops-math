# BitwiseAnd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                             |    √     |
| <term>Atlas A3系列产品</term>     |    √     |
| <term>Atlas A2系列产品</term> |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    √     |
| <term>Atlas训练系列产品</term>                              |    √     |

## 功能说明

- 算子功能：计算输入tensor中每个元素与另一个输入（tensor或标量）对应元素的按位与结果。输入必须是整型或者BOOL型变量，当输入为BOOL时，计算逻辑与的结果。
- 计算公式：

  - 当other为张量时：

    $$
    out_i = self_i \& other_i
    $$

  - 当other为标量时：

    $$
    out_i = self_i \& other
    $$

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
      <td>self</td>
      <td>输入</td>
      <td>公式中的输入self。</td>
      <td>INT16、UINT16、INT32、INT64、INT8、UINT8、UINT32、UINT64、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>other</td>
      <td>输入</td>
      <td>公式中的输入other。</td>
      <td>INT16、UINT16、INT32、INT64、INT8、UINT8、UINT32、UINT64、BOOL</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>公式中的输出out。</td>
      <td>BOOL、INT8、INT16、UINT16、INT32、UINT32、INT64、UINT64、UINT8、FLOAT、FLOAT16、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- Atlas A3系列产品、Atlas A2系列产品、Atlas推理系列产品、Atlas训练系列产品：self、other的数据类型不支持UINT32、UINT64；out的数据类型不支持UINT32、UINT64、COMPLEX64、COMPLEX128。

## 约束说明

- 输入的数据类型需满足互推导规则（参见[互推导关系](../../docs/zh/context/deduction_relationship.md)），推导后的数据类型需在支持范围内。
- 各接口默认确定性实现，具体约束参见对应 aclnn 接口文档。

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                         |
| ---------------- | --------------------------- |----------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_bitwise_and_scalar.cpp](examples/test_aclnn_bitwise_and_scalar.cpp) | 通过[aclnnBitwiseAndScalar](docs/aclnnBitwiseAndScalar.md)接口方式调用BitwiseAndScalar算子。 |
| aclnn接口  | [test_aclnn_bitwise_and_tensor.cpp](examples/test_aclnn_bitwise_and_tensor.cpp) | 通过[aclnnBitwiseAndTensor](docs/aclnnBitwiseAndTensor.md)接口方式调用BitwiseAndTensor算子。 |
| aclnn接口  | [test_aclnn_bitwise_and_tensor_out.cpp](examples/test_aclnn_bitwise_and_tensor_out.cpp) | 通过[aclnnBitwiseAndTensorOut](docs/aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut.md)接口方式调用BitwiseAndTensorOut算子。 |
| aclnn接口  | [test_aclnn_inplace_bitwise_and_tensor_out.cpp](examples/test_aclnn_inplace_bitwise_and_tensor_out.cpp) | 通过[aclnnInplaceBitwiseAndTensorOut](docs/aclnnBitwiseAndTensorOut&aclnnInplaceBitwiseAndTensorOut.md)接口方式调用InplaceBitwiseAndTensorOut算子。 |
| aclnn接口  | [test_aclnn_inplace_bitwise_and_scalar.cpp](examples/test_aclnn_inplace_bitwise_and_scalar.cpp) | 通过[aclnnInplaceBitwiseAndScalar](docs/aclnnInplaceBitwiseAndScalar.md)接口方式调用InplaceBitwiseAndScalar算子。 |
| aclnn接口  | [test_aclnn_inplace_bitwise_and_tensor.cpp](examples/test_aclnn_inplace_bitwise_and_tensor.cpp) | 通过[aclnnInplaceBitwiseAndTensor](docs/aclnnInplaceBitwiseAndTensor.md)接口方式调用InplaceBitwiseAndTensor算子。 |
