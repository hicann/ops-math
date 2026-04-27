# PadV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                       |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- **算子功能**：对输入 tensor 进行指定值的常量填充。
- **示例**：

  ```
  输入 tensor([[0,1,2]])
  paddings([[2,2]])
  constant_values(0)
  
  输出为([[0,0,0,1,2,0,0]])
  ```

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
      <td>x</td>
      <td>输入</td>
      <td>待进行填充的原始 tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>填充配置，shape=[N, 2]，其中 N 为 x 的维度数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>constant_values</td>
      <td>输入</td>
      <td>填充常量值，标量 tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>填充后的 tensor。</td>
      <td>FLOAT、FLOAT16、INT32、INT64、INT16、INT8、UINT8、UINT16、UINT32、UINT64、BOOL、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>、<term>Atlas 200I/500 A2 推理产品</term>：数据类型不支持BFLOAT16、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：数据类型不支持 HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT8_E8M0、FLOAT4_E2M1、FLOAT4_E1M2。

### 输出 shape 计算

```
y.shape[d] = x.shape[d] + paddings[d][0] + paddings[d][1]
```

## 约束说明

1. **维度约束**：
   - x 的维度数必须在 [1, 8] 范围内
   - paddings 的第一维必须等于 x 的维度数
   - paddings 的第二维必须等于 2

2. **数据类型约束**：
   - x、constant_values、y 必须使用相同的数据类型
   - paddings 必须使用 INT32 或 INT64

3. **填充约束**：
   - 负填充（slice）时，输出 shape = x.shape[d] + left + right >= 0

4. **paddings 参数约束**
   - paddings 的形状必须为 [rank, 2]，其中 rank 为输入 x 的维度数（1~8）
   - 每一行 [left, right] 表示对应维度的填充数量
     - left: 在该维度的开头填充的元素数
     - right: 在该维度的末尾填充的元素数
   - paddings的值可以为：
     - 正数：表示填充
     - 负数：表示 slice（裁剪）
     - 零：表示不填充

## 调用说明

| 调用方式　 | 调用样例　　　　　　　　　　　　　　　　　　　　　　　　| 说明　　　　　　　　　　　　　　　　　　　　　　　　　　　　　 |
| ------------| ---------------------------------------------------------| ----------------------------------------------------------------|
| aclnn 调用 | [test_geir_pad_v2.cpp](./examples/test_geir_pad_v2.cpp) | 通过 [算子IR](op_graph/pad_v2_proto.h) 接口方式调用 PadV2 算子 |
