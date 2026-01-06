# DivNoNan
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理产品</term> |    √     |


## 功能说明

- 算子功能：在处理除法时，特别是在分母可能为0的情况下，可以帮助避免由于分母为0导致的NaN值问题，从而确保模型的稳定性和准确性。
### DivNoNan接受两个参数:
- x: 第一个输入Tensor，作为分子。y: 第二个输入Tensor，作为分母，与x应该具有相同的数据类型。当y中的某个元素为0，则Tensor中对应未知的值将为0，而不是NaN。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
<col style="width: 140px">
<col style="width: 140px">
<col style="width: 180px">
<col style="width: 213px">
<col style="width: 100px">
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
    <td>输入</td>
    <td>输入Tensor。</td>
    <td>DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>x2</td>
    <td>输入</td>
    <td>输入Tensor。</td>
    <td>DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>y</td>
    <td>输出</td>
    <td>输出结果。</td>
    <td>DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT32, DT_UINT8, DT_INT8</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

无


## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式接口 | [test_geir_div_no_nan](examples/test_geir_div_no_nan.cpp) | 通过[算子IR](op_graph/div_no_nan_proto.h)接口方式调用DivNoNan算子。 |