# Variable

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

创建一个变量张量。Variable算子通过输入x赋值，在内部记录变量张量的值，输出y为创建的变量张量。InferShape函数返回成功但不显式设置输出描述，由框架默认推导（output跟随input）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 996px"><colgroup>
  <col style="width: 102px">
  <col style="width: 168px">
  <col style="width: 203px">
  <col style="width: 403px">
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
      <td>用于给变量张量赋值的输入张量，调用者无需传递变量张量的值。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>index</td>
      <td>属性</td>
      <td>输入张量的索引。</td>
      <td>Int，默认值为0</td>
      <td>-</td>
    </tr>
    <tr>
      <td>value</td>
      <td>属性</td>
      <td>用于传递和记录变量张量的值。</td>
      <td>Tensor，默认值为Tensor()</td>
      <td>-</td>
    </tr>
    <tr>
      <td>container</td>
      <td>属性</td>
      <td>变量张量的容器。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shared_name</td>
      <td>属性</td>
      <td>变量张量的共享名称。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>创建的变量张量。</td>
      <td>TensorType::ALL()</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

无

## 调用说明

Variable算子为图引擎基础算子，通常作为图中变量节点使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_variable](./examples/test_geir_variable.cpp) | 通过[算子IR](./op_graph/variable_proto.h)构图方式调用Variable算子 |
