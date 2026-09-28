# VarHandleOp

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

创建一个指向变量资源的句柄。输出张量y为标量资源句柄（DT_RESOURCE类型），shape和dtype属性描述句柄所指向变量的形状和数据类型（而非输出本身的形状），供下游资源类算子（如ReadVariableOp、AssignVariableOp）使用。container和shared_name属性用于指定变量的容器和共享名称。

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
      <td>container</td>
      <td>属性</td>
      <td>可选。变量所在的容器。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shared_name</td>
      <td>属性</td>
      <td>可选。变量的共享名称。</td>
      <td>String，默认值为""</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dtype</td>
      <td>属性</td>
      <td>必选。句柄所指向变量的元素数据类型。</td>
      <td>Type</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shape</td>
      <td>属性</td>
      <td>可选。句柄所指向变量的形状。</td>
      <td>ListInt，默认值为UNKNOWN_SHAPE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>标量资源句柄（DT_RESOURCE类型）。</td>
      <td>TensorType({DT_RESOURCE})</td>
      <td>ND</td>
    </tr>

  </tbody></table>

## 约束说明

- 本算子为实验性功能（EXPERIMENTAL），请勿在生产环境使用。
- 本算子为资源句柄源节点，输出为标量 DT_RESOURCE 张量，无运行时计算语义，无法通过 RunGraph/BuildModel 独立执行。
- shape/dtype 属性描述句柄所指向变量的形状与类型（并非输出本身的形状与类型），推导时经 InferenceContext 的 HandleShapesAndTypes 机制传递给下游资源类算子（框架 peer 传播），下游经 GetInputHandleShapesAndTypes 读取。

## 调用说明

VarHandleOp算子为图引擎基础算子，作为资源句柄源节点配合下游资源类算子（如ReadVariableOp、AssignVariableOp）使用，不单独调用。

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| 图模式调用 | [test_geir_var_handle_op](./examples/test_geir_var_handle_op.cpp) | 通过[算子IR](./op_graph/var_handle_op_proto.h)构图方式调用VarHandleOp算子 |
