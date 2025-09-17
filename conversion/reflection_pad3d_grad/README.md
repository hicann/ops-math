# ReflectionPad3dGrad
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：计算aclnnReflectionPad3d api的反向传播。

## 参数说明

| 参数名     | 输入/输出/属性 | 描述                                            | 数据类型                                         | 数据格式 |
| ---------- | -------------- | ----------------------------------------------- | ------------------------------------------------ | -------- |
| gradOutput | 输入           | 反向传播的输入。                                | FLOAT16、FLOAT32、DOUBLE、 COMPLEX64、COMPLEX128 | ND       |
| self       | 输入           | 正向的输入张量。                                | FLOAT16、FLOAT32、DOUBLE、 COMPLEX64、COMPLEX128 | ND       |
| padding    | 输入           | 长度为6，数值依次代表左右上下前后需要填充的值。 | aclIntArray数组                                  | -        |
| gradInput  | 输出           | 反向传播的输出。                                | FLOAT16、FLOAT32、DOUBLE、 COMPLEX64、COMPLEX128 | ND       |

## 约束说明

输入shape限制：gradOutput、self 和 gradInput 的维度需一致（支持四维或五维），且它们的形状需与 reflection_pad3d 正向传播的输出形状相互一致。

输入值域限制：padding前两个数值需小于self最后一维度的数值，中间两个数值需小于self倒数第二维度的数值，后两个数值需小于self倒数第三维度的数值。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_reflection_pad3d_grad](tests/ut/op_kernel/test_reflection_pad3d_grad.cpp) | 通过[aclnnReflectionPad3dBackward](docs/aclnnReflectionPad3dBackward.md)接口方式调用ReflectionPad3dBackward算子。 |