# PadV3GradL0

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 算子功能：
  
  根据不同条件调用不同的底层padgrad算子（pad_v3_grad_replicate/pad_v3_grad_replication/pad_v4_grad/reflection_pad3d_grad）。

## 调用说明
  
  | 算子名称 | 对应README                                                        |调用条件     |
  |--------------|------------------------------------------------------------------------|-------------------------|
  | circular_pad_grad | [circular_pad_grad](../circular_pad_grad/README.md) |当输入模式（mode）“circular”时调用算子circular_pad_grad。|
  | pad_v4_grad | [pad_v4_grad](../pad_v4_grad/README.md) |当输入张量维度小于等于4，输入模式（mode）为“reflect”时调用算子pad_v4_grad。|
  | pad_v3_grad_replicate | [pad_v3_grad_replicate](../pad_v3_grad_replicate/README.md) |当输入张量维度小于等于4，输入模式（mode）为“edge”时调用算子pad_v3_grad_replicate。|
  | pad_v3_grad_replication | [pad_v3_grad_replication](../pad_v3_grad_replication/README.md) |当输入张量维度等于5，输入模式（mode）为“edge”时调用算子pad_v3_grad_replication。|
  | reflection_pad3d_grad | [reflection_pad3d_grad](../reflection_pad3d_grad/README.md) |当输入张量维度等于5，输入模式（mode）为“reflect”时调用算子reflection_pad_3d_grad。|
