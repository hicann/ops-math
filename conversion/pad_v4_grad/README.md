# PadV4Grad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：
  
  reflection_pad1d的反向传播。
  reflection_pad2d的反向传播。

## 参数说明

</style>
<table class="tg" style="undefined;table-layout: fixed; width: 922px"><colgroup>
<col style="width: 168.333333px">
<col style="width: 116.666666px">
<col style="width: 305.666666px">
<col style="width: 233.666666px">
<col style="width: 97.666666px">
</colgroup>
<thead>
  <tr>
    <th class="tg-0pky">参数名</th>
    <th class="tg-0pky">输入/输出/属性</th>
    <th class="tg-0lax">描述</th>
    <th class="tg-0lax">数据类型</th>
    <th class="tg-0lax">数据格式</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky">gradOutput</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0lax">reflection_pad1d/reflection_pad2d的正向传播，shape支持2~4维且维度需要与self和gradInput保持一致，2/3维为1d，3/4维为2d。</td>
    <td class="tg-0lax">FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">self</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0lax">shape支持2~4维且维度需要与self和gradInput保持一致，shape与gradInput一致。</td>
    <td class="tg-0lax">FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
    <td class="tg-0lax">ND</td>
  </tr>
  <tr>
    <td class="tg-0pky">padding</td>
    <td class="tg-0pky">输入</td>
    <td class="tg-0lax">长度为2或4，长度为2时表示1d左右需要填充的值，padding的前两个值都需要小于self最后一维度的数值；长度为4时表示2d左右上下需要填充的值，padding的前两个值都需要小于self最后一维度的数值，后两个值需要小于倒数第二维的数值。</td>
    <td class="tg-0lax">INT64</td>
    <td class="tg-0lax"></td>
  </tr>
  <tr>
    <td class="tg-0pky">gradInput</td>
    <td class="tg-0pky">输出</td>
    <td class="tg-0lax">数据类型与self保持一致</td>
    <td class="tg-0lax">FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128</td>
    <td class="tg-0lax">ND</td>
  </tr>
</tbody></table>

## 约束说明

  输入shape限制：gradOutput、self 和 gradInput 的维度需一致（支持三/四维），且它们的形状需与 reflection_pad1d/reflection_pad2d 正向传播的输出形状相互一致。

  输入值域限制：padding长度为2时，padding的前两个值都需要小于self最后一维度的数值；长度为4时，padding的前两个值都需要小于self最后一维度的数值，后两个值需要小于倒数第二维的数值。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn调用 | [test_aclnn_pad_v4_grad.cpp](exampels/test_aclnn_pad_v4_grad.cpp) | 通过[aclnnReflectionPad1dBackward](docs/aclnnReflectionPad1dBackward.md)接口方式调用PadV4Grad算子。    |
| aclnn调用 | [test_aclnn_reflection_pad2d_backward](examples/test_aclnn_reflection_pad2d_backward.cpp) | 通过[aclnnReflectionPad2dBackward](docs/aclnnReflectionPad2dBackward.md)接口方式调用PadV4Grad算子。(PadV4Grad算子aclnn接口aclnnReflectionPad2dBackward和PadV3GradReplicate算子共用一个aclnn接口，[aclnnReflectionPad2dBackward](../pad_v3_grad_replicate/op_host/op_api/aclnn_pad2d_backward.h)调用此示例需两个算子联合编译：bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] --ops=pad_v4_grad,pad_v3_grad_replicate)   |