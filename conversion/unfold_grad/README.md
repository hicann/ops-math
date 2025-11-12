# aclnnUnfoldGrad
## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：实现Unfold算子的反向功能，计算相应的梯度。

  Unfold算子根据入参self，计算出维度$dim$的所有大小为$size$的切片。两个切片之间的步长由$step$给出。如果$sizedim$是入参self的维度$dim$的大小，则返回的张量中维度$dim$的大小将为$(sizedim-size)/step+1$。返回的张量中附加了一个大小为$size$的附加维度。

  UnfoldGrad算子入参gradOut的shape为Unfold正向输出的shape，入参inputSizes为Unfold正向输入self的shape，UnfoldGrad算子出参gradIn的shape为Unfold正向入参self的shape。

  例子：

  ```
  >>> x = torch.arange(1., 8)
  >>> x
  tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.])
  >>> x.unfold(0, 2, 1)
  tensor([[ 1.,  2.],
          [ 2.,  3.],
          [ 3.,  4.],
          [ 4.,  5.],
          [ 5.,  6.],
          [ 6.,  7.]])
  >>> grad = x.unfold(0, 2, 2)
  tensor([[ 1.,  2.],
          [ 3.,  4.],
          [ 5.,  6.]])
  >>> res = torch.ops.aten.unfold_backward(grad, [7], 0, 2, 2)
  tensor([1, 2, 3, 4, 5, 6, 0])
  ```


## 参数说明

| 参数名     | 输入/输出/属性 | 描述                                                         | 数据类型                 | 数据格式 |
| ---------- | -------------- | ------------------------------------------------------------ | ------------------------ | -------- |
| gradOutput | 输入张量       | 表示梯度更新系数，shape为(..., (sizedim-size)/step+1, size)，要求满足gradOut的第dim维等于$(inputSizes[dim]-size)/step+1$和gradOut的size等于inputSizes的size+1。 | FLOAT、FLOAT16、BFLOAT16 | ND       |
| inputSizes | 输入数组       | 表示输出张量的形状，值为(..., sizedim)，inputSizes的size小于等于8。 | INT64                    | ND       |
| dim        | 输入           | 公式中的$dim$。表示展开发生的维度。$dim$需要满足dim大于等于0且dim小于inputSizes的size。 | INT64                    | -        |
| size       | 输入           | 公式中的$size$。表示展开的每个切片的大小。$size$需要满足size大于0且size小于等于inputSizes的第dim维。 | INT64                    | -        |
| step       | 输入           | 公式中的$step$。表示每个切片之间的步长。$step$需要满足step大于0。 | INT64                    |          |
| gradIn     | 输出           | 表示Unfold的对应梯度，数据类型必须和gradOut一致。            | FLOAT、FLOAT16、BFLOAT16 |          |

## 约束说明

gradOut的shape满足约束:
- gradOut的第dim维等于(inputSizes[dim]-size)/step+1。
- gradOut的size等于inputSizes的size+1。

dim大于等于0且dim小于inputSizes的size。size大于0且size小于等于inputSizes的第dim维。step大于0。

## 调用说明

| 调用方式  | 样例代码                                                     | 说明                                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| aclnn接口 | [test_unfold_grad](tests/ut/op_kernel/test_unfold_grad.cpp) | 通过[aclnnUnfoldGrad](docs/aclnnUnfoldGrad.md)接口方式调用UnfoldGrad算子。 |

