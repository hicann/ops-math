# FloorMod

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                               |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明
- 算子功能: 将scalar self进行broadcast成和tensor other一样shape的tensor以后，其中的每个元素都转换为除以other的对应元素以后得到的余数。该结果与除数other同符号，并且该结果的绝对值是小于other的绝对值。
  实际计算remainder(self, other) 等效于以下公式：

  $$
  out_i = self - floor(self / other_i) * other_i
  $$

- 示例：

```
self = 5.0   # float
other = tensor([[-1, -2],
                [-3, -4]]).type(int32)
result = remainder(self, other)

# result的值
# tensor([[ 0., -1.],
#         [-1., -3.]])  float

# 对于元素other中的-4来说，计算结果为 5 - floor(5 / -4) * -4 = -3
# 可以看到，最终结果-3的绝对值小于原来的-4的绝对值。
```

- **参数说明：**

    * self(aclScalar*, 计算输入)：公式中的输入`self`，Host侧的aclScalar。
        - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与other的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](common/TensorScalar互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
        - <term>Atlas 训练系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE。数据类型与other的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。

    * other(aclTensor*, 计算输入)：公式中的输入`other`, Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND，数据维度不支持8维以上。
        - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与self的数据类型需满足数据类型推导规则（参见[TensorScalar互推导关系](common/TensorScalar互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。
        - <term>Atlas 训练系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE。数据类型与self的数据类型需满足数据类型推导规则（参见[互推导关系](common/互推导关系.md)），并且推导出的数据类型必须能转换为out的数据类型。

    * out(aclTensor \*, 计算输出)：公式中的输出`out`，Device侧的aclTensor。shape需要与other一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
        - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE、BFLOAT16。
        - <term>Atlas 训练系列产品</term>：数据类型支持INT32、INT64、FLOAT16、FLOAT、DOUBLE。

    * workspaceSize(uint64_t \*，出参)：返回需要在Device侧申请的workspace大小。

    * executor(aclOpExecutor \*\*，出参)：返回op执行器，包含了算子计算流程。

## 调用说明

| 调用方式 | 调用样例                                             | 说明                                                                                         |
|---------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| aclnn调用 | [test_aclnn_remainder_scalar_tensor](./examples/test_aclnn_remainder_scalar_tensor.cpp) | 通过[aclnnRemainderScalarTensor](./docs/aclnnRemainderScalarTensor.md)接口方式调用FloorDiv算子  |