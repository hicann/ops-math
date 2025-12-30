# Reciprocal
    
##  产品支持情况
    
| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |
    
## 功能说明

- 算子功能：返回一个具有每个输入元素倒数的新张量。

- 计算公式：
    
    
$$
out = {\frac{1} {input}}
$$

- 示例：

input = tensor([1,2,3,4])，经过reciprocal计算后，out = tensor([1.00, 0.50, 0.33, 0.25])

## 参数说明

* self(aclTensor*,计算输入): Device侧的aclTensor，支持[非连续的Tensor](common/非连续的Tensor.md)，支持空Tensor传入，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、COMPLEX64、COMPLEX128、FLOAT32、FLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持DOUBLE、COMPLEX64、COMPLEX128、FLOAT32、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、BOOL。
* out(aclTensor \*，计算输出): Device侧的aclTensor，shape需要与self一致。支持[非连续的Tensor](common/非连续的Tensor.md)，支持空Tensor传入，[数据格式](common/数据格式.md)支持ND。
    - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持DOUBLE、COMPLEX64、COMPLEX128、FLOAT32、FLOAT16。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：数据类型支持DOUBLE、COMPLEX64、COMPLEX128、FLOAT32、FLOAT16、BFLOAT16。

## 约束说明

无
 	 
## 调用说明
 
| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_reciprocal](./examples/test_aclnn_reciprocal.cpp) | 通过[aclnnReciprocal](./docs/aclnnReciprocal&aclnnInplaceReciprocal.md)接口方式调用Reciprocal算子。  |
| aclnn调用 | [test_aclnn_inplace_reciprocal](./examples/test_aclnn_inplace_reciprocal.cpp) | 通过[aclnnInplaceReciprocal](./docs/aclnnReciprocal&aclnnInplaceReciprocal.md)接口方式调用Reciprocal算子。  |