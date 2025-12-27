# Triu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |     √      |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √       |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：将输入的self张量的最后二维（按shape从左向右数）沿对角线的左下部分置零。参数diagonal可正可负，默认为零，正数表示主对角线向右上方向移动，负数表示主对角线向左下方向移动。
- 计算公式：下面用i表示遍历倒数第二维元素的序号（i是行索引），用j表示遍历最后一维元素的序号（j是列索引），用d表示diagonal，在(i, j)对应的二维坐标图中，i+d==j表示在对角线上。

  $$
  对角线及其右上方，即i+d<=j，保留原值： out_{i, j} = self_{i, j}\\
  而位于对角线左下方的情况，即i+d>j，置零（不含对角线）：out_{i, j} = 0
  $$

- 示例：

  $self = \begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [3&4&1] \end{bmatrix}$，
  triu(self, diagonal=0)的结果为：
  $\begin{bmatrix} [9&6&3] \\ [0&2&3] \\ [0&0&1] \end{bmatrix}$；
  调整diagonal的值，triu(self, diagonal=1)结果为：
  $\begin{bmatrix} [0&6&3] \\ [0&0&3] \\ [0&0&0] \end{bmatrix}$；
  调整diagonal为-1，triu(self, diagonal=-1)结果为：
  $\begin{bmatrix} [9&6&3] \\ [1&2&3] \\ [0&4&1] \end{bmatrix}$。

  - **参数说明：**

    - self(aclTensor*, 计算输入)：公式中的$self$，Device侧的aclTensor，shape支持2-8维和空tensor。[数据格式](common/数据格式.md)支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
      - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16。
      - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64。
    - diagonal(int64_t, 计算输入)：对角线偏移量，数据类型int64_t。
    - out(aclTensor*, 计算输出)：公式中的$out$，Device侧的aclTensor，shape支持2-8维和0维，数据类型和shape需要与self保持一致。 [数据格式](common/数据格式.md)需要与self保持一致，支持ND、NCHW、NHWC、HWCN、NDHWC、NCDHW。
      - <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL。
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16。
      - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持UINT64、INT64、UINT32、 INT32、UINT16、INT16、UINT8、 INT8、FLOAT16、FLOAT32、DOUBLE、BOOL、BFLOAT16、COMPLEX32、COMPLEX64。
    - workspaceSize(uint64_t*, 出参)：返回需要在Device侧申请的workspace大小。
    - executor(aclOpExecutor**, 出参)：返回op执行器，包含了算子计算流程。

## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_triu](./examples/test_aclnn_triu.cpp) | 通过[aclnnTriu](./docs/aclnnTriu&aclnnInplaceTriu.md)接口方式调用Triu算子。 |

