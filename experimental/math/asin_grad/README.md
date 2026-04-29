# AsinGrad 自定义算子

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR / Ascend 950DT | Yes |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | No |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | No |
| Atlas 200I/500 A2 推理产品 | No |
| Atlas 推理系列产品 | No |
| Atlas 训练系列产品 | No |

## 功能说明

AsinGrad 是 Asin（反正弦）算子的梯度算子，用于深度学习框架自动微分的反向传播阶段。给定正向 Asin 算子的输入 x 和上游梯度 dy，计算输入梯度 dx。

**计算公式**：

$$dx_i = \frac{dy_i}{\sqrt{1 - x_i^2}}$$

其中：

- x：正向 Asin 算子的输入 tensor，值域 [-1, 1]
- dy：上游传播的梯度（grad_output）
- dx：计算得到的输入梯度（grad_input）

## 参数说明

### aclnnAsinGradGetWorkspaceSize

```cpp
aclnnStatus aclnnAsinGradGetWorkspaceSize(
    const aclTensor *dy,
    const aclTensor *x,
    aclTensor       *dx,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor);
```

| 参数名 | 输入/输出 | 描述 |
| ------ | :-------: | ---- |
| dy | 输入 | 上游传播的梯度 tensor。数据类型支持 FLOAT16、FLOAT32、BFLOAT16。shape 维度 0-8 维。支持空 Tensor。 |
| x | 输入 | 正向 Asin 算子的输入 tensor。数据类型与 dy 一致，shape 与 dy 一致。支持空 Tensor。 |
| dx | 输出 | 计算得到的输入梯度 tensor。数据类型与 dy 一致，shape 与 dy 一致。 |
| workspaceSize | 输出 | 返回需要在 Device 侧申请的 workspace 大小。 |
| executor | 输出 | 返回 op 执行器，包含了算子计算流程。 |

### aclnnAsinGrad

```cpp
aclnnStatus aclnnAsinGrad(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream);
```

| 参数名 | 输入/输出 | 描述 |
| ------ | :-------: | ---- |
| workspace | 输入 | 在 Device 侧申请的 workspace 内存地址。 |
| workspaceSize | 输入 | 在 Device 侧申请的 workspace 大小，由第一段接口获取。 |
| executor | 输入 | op 执行器，包含了算子计算流程。 |
| stream | 输入 | 指定执行任务的 Stream。 |

## 约束说明

- **类型约束**：dy 和 x 的数据类型必须一致，dx 的数据类型与 dy/x 一致。支持 FLOAT16、FLOAT32、BFLOAT16。
- **shape 约束**：dy 和 x 的 shape 必须完全一致（不支持广播），dx 的 shape 与 dy/x 一致。支持 0-8 维 tensor，支持空 tensor（0 元素）。
- **值域约束**：x 的值域应在 [-1, 1] 范围内。当 |x| = 1 时，计算结果为 inf；当 |x| > 1 时，计算结果为 NaN。这与 PyTorch 行为一致。
- **确定性说明**：aclnnAsinGrad 为确定性实现。

## 调用说明

### 调用方式

| 调用方式 | 是否支持 |
| -------- | :------: |
| ACLNN 调用 | Yes |
| torch_npu 单算子 | No |
| torch.compile 入图 | No |
| GE 图模式-静态 shape | No |
| GE 图模式-动态 shape | No |

### ACLNN 调用流程

1. 调用 `aclnnAsinGradGetWorkspaceSize` 获取 workspace 大小和 executor
2. 根据返回的 workspaceSize 在 Device 侧申请 workspace 内存
3. 调用 `aclnnAsinGrad` 执行计算
4. 同步 Stream 等待执行完成

### 调用示例

请参考 [examples/test_aclnn_asin_grad.cpp](examples/test_aclnn_asin_grad.cpp) 获取完整的 ACLNN 调用示例。

## 目录结构

```text
asin_grad/
├── CMakeLists.txt                     # 构建配置（根）
├── README.md                          # 本文件
├── examples/                          # 调用示例
│   └── test_aclnn_asin_grad.cpp       # ACLNN 调用示例
├── op_host/                           # Host 侧实现
│   ├── CMakeLists.txt                 # Host 构建配置
│   ├── asin_grad_def.cpp              # 算子原型注册
│   ├── asin_grad_infershape.cpp       # Shape 推导
│   └── asin_grad_tiling.cpp           # Tiling 实现
├── op_kernel/                         # Kernel 侧实现
│   ├── asin_grad.cpp                  # Kernel 入口
│   ├── asin_grad.h                    # Kernel 类定义
│   ├── asin_grad_tiling_data.h        # TilingData 定义
│   └── asin_grad_tiling_key.h         # TilingKey 定义
└── tests/                             # 测试代码
```
