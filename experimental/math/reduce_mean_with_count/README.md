# ReduceMeanWithCount 自定义算子

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| Ascend 950PR / Ascend 950DT | Yes |
| Atlas A3 训练系列 / Atlas A3 推理系列 | No |
| Atlas A2 训练系列 / Atlas A2 推理系列 | No |
| Atlas 200I/500 A2 推理产品 | No |
| Atlas 推理系列产品 | No |
| Atlas 训练系列产品 | No |

## 功能说明

ReduceMeanWithCount 算子沿指定轴对输入张量执行归约求均值操作，同时输出参与归约的元素计数。将均值计算和计数操作融合为一次数据遍历，适用于分布式训练梯度聚合、LayerNorm / BatchNorm 统计量计算等场景。

**计算公式**:

$$
\text{meanResult}_j = \frac{\sum_{i \in \text{reduce\_axes}} \text{input}_{j,i}}{N}
$$

$$
\text{countResult}_j = N
$$

其中 j 为非归约维度索引，i 为归约维度索引，N 为沿归约轴的元素总数（各归约维度 size 的乘积）。

## 参数说明

### 输入

| 参数名 | 类型 | 必选/可选 | 说明 |
|--------|------|-----------|------|
| input | aclTensor* | 必选 | 输入张量，支持 0-8 维，数据格式 ND。支持数据类型: FLOAT32, FLOAT16, BFLOAT16。不支持空 Tensor。 |
| axis | aclIntArray* | 可选 | 归约轴列表。指定沿哪些维度进行归约。传入 nullptr 或空数组时对所有维度归约。每个值需在 [-rank, rank-1] 范围内，不允许重复（归一化后）。 |
| keepdim | bool | 可选 | 是否保持归约维度，默认 false。为 true 时归约维度保留为 1；为 false 时归约维度被消除。 |

### 输出

| 参数名 | 类型 | 说明 |
|--------|------|------|
| meanResult | aclTensor* | 均值结果张量。数据类型与 input 一致。shape 由 input shape、axis、keepdim 共同决定。 |
| countResult | aclTensor* | 计数结果张量。数据类型固定为 INT64。shape 与 meanResult 一致，所有元素值为参与归约的元素总数 N。 |

### 输出 Shape 推导规则

设输入 shape 为 `[d0, d1, ..., d_{n-1}]`，归约轴为 `axes`:

- **keepdim=false**: 从输出 shape 中移除 axes 指定的维度
  - 例: input `[2, 3, 4]`, axis=`[1]` -> output `[2, 4]`
- **keepdim=true**: axes 指定的维度变为 1
  - 例: input `[2, 3, 4]`, axis=`[1]` -> output `[2, 1, 4]`
- **axis 为空**: 对所有维度归约
  - keepdim=false -> output `[]`（标量）
  - keepdim=true -> output `[1, 1, ..., 1]`（与输入同维度）

## 约束说明

- 仅支持 Ascend 950PR / Ascend 950DT（arch35, DAV_3510）。
- input 不支持空 Tensor（元素数为 0）。
- axis 中每个值需在 [-rank, rank-1] 范围内，不允许重复维度（归一化后）。
- meanResult 的数据类型必须与 input 一致；countResult 的数据类型必须为 INT64。
- meanResult 和 countResult 的 shape 必须与根据 input shape、axis、keepdim 推导出的 shape 一致。
- 支持非连续 Tensor 输入（内部自动做 Contiguous）。
- 默认确定性实现，相同输入保证相同输出。
- FP16/BF16 类型内部使用 FP32 中间累加，避免大规模归约溢出。

## 调用说明

本算子通过 ACLNN 两段式接口调用，完整调用流程:

1. 调用 `aclnnReduceMeanWithCountGetWorkspaceSize` 获取 workspace 大小和执行器
2. 申请 Device 端 workspace 内存
3. 调用 `aclnnReduceMeanWithCount` 执行计算
4. 同步 stream，读取结果

详细调用示例参见 [examples/](examples/) 目录:

- [test_aclnn_reduce_mean_with_count.cpp](examples/test_aclnn_reduce_mean_with_count.cpp) - ACLNN 两段式调用示例（含精度比对）
- [test_geir_reduce_mean_with_count.cpp](examples/test_geir_reduce_mean_with_count.cpp) - GE IR 图模式调用示例（占位）

### 编译运行

```bash
# 1. 设置 CANN 环境
source /path/to/cann/set_env.sh

# 2. 编译算子包
cd reduce_mean_with_count
bash build.sh --soc=ascend950

# 3. 安装自定义算子包
./build/custom_opp_ubuntu_*.run --install

# 4. 编译示例
cd examples
g++ -std=c++17 -o test_aclnn test_aclnn_reduce_mean_with_count.cpp \
    -I${ASCEND_HOME_PATH}/include \
    -L${ASCEND_HOME_PATH}/lib64 \
    -lascendcl -lnnopbase -lopapi -lcust_opapi

# 5. 运行
./test_aclnn
```

## 精度说明

本算子采用 MERE/MARE 精度标准（单标杆比对），标杆为 PyTorch CPU 实现。

**误差指标定义**:

- MERE (Mean Relative Error): `avg(|actual - golden| / (|golden| + 1e-7))`
- MARE (Max Relative Error): `max(|actual - golden| / (|golden| + 1e-7))`

**各数据类型精度标准**:

| 数据类型 | MERE Threshold | MARE Threshold | 判定条件 |
|----------|---------------|----------------|----------|
| FLOAT32 | 2^-13 (~0.000122) | 2^-13 * 10 (~0.00122) | MERE < Threshold AND MARE < 10 * Threshold |
| FLOAT16 | 2^-10 (~0.000977) | 2^-10 * 10 (~0.00977) | MERE < Threshold AND MARE < 10 * Threshold |
| BFLOAT16 | 2^-7 (~0.00781) | 2^-7 * 10 (~0.0781) | MERE < Threshold AND MARE < 10 * Threshold |

- count_result 为整数计数，要求与 Golden 完全一致（精确匹配）。
- 输入包含 NaN/Inf 时，按 NaN/Inf 一致性标准判定。

## 目录结构

```text
reduce_mean_with_count/
├── CMakeLists.txt                 # 顶层构建文件
├── build.sh                       # 一键编译脚本
├── README.md                      # 本文件
├── op_host/                       # Host 端代码
│   ├── reduce_mean_with_count_def.cpp          # 算子定义与注册
│   ├── reduce_mean_with_count_infershape.cpp   # Shape 推导
│   └── arch35/
│       └── reduce_mean_with_count_tiling.cpp   # Tiling 切分逻辑
├── op_kernel/                     # Kernel 端代码
│   ├── reduce_mean_with_count_arch35.cpp       # Kernel 入口
│   └── arch35/
│       ├── reduce_mean_with_count.h            # Kernel 实现
│       ├── reduce_mean_with_count_tiling_data.h # Tiling 数据结构
│       └── reduce_mean_with_count_tiling_key.h  # TilingKey 定义
├── op_api/                        # ACLNN 接口封装
│   ├── aclnn_reduce_mean_with_count.h
│   ├── aclnn_reduce_mean_with_count.cpp
│   ├── reduce_mean_with_count.h
│   └── reduce_mean_with_count.cpp
├── examples/                      # 调用示例
├── tests/                         # 测试用例
│   ├── ut/                        # 单元测试
│   └── st/                        # 系统测试
├── docs/                          # 文档
│   ├── REQUIREMENTS.md            # 需求文档
│   ├── DESIGN.md                  # 设计文档
│   ├── PLAN.md                    # 迭代计划
│   └── LOG.md                     # 开发日志
└── issues/                        # 问题记录
```
