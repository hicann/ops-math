# 算子开发指南

## 概述

本章以`AddExample`算子为例，介绍算子开发过程以及涉及的交付件。端到端流程如图所示：

```mermaid
graph LR
	A[(环境准备)] -->B([算子设计])
    B --> C([工程创建])
    C -->D([算子实现])
    D -->E([编译部署])
    E -->F([算子调用])
```


1. [环境准备](#环境准备)：开发算子前，请确保依赖的驱动、固件、CANN软件包等已安装。
2. [算子设计](#算子设计)：根据实际业务场景和硬件本身要求，设计合适的算子输入、输出、属性等信息，包括shape、数据类型、数据格式等。
3. [工程创建](#工程创建)：一键创建工程目录。
5. [算子Kernel实现](#算子Kernel实现)：实现Device侧算子的核函数。
6. [InferShape与InferDataType实现](#InferShape与InferDataType实现)（可选）：仅当需要算子入图时，才需要完成dataType推导与shape推导。
7. [框架适配](#框架适配)：aclnn接口适配及图模式适配。
8. [算子包编译部署](#算子包编译部署)：通过工程编译脚本完成算子的编译和安装。

对于上述流程，在不同场景下，算子开发实现的步骤不同，请根据实际情况按需选择。
- KernelLaunch调用：实现步骤1~5、步骤8
- aclnn调用：实现步骤1~5、步骤7、步骤8
- 图模式调用：实现步骤1~8

## 环境准备

环境准备参考项目首页[环境准备](../README.md#环境准备)。

## 算子设计

算子设计是整个算子开发流程中的核心环节，其目标是将模型中的数学计算逻辑转化为可执行的代码逻辑，并为后续的工程实现、编译部署和调用提供清晰的接口和规范。

以下以`AddExample`样例算子为例，介绍算子设计的完整流程，并结合流程图进行说明。

```mermaid
graph LR
	A[(设计数学表达式)] -->B([明确输入输出])
    B --> C([设计核函数])
    C -->D([明确所需接口])
```

### 1、设计数学表达式

算子设计的第一步是明确其数学表达式。`AddExample`算子的功能是对两个四维张量进行逐元素相加操作，其数学表达式为
```
y[i] = x1[i] + x2[i]
```
其中`i`表示张量中每个元素的索引，`x1`和`x2`是两个输入张量，`y`是输出张量。

### 2、明确输入和输出

在明确了算子的数学表达式之后，需要进一步明确其输入输出的格式、数据类型和形状。
- 输入：

        - x1: 四维张量，shape为(32,4,4,4)
        - x2: 四维张量，shape为(32,4,4,4)
        - 数据类型支持：float32或int32
        - 数据格式: ND
- 输出：

        - y: 四维张量，shape为(32,4,4,4)
        - 数据类型与输入一致
        - 数据格式: ND

### 3、确定核函数名称和参数

核函数是算子的核心实现部分，负责完成具体的计算逻辑。
- 核函数名称：add_example

- 参数说明：

        - x1: 第一个输入张量
        - x2: 第一个输入张量
        - y: 输出张量
核函数的参数顺序为：输入1，输入2，输出。

核函数名称以**算子名小写下划线**命名。

### 4、确定算子实现所需接口

在实现算子时，需要调用Ascend C提供的相关接口，完成数据搬运、内存管理、计算操作与任务调度等功能。

1、数据搬运：

- 使用`DataCopy`接口实现从外部存储搬运到内部存储的数据搬运。

2、相加计算：

- 使用双目运算接口`Add`实现两个张量逐元素相加操作。

3、内存管理：

- 使用`AllocTensor`申请张量内存空间。
- 使用`FreeTensor`释放张量内存空间。

4、任务调度与同步：
- 使用`EnQue`和`DeQue`接口实现并行任务之间的同步与调度。

### 5、算子设计规格总结

通过以上分析，`AddExample`算子的设计规格如下：

<table>
<tr>
<th>算子类型</th>
<td colspan="4" align="center">AddExample</td>
</tr>
<tr>
<th>算子表达式</th>
<td colspan="4" align="center">y[i] = x1[i] + x2[i] </td>
</tr>
<tr>
<th rowspan="3" >算子输入</th>
<th>name</th>
<th>shape</th>
<th>dataType</th>
<th>format</th>
</tr>
<tr>
<td>x1</td>
<td>(32,4,4,4)</td>
<td>float/int32</td>
<td>ND</td>
</tr>
<tr>
<td>x2</td>
<td>(32,4,4,4)</td>
<td>float/int32</td>
<td>ND</td>
</tr>
<tr>
<th>算子输出</th>
<td>y</td>
<td>(32,4,4,4)</td>
<td>float/int32</td>
<td>ND</td>

</tr>
<tr>
<th rowspan="5" >算子实现文件</th>
<td colspan="4" >[add_example.h]</td>
</tr>
</table>

## 工程创建

工程创建时算子开发中的重要步骤，它为后续的代码编写、编译构建和调试提供统一的目录结构和文件组织方式。后续支持使用工具一键生成项目目录结构。当前手动创建。

以下是`AddExample`样例算子的工程目录结构说明：

```
├── op_graph                                        // 图融合相关
│   ├── CMakeLists.txt                              // op_graph侧cmakelist文件
│   ├── add_example_graph_infer.cpp                 // 算子inferDataType文件
│   └── add_example_proto.h                         // 算子原型
├── op_host                                         // 算子Host侧实现目录
│   ├── add_example_def.cpp                         // 算子信息库
│   ├── add_example_infershape.cpp                  // 算子InferShape实现
│   └── CMakeLists.txt                              // host侧cmakelist文件
└── op_kernel_aicpu                                 // 算子Device侧Kernel实现目录
│   ├── add_example.cpp                             // 算子kernel入口文件
│   └── add_example.h                               // 算子kernel实现文件
└── CMakeLists.txt                                  // 算子cmakelist入口
```

### 目录与文件说明：

1、`op_graph`目录

该目录用于图融合阶段的相关文件，主要包含算子原型定义文件。
- `add_example_proto.h`: 定义算子的原型信息，用于图优化和融合阶段识别算子。
- `add_example_graph_infer.cpp`: 实现算子的类型推导逻辑，用于在运行时推导输出张量的dataType。
- `CMakeLists.txt`: 用于配置`op_graph`模块的构建规则。

2、`op_host`目录

该目录存放算子在Host侧的实现文件，主要包括算子的元信息，形状推导，任务划分等逻辑。
- `add_example_def.cpp`: 定义算子的基本信息，如名称、输入输出数量、数据类型等。
- `add_example_infershape.cpp`: 实现算子的形状推导逻辑，用于在运行时推导输出张量的shape。
- `CMakeLists.txt`: 用于配置`op_host`模块的构建规则。

3、`op_kernel`目录
- `add_example.cpp`: 算子kernel的入口文件，包含主函数和调度逻辑。
- `add_example.h`: 定义kernel的头文件，包含函数声明，结构定义及逻辑实现。

## 算子Kernel实现
### kernel简介

Kernel实现即算子核函数实现，AICPU算子的开发接口即为原生c++接口，开发完成后在NPU上执行。

kernel入口文件以`{算子名小写下划线格式}.cpp`命名，如`add_example.cpp`。kernel核心实现在`.h`头文件中编写。

完整样例参考[add_example.h](../../example/add_example_aicpu/op_kernel_aicpu/add_aicpu_example.h)。文档参考[Kernel侧算子实现](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/tbeaicpudevg/atlasopdev_10_0072.html)。

## InferShape与InferDataType实现

在深度学习中，当一个算子（Op）被加入计算图时，为了确保图的正确性和后续的编译、优化、执行流程顺利进行，通常需要为该算子实现两个关键的推导函数：
  - InferShape：用于推导输出张量的形状（shape）。
  - InferDataType：用于推导输出张量的数据类型（dataType）。

### 1、注册InferShape与InferData
在实现这两个函数之前，需要先进行注册，告诉框架该算子的shape和data type 推导逻辑由哪两个函数来处理。注册方式如下：

`AddExample`样例算子实现数相加的逻辑，其输出的shape大小与输入相同，输出的dataType与输入一致。

Infershape
```C++
IMPL_OP_INFERSHAPE(AddExample).
    InferShape(InferShapeAddExample);
```

InferDataType
```C++
IMPL_OP(AddExample).
    InferDataType(InferDataTypeAddExample);
```
- AddExample：算子的类名。
- InferShapeAddExample：shape推导函数。
- InferDataTypeAddExample：data type 推导函数。

<br>

**命名规范**

- shape推导函数：`InferShape算子名大驼峰`，如`InferShapeAddExample`。
- data type 推导函数：`InferDataType算子名大驼峰`，如`InferDataTypeAddExample`。
- shape推导实现文件：`算子名小写下划线+infershape.cpp`，如[add_example_infershape.cpp](../../example/add_example/op_host/add_example_infershape.cpp)。
- dataType推导实现文件：`算子名小写下划线+graph_infer.cpp`，如[add_example_graph_infer.cpp](../../example/add_example/op_graph/add_example_graph_infer.cpp)。

### 2、InferShape推导实现
Infershape函数的作用是根据输入的shape推导输出的shape。对于`AddExample`样例算子来说，其逻辑是两个数相加，因此输出的shape与输入的shape一致。
```C++
// 获取输入shape
const gert::Shape* xShape = context->GetInputShape(IDX_0);
// 获取输出shape
gert::Shape* yShape = context->GetOutputShape(IDX_0);
// 获取输入DimNum
auto xShapeSize = xShape->GetDimNum();
// 设置输出的DimNum
yShape->SetDimNum(xShapeSize);
// 依次将输入Dim值设置给输出
for (size_t i = 0; i < xShapeSize; i++) {
    int64_t dim = xShape->GetDim(i);
    yShape->SetDim(i, dim);
}
```

### 2、InferDataType推导实现
InferDataType函数的作用是根据输入的data type推导输出的data type。对于`AddExample`样例算子来说，其逻辑是两个数相加，因此输出的data type与输入的shape一致。
```C++
// 获取输入的dataType
ge::DataType sizeDtype = context->GetInputDataType(IDX_0);
// 将输出dataType设置到输出
context->SetOutputDataType(IDX_0, sizeDtype);
```

完整样例参考[add_example_infershape.cpp](../../example/add_example/op_host/add_example_infershape.cpp)。

## 框架适配

当前算子调用支持两种方式：`aclnn调用`和`图模式调用`。

### 1、aclnn适配

aclnn调用方式，是指直接调用aclnn接口，基于C语言的API执行算子。完成自定义算子编译后，会自动生成aclnn，可以直接在应用程序中调用。文档参考[aclnn调用](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0070.html)。

使用aclnn调用方式，需要依赖算子的二进制包。为了生成该二进制包，需要完成以下配置步骤：

- 二进制配置文件以`算子名小写下划线格式+binary.json`命名放在支持的`soc版本`目录下，如 [add_example_binary.json](../../example/add_example/op_host/config/ascend910b/add_example_binary.json) ，文件中配置算子的输入、输出shape、data type、format等信息。

- [ascendc_config.json](../../scripts/kernel/binary_config/ascendc_config.json) 中注明算子`soc版本`及实现模式，如：
```json
    {"name":"AddExample", "compute_units": ["${soc_version}"], "auto_sync":true, "impl_mode" : "high_performance"},
```

**命名规范**
- 二进制配置json：`算子名小写下划线格式+binary.json` 如 `add_example_binary.json`

### 2、图模式适配

图模式调用，需要将算子原型注册到Graph Engine（简称GE）中，以便GE能够识别该类型算子的输入、输出及属性信息。注册通过`REG_OP`接口完成。开发者需要定义算子的输入输出张量类型及数量等基本信息。

以下示例代码，展示了如何注册`AddExample`样例算子。
```c++
REG_OP(AddExample)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .OP_END_FACTORY_REG(AddExample)
```

完整样例参考[add_example_proto.h](../../example/add_example/op_graph/add_example_proto.h)。
## 算子包编译部署

### 1、环境准备

环境准备参考[环境准备](#环境准备)。

安装base包，参考项目[ops-base-dev](https://gitcode.com/cann/ops-base-dev#编译执行)编译执行章节。

### 2、样例算子获取

参考项目首页[源码获取](../../README.md#源码下载)。获取源码之后，样例代码在
[add_example](../../example/add_example/)。

### 3、编译自定义算子包

进入本项目根目录，执行如下编译命令：

```bash
# 编译指定算子，如add_example
bash build.sh --package --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op1,op2,...}
```
- --soc：\$\{soc\_version\}表示NPU型号，可通过`npu-smi info`命令查询，在查询到的“Name”前增加ascend信息，例如“Name”取值为xxxyy（仅保留xxx部分），实际配置的soc\_version值为ascendxxx（注意全改为小写）。
- --vendor_name：\$\{vendor\_name\}表示构建自定义算子包包名。
- --ops（可选）：**仅编译指定算子场景设置（若不设置则默认编译全部算子）**。\$\{op1,op2,...\}表示待编译的算子，多个算子之间使用英文逗号”,“分隔。

若提示如下信息，则说明编译成功：

```bash
Self-extractable archive "cann-ops-math-${vendor_name}-linux.${arch}.run" successfully created.
```

如果未指定`${vendor_name}`则默认使用`custom`作为名称。编译成功后，生成的`.run`包存放于build_out目录下。

构建过程文件见在`build`目录中，部分说明：

- `libcust_opapi.so`：包含aclnn接口相关实现。
- `libcust_opmaster_rt2.0.so`：包含tiling相关实现。
- `binary/{soc_version}/bin/{soc_version}/{op_name}/{OpName}_*.o`：对应算子的二进制文件。

构建结果文件见`build_out`目录，部分说明：
- cann-ops-math-${vendor_name}-linux.${arch}.run：自解压格式的算子自定义包，可用于部署和安装。

### 4、安装自定义算子包
执行以下命令进行安装：
```bash
./${vendor_name}-ops-math-${cann_version}-linux.${arch}.run
```
安装完成后，自定义算子包将被存储在如下路径中：
```bash
`${ASCEND_HOME_PATH}/latest/opp/vendor`

```
其中`${ASCEND_HOME_PATH}`是在[环境准备](#环境准备)章节通过环境变量设置的路径，表示软件安装的根目录。

安装完成后，自定义算子包的目录结构示例如下，路径从`${ASCEND_HOME_PATH}/latest/opp/vendor`展开：
```
├── cann-ops-math-${vendor_name}-linux.${arch}      // 包名
├── bin
│   └── set_env.bash                                // 环境变量source脚本
├── op_api
│   ├── include
│   │   ├── aclnn_add_example.h                     // aclnn头文件
│   └── lib
│       └── libcust_opapi.so                        // 算子 aclnn接口so
├── op_impl
│   └── ai_cpu
│       └── tbe
│           ├── config
│           │   └── ${soc_version}
│           │       └── aic-${soc_version}-ops-info.json     // 算子信息库
│           ├── custom_impl
│           │   ├── ascendc
│           │   │   ├── add_example
│           │   ├── add_example.cpp                     // kernel实现
│           │   │   ├── add_example.h
│           │   └── dynamic
│           │       └── add_example.py
│           ├── kernel
│           │   ├── ${soc_version}                      // 二进制文件
│           │   │   └── add_example
│           │   │       ├── AddExample_11132827238e1555db7b997c7bce2928_high_performance.json
│           │   │       ├── AddExample_11132827238e1555db7b997c7bce2928_high_performance.o
│           │   │       ├── AddExample_a1532827238e1555db7b997c7bce2928_high_performance.json
│           │   │       └── AddExample_a1532827238e1555db7b997c7bce2928_high_performance.o
│           │   └── config
│           │       └── ${soc_version}                  // 二进制配置
│           │           ├── add_example.json
│           │           └── binary_info_config.json
│           └── op_tiling                               // tiling 相关
│               ├── lib
│               │   └── linux
│               │           └── ${arch}
│               │               └── libcust_opmaster_rt2.0.so
│               └── liboptiling.so -> lib/linux/${arch}/libcust_opmaster_rt2.0.so
├── op_proto
│   ├── inc
│   │   └── add_example_proto.h
│   └── lib
│       └── linux
│           └── ${arch}
│               └── libcust_opsproto_rt2.0.so
└── version.info                                        // 包信息
```

## 算子验证

开发好的算子可通过多种方式调用，本项目已提供常见的调用方式（如单算子模式、图模式、AI框架调用（如PyTorch）等），详细的算子调用流程请参见[算子调用示例](./算子调用样例.md)。同时，也支持开发者对接自身业务框架调用，如有调用遇到困难可通过issue方式联系技术支持。
