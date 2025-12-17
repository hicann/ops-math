# AI CPU算子开发指南

> 说明：
>
> - 算子开发过程中涉及的基本概念、AI CPU接口等，详细介绍请参考[《TBE&AI CPU算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevWizard)。
> - 若基于社区版CANN包对AI CPU算子源码修改，请使用自定义算子包方式编译执行。

开发指南以`AddExample`算子开发为例，介绍新算子开发流程以及涉及的交付件，流程图如下，完整样例代码请访问项目`examples`目录。

```mermaid
graph LR
	A([前提条件]) --> W([工程创建])
    W --> C([Kernel实现])
    C --> E([图模式适配])
    E --> F([编译部署])
    F --> G([算子验证])
```

1. [工程创建](#工程创建)：开发算子前，需按要求创建算子目录，方便后续算子的编译和部署。

2. [算子定义](#算子定义)：算子功能说明与原型定义。

3. [Kernel实现](#Kernel实现)：实现Device侧算子核函数。

4. [图模式适配](#图模式适配)：AI CPU算子目前仅支持图模式调用，需完成InferShape和InferDataType，后续会支持aclnn接口调用。

5. [编译部署](#编译部署)：通过工程编译脚本完成自定义算子的编译和安装。

6. [算子验证](#算子验证)：通过常见算子调用方式，验证自定义算子功能。  

##  工程创建
**1. 环境部署**

开发算子前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。

**2. 目录创建**

目录创建是算子开发的重要步骤，为后续代码编写、编译构建和调试提供统一的目录结构和文件组织方式。

本项目`build.sh`，支持快速创建算子目录。进入项目根目录，执行以下命令：

```bash
# 创建指定算子目录，如bash build.sh --genop=examples/add_example
# ${op_class}表示算子类型，如math类。
# ${op_name}表示算子名的小写下划线形式，如`AddExample`算子对应为add_example。
bash build.sh --genop_aicpu=${op_class}/${op_name}
```

如果命令执行成功，会看到如下提示信息：

```bash
Create the AI CPU initial directory for ${op_name} under ${op_class} success
```
创建完成后，关键目录结构如下所示：

```
${op_name}                              # 替换为实际算子名的小写下划线形式
├── op_host                             # Host侧实现
│   ├── ${op_name}_def.cpp              # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   ├── ${op_name}_infershape.cpp       # InferShape实现，实现算子形状推导，在运行时推导输出shape
│   └── CMakeLists.txt                  # Host侧cmakelist文件
├── op_graph                            # 图融合相关实现
│   ├── CMakeLists.txt                  # op_graph侧cmakelist文件
│   ├── ${op_name}_graph_infer.cpp      # InferDataType文件，实现算子类型推导，在运行时推导输出dataType
│   └── ${op_name}_proto.h              # 算子原型定义，用于图优化和融合阶段识别算子
├── op_kernel_aicpu                     # Device侧Kernel实现
│   ├── ${op_name}_aicpu.cpp            # Kernel入口文件，包含主函数和调度逻辑
│   └── ${op_name}_aicpu.h              # Kernel头文件，包含函数声明、结构定义、逻辑实现
├── test                                # ut文件夹 
└── CMakeLists.txt                      # 算子Cmakelist入口
```
使用上述命令行创建算子工程后，若要手动删除新创建出的算子工程，需要同时删除与算子工程同目录CMakeLists.txt中新添加的add_subdirectory(${op_class})。

## 算子定义
算子定义需要完成两个交付件：`README.md` `${op_name}_def.cpp` `${op_name}_proto.h`

**交付件1：README.md**

开发算子前需要先确定目标算子的功能和计算逻辑。

以自定义`AddExample`算子说明为例，请参考[AddExample算子说明](../../examples/add_example/README.md)。

**交付件2：${op_name}_def.cpp**

算子信息库。

以自定义`AddExample`算子说明为例，请参考[AddExample算子信息库](../../examples/add_example_aicpu/op_host/add_example_def.cpp)。

**交付件3：${op_name}_proto.h**

图模式调用需要将算子原型注册到[Graph Engine](https://www.hiascend.com/cann/graph-engine)（简称GE）中，以便GE能够识别该类型算子的输入、输出及属性信息。注册通过`REG_OP`接口完成，开发者需要定义算子的输入、输出张量类型及数量等基本信息。

完整代码请参考`examples/add_example_aicpu/op_graph`下[add_example_proto.h](../../examples/add_example_aicpu/op_graph/add_example_proto.h)。


## Kernel实现

Kernel是算子在NPU执行的核心部分，Kernel实现包括如下步骤：

```mermaid
graph LR
	H([算子类声明]) -->A([Compute函数实现])
	A -->B([注册算子])
```
Kernel一共需要两个交付件：`${op_name}_aicpu.cpp` `${op_name}_aicpu.h`

**交付件1：${op_name}_aicpu.h**
   算子类声明

   Kernel实现的第一步，需在头文件`op_kernel_aicpu/${op_name}_aicpu.h`进行算子类的声明，算子类需继承CpuKernel基类。
   示例如下，`AddExample`算子完整代码请参考`examples/add_example_aicpu/op_kernel_aicpu`下[add_example_aicpu.h](../../examples/add_example_aicpu/op_kernel_aicpu/add_example_aicpu.h)。


   ```CPP
// 1、算子类声明
// 包含AI CPU基础库头文件
#include "cpu_kernel.h"
// 定义命名空间aicpu(固定不允许修改)，并定义算子Compute实现函数
namespace aicpu {
// 算子类继承CpuKernel基类
class AddExampleCpuKernel : public CpuKernel {
 public:
  ~AddExampleCpuKernel() = default;
  // 声明函数Compute（需要重写），形参CpuKernelContext为CPUKernel的上下文，包括算子输入、输出和属性信息
  uint32_t Compute(CpuKernelContext &ctx) override;
};
}  // namespace aicpu
```

**交付件2：${op_name}_aicpu.cpp**
   Compute函数实现与AI CPU 算子注册

   获取输入/输出Tensor信息并进行合法性校验，然后实现核心计算逻辑（如加法操作），并将计算结果设置到输出Tensor中。
   
   示例如下，`AddExample`算子完整代码请参考`examples/add_example_aicpu/op_kernel_aicpu`下[add_example_aicpu.cpp](../../examples/add_example_aicpu/op_kernel_aicpu/add_example_aicpu.cpp)。

```C++
// 2、Compute函数实现
#include "add_example_aicpu.h"

namespace {
// 算子名
const char* const kAddExample = "AddExample";
const uint32_t kParamInvalid = 1;
}  // namespace

// 定义命名空间aicpu
namespace aicpu {
// 实现自定义算子类的Compute函数
uint32_t AddExampleCpuKernel::Compute(CpuKernelContext& ctx) {
  // 从CpuKernelContext中获取input tensor
  Tensor* input0 = ctx.Input(0);
  Tensor* input1 = ctx.Input(1);
  // 从CpuKernelContext中获取output tensor
  Tensor* output = ctx.Output(0);

  // 对tensor进行基本校验, 判断是否为空指针
  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    return kParamInvalid;
  }

  // 获取input tensor的数据类型
  auto data_type = static_cast<DataType>(input0->GetDataType());
  // 获取input tensor的数据地址，例如输入的数据类型是int32
  auto input0_data = reinterpret_cast<int32_t*>(input0->GetData());
  // 获取tensor的shape
  auto input0_shape = input->GetTensorShape();

  // 获取output tensor的数据地址，例如输出的数据类型是int32
  auto y = reinterpret_cast<int32_t*>(output->GetData());

  // AddCompute函数根据输入类型执行相应计算。
  // 由于C++自身不支持半精度浮点类型，可借助第三方库Eigen（建议使用3.3.9版本）表示。
  switch (data_type) {
    case DT_FLOAT:
      return AddCompute<float>(...);
    case DT_INT32:
      return AddCompute<int32>(...);
    case DT_INT64:
      return AddCompute<int64>(...);
      ....
    default : return PARAM_INVALID;
  }
}

// 3、注册算子Kernel实现，用于框架获取算子Kernel的Compute函数。
REGISTER_CPU_KERNEL(kAddExample, AddExampleCpuKernel);
}  // namespace aicpu
```

## 图模式适配

图模式适配请参考文档[graph_develop_guide.md](./graph_develop_guide.md)

## 编译部署

算子开发完成后，需对算子工程进行编译，生成自定义算子安装包\*\.run，详细的编译操作如下：

1. **准备工作。**

   参考[前提条件](#前提条件)完成基础环境搭建，同时检查算子开发交付件是否完备，是否在对应算子分类目录下。

2. **编译自定义算子包。**

   以`AddExample`算子为例，假设开发交付件在`examples`目录，完整代码参见[add_example_aicpu](../../examples/add_example_aicpu)目录。

   进入项目根目录，执行如下编译命令：

    ```bash
   # 编译指定算子，如--ops=add_example
   bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op_list}
    ```

   若提示如下信息，说明编译成功：

    ```bash
   Self-extractable archive "cann-ops-math-${vendor_name}_linux-${arch}.run" successfully created.
    ```

   若未指定\$\{vendor\_name\}默认使用custom作为包名。编译成功后，生成的自定义算子\*\.run包存放于build_out目录。
   说明：当前自定义算子包\$\{vendor\_name\}和\$\{op\_list\}均为可选，若都不传编译的是built-in包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。

   注意，构建过程文件在`build`目录，关键文件如下：

    - `libcust_opapi.so`：包含aclnn接口相关实现。
    - `libcust_opmaster_rt2.0.so`：包含Tiling相关实现。

3. **安装自定义算子包。**

   执行以下命令进行安装：

    ```bash
   ./build_out/cann-ops-math-${vendor_name}_linux-${arch}.run
    ```
   自定义算子包安装在`${ASCEND_HOME_PATH}/cann/opp/vendors`路径中，`${ASCEND_HOME_PATH}`表示CANN软件安装目录，可提前在环境变量中配置。自定义算子包不支持卸载。

## 算子验证
```bash
    # 执行前需要导入环境变量
    export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/cann/opp/vendors/${vendor_name}/op_api/lib:${LD_LIBRARY_PATH}
```

1. **图模式调用验证**

  开发好的算子完成编译部署后，可通过图模式验证功能，方法请参考[算子调用方式](../invocation/op_invocation.md)。