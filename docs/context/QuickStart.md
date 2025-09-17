# 快速入门

## 目录结构

关键目录如下：

```
├── build.sh                       # 项目工程编译脚本
├── cmake                          # 项目工程编译目录
├── CMakeLists.txt
├── common                         # 项目公共头文件和源代码                 
├── docs                           # 项目算子相关文档介绍
├── example                        # 项目通用算子开发和调用示例               
├── conversion                     # conversion类算子
├── ...
├── math                           # math类算子
│   ├── abs                        # abs算子所有交付件，如Tiling、Kernel、InferShape等
│   ├── ...
├── tests                          # 测试用例目录
├── README.md
├── requirements.txt               # 本项目需要的第三方依赖包
└── scripts                        # 脚本目录，包含自定义算子、Kernel构建相关配置文件
```

## 版本配套

- 本项目会创建与CANN软件版本适配的标签并发行，两者的配套关系请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"。**需注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与GitCode标签源码，使用master分支可能存在版本不匹配风险。**

- 本项目支持的固件驱动版本与配套CANN软件支持的固件驱动版本相同，开发者可通过“[昇腾社区-固件与驱动](https://www.hiascend.com/hardware/firmware-drivers/community?product=2&model=28)”，根据产品型号与CANN软件版本获取配套的固件与驱动。

## 环境准备

> 说明：
>
> 本项目支持与CANN 8.3.RC1及之前商发版本的开发套件`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`配合使用，使用指导请参见CANN 8.3.RC1商发文档中“[版本说明](https://idp.huawei.com/idp-designer-war/design?op=edit&locate=newMode/EDIT/53011869427/zh-cn_BOOKMAP_0000002456008653/ZH-CN_TOPIC_0000002422451490/2)”。

ops-math项目支持源码编译，进行源码编译前，请根据如下步骤完成相关环境准备。

1. **获取软件包**

   请参见"[开放项目与CANN版本配套表](https://gitee.com/ascend/cann-community/blob/master/README.md#cannversionmap)"获取对应的CANN软件包`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`。
   - \$\{package\}表示待安装的CANN软件包名
   - \$\{cann\_version\}表示CANN包版本号
   - \$\{arch\}表示CPU架构，如aarch64、x86_64

   为确保您的源码定制开发顺利，请选择与CANN版本配套的GitCode分支源码，使用master分支可能存在版本不匹配风险。

2. **安装软件包**

   注意，执行安装命令时，请确保安装用户对软件包具有可执行权限。

   基础环境的搭建请参见《[CANN 软件安装指南](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)》，按要求完成NPU驱动和固件、`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`软件包的安装。

3. **安装依赖**

   使用`git `命令下载对应分支ops-math源码，源码编译用到的依赖如下，请确保已安装并且满足版本要求。

   - python >= 3.7.0
   - gcc >= 7.3.0
   - cmake >= 3.16.0
   - pigz（可选，安装后可提升打包速度，建议版本 >= 2.8）
   - dos2unix
   - googletest（仅执行UT时依赖，建议版本 [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0)）

   - 本项目需要使用的python依赖包，具体参见工程目录中requirements.txt，可通过如下命令一键安装：
     ```bash
     pip3 install -r requirements.txt
     ```

4. **环境变量配置**

    根据实际场景，选择合适的命令。

    ```bash
    # 默认路径安装，以root用户为例（非root用户，将/usr/local替换为${HOME}）
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # 指定路径安装
    # source ${install-path}/ascend-toolkit/set_env.sh
    ```

## 编译执行
> 说明：
>
> 若基于CANN 8.3.RC1及之前商发版本的开发套件包`Ascend-cann-toolkit_${cann_version}_linux-${arch}.run`进行算子源码定制化修改，请使用“自定义算子包”方式进行编译和安装。

基于CANN软件包`Ascend-cann-${package}_${cann_version}_linux-${arch}.run`进行算子源码定制化修改时，支持使用[自定义算子包](#自定义算子包)和[ops-math包](#ops-math包)方式进行编译和安装。

编译方式说明：

- 自定义算子包：选择ops-math项目中部分算子编译生成的包称为自定义算子包，不改变原始CANN软件包，**挂载优先级更高的自定义算子包**。
- ops-math包：选择ops-math完整项目编译生成的包称为ops-math包，可**完整替换**CANN软件包中对应的部分。

### 自定义算子包

1. **编译自定义算子包。**

    进入本项目根目录，执行如下编译命令：
    
    ```bash
    # 方式1：编译所有算子
    bash build.sh --package --soc=${soc_version} --vendor_name=${vendor_name}
    # 方式2：编译指定算子，如op1、op2
    bash build.sh --package --soc=${soc_version} --vendor_name=${vendor_name} --ops=${op1,op2,...}
    ```
    - --soc：\$\{soc\_version\}表示NPU型号，可通过`npu-smi info`命令查询，在查询到的“Name”前增加ascend信息，例如“Name”取值为xxxyy，实际配置的soc\_version值为ascendxxxyy。
    - --vendor_name：\$\{vendor\_name\}表示构建的自定义算子包名称。
    - --ops（可选）：**仅编译部分算子设置**。\$\{op1,op2,...\}表示待编译的算子，多个算子之间使用英文逗号","分隔。
    
    若提示如下信息，则说明编译成功。
    ```bash
    Self-extractable archive "cann-ops-math-${vendor_name}-linux.${arch}.run" successfully created.
    ```
    ${vendor_name}如果没有输入，则默认名为custom，编译成功后，run包存放于项目根目录中的build_out目录下。

2. **安装自定义算子包。**
    ```bash
    ./cann-ops-math-${vendor_name}-linux.${arch}.run
    ```

    自定义算子包安装路径为`${ASCEND_HOME_PATH}/opp/vendors`，\$\{ASCEND\_HOME\_PATH\}已在[环境准备](#环境准备)章节通过环境变量配置。

### ops-math包

1. **编译ops-math包。**

    进入本项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --package [--jit] --soc=${soc_version}
    ```
    - --jit（可选）：推荐设置，表示不编译算子的二进制文件。
    - --soc：\$\{soc\_version\}表示NPU型号，可通过`npu-smi info`命令查询，在查询到的“Name”前增加ascend信息，例如“Name”取值为xxxyy，实际配置的soc\_version值为ascendxxxyy。

    若提示如下信息，则说明编译成功。

    ```bash
    Self-extractable archive "CANN-ops-math-${cann_version}-linux.${arch}.run" successfully created.
    ```

    编译成功后，run包存放于build_out目录下。

2. **安装ops-math包。**
   
    ```bash
    ./CANN-ops-math-${cann_version}-linux.${arch}.run --full --quiet 
    ```

    ops-math包默认安装路径为：`/usr/local/Ascend`，如需自定义安装路径，可使用"--install-path"参数指定。

## 本地验证 
通过根目录下的build.sh脚本可执行算子样例、UT用例等，build全量参数介绍参见[build参数说明](./build参数说明.md#参数说明)。

- **执行算子样例**
  
    安装完编译生成的包后，可执行项目中已有算子的调用样例。编译执行命令如下：

    ```bash
    bash build.sh --run_example ${op} ${mode}
    ```
    - \$\{op\}：表示待执行样例的算子名。
    - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    
    执行完成后会打印运行结果。

- **执行算子UT**

   安装完编译生成的包后，可执行UT验证功能是否正常，具体命令如下：

   > 说明：执行UT用例依赖googletest单元测试框架，介绍请参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

   ```bash
   # 方式1: 编译并执行所有的UT测试用例
   bash build.sh -u
   # 方式2: 编译所有的UT测试用例但不执行
   bash build.sh -u --noexec
   # 方式3: 编译并执行对应功能的UT测试用例（四选一）
   bash build.sh -u --[opgraph|opapi|ophost|opkernel]
   # 方式4: 编译对应功能的UT测试用例但不执行（四选一）
   bash build.sh -u --noexec --[opgraph|opapi|ophost|opkernel]
   ```
   假设验证ophost功能是否正常，执行如下命令：
   ```bash
   bash bulid.sh -u --ophost
   ```
   执行完成后出现如下内容，表示执行成功。
   ```bash
   Global Environment TearDown
   [==========] ${n} tests from ${m} test suites ran. (${x} ms total)
   [  PASSED  ] ${n} tests.
   [100%] Built target math_op_host_ut
   ```
   其中\$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。