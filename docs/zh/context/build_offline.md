# 离线编译
离线编译是指在没有连接互联网的环境下，将软件源代码编译成可执行程序，并安装或配置到目标服务器上的过程。
本项目[算子调用](../invocation/quick_op_invocation.md)或[算子开发](../develop/aicore_develop_guide.md)过程中均需编译算子包，编译过程中会依赖一些开源第三方软件，这些软件联网时会自动下载，离线状态下无法直接下载。

本章提供了离线编译安装指导，在此之前请确保已按[环境部署](quick_install.md)完成基础环境搭建。
## 获取依赖
离线编译前，需准备如下依赖。

- 依赖json

下载[json](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip)

- 依赖makeself

下载[makeself](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz)

- 依赖eigen

下载[eigen](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0/eigen-5.0.0.tar.gz)

- 依赖protobuf

下载[protobuf](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz)

- 依赖abseil-cpp

下载[abseil-cpp](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz)


在代码仓目录下新建`third_party`目录，并将上述所有依赖包放置到`third_party/`下

## 离线编译
编程过程中会自动使用`third_party/`下的依赖：

```bash
# 自定义算子包编译
bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]

# ops-math整包编译
bash build.sh --pkg [--jit] --soc=${soc_version}
```

> 编译指令详细信息可查看[算子调用-编译执行](../invocation/quick_op_invocation.md#编译执行)