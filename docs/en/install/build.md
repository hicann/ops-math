# build Parameter Description

## Introduction

build.sh is the build script for this project, located by default in the project root directory. Its function is to automatically compile, link and configure source code, ultimately generating executable files, library files or other target files for installation or direct execution. Specifically, the script implements various functions by configuring different parameters, including building multiple target libraries (such as libophost_math.so), compiling operator packages, executing unit tests, etc.

## Usage Method

1. **Configure Environment Variables**

   Refer to [Environment Deployment](quick_install.md) to complete environment variable configuration.

   ```bash
   # Default path installation, using root user as example
   source /usr/local/Ascend/cann/set_env.sh
   ```

2. **Build Command Format**

   Taking the compile operator package command as an example, the style is as follows, where `--vendor_name`, `--ops` and `-j` are optional in this scenario. Appropriate compilation thread count can speed up compilation.

   ```bash
   bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] [-j${n}]
   ```

   For full parameter meanings, see the parameter description below. Please choose appropriate parameters according to actual situation.

## Parameter Description

build.sh supports multiple functions. You can view all function parameters through the following command.

```bash
bash build.sh --help
```

| Parameter Name | Optional/Required | Parameter Description |
|----------------|--------|------------------------------------------------------------------------------|
| -j${n} | Optional | Specify compilation thread count, ${n} is the specific thread count, default value is 8 (such as -j8); if thread count exceeds CPU core count, it will automatically adjust to CPU core count. |
| -v | Optional | View CMake compilation configuration information. |
| -O${n} | Optional | Specify compilation optimization level, supports O0/O1/O2/O3 (such as -O3), ${n} is optimization level identifier. |
| -u | Optional | Enable unit test (UT) compilation mode, compile all UT targets. |
| --help, -h | Optional | Print script usage help information. |
| --ops | Optional | Specify operators to compile, such as add,add_lora, multiple operators separated by English comma ",", cannot be used with --ophost, --opapi, --opgraph simultaneously. |
| --soc | Optional | Specify NPU model, each compilation only supports 1 NPU model. |
| --jit | Optional | In static graph scenario, when compiling `cann-${soc_name}-ops-math_${cann_version}_linux-${arch}.run` package, no need to compile operator binary files (graph runtime will compile online), can configure this option to improve compilation speed. |
| --static | Optional | After configuration, means generate static library files, including libcann_math_static.a and aclnn interface header files, combined with --pkg parameter, generate static library compressed package. |
| --vendor_name | Optional | Specify custom operator package name, default value is custom. |
| --debug | Optional | Enable debug mode. |
| --cov | Optional | Reserved parameter, developers do not need to pay attention temporarily. |
| --noexec | Optional | Only compile unit test binary files, do not automatically execute compiled UT executable files. |
| --opkernel | Optional | Compile binary kernel (kernel logs stored in build/binary/${soc}/bin/build_log). |
| --pkg | Optional | Generate installation package, cannot be used with -u (UT mode) or --ophost, --opapi, --opgraph simultaneously. |
| --disable_asan | Optional | Disable ASAN (AddressSanitizer) memory detection function. |
| --valgrind | Optional | Reserved parameter, developers do not need to pay attention temporarily. |
| --make_clean | Optional | Execute basic cleanup operation (clean compilation products), script exits after execution. |
| --ophost | Optional | Compile libophost_math.so library, cannot be used with --pkg, --ops simultaneously. |
| --opapi | Optional | Compile libopapi_math.so library, cannot be used with --pkg, --ops simultaneously. |
| --opgraph | Optional | Compile libopgraph_math.so library, cannot be used with --pkg, --ops simultaneously. |
| --ophost_test | Optional | Compile ophost related unit tests, equivalent to -u --ophost combination. |
| --opapi_test | Optional | Compile opapi related unit tests, equivalent to -u --opapi combination. |
| --opgraph_test | Optional | Reserved parameter, developers do not need to pay attention temporarily. |
| --opkernel_test | Optional | Compile opkernel related unit tests, equivalent to -u --opkernel combination. |
| --run_example | Optional | Compile specified operator and mode examples and execute compiled executable files. |
| --simulator | Optional | Enable simulator mode to execute --run_example task. In simulator mode, will link corresponding simulator library according to soc_version. |
| --genop | Optional | Create AI Core custom operator initial directory. |
| --genop_aicpu | Optional | Create AI CPU custom operator initial directory. |
| --experimental | Optional | Compile user operators under experimental directory. |
| --mssanitizer | Optional | Enable kernel-side mssanitizer memory detection function, cannot be used with --bisheng_flags simultaneously. |
| --oom | Optional | Enable kernel-side oom memory detection function, cannot be used with --bisheng_flags simultaneously. |
| --dump_cce | Optional | Enable kernel-side dump precompiled file function, cannot be used with --bisheng_flags simultaneously. |
| --bisheng_flags | Optional | Specify Bisheng compiler compilation parameters, multiple compilation parameters separated by English comma ",", cannot be used with --mssanitizer, --oom, --dump_cce simultaneously. |
| --kernel_template_input | Optional | Specify template parameters when compiling kernel, multiple template parameters separated by English semicolon ";", used with --ops and only one operator can be specified. |
| --cann_3rd_lib_path | Optional | Directory where third-party libraries are stored in offline compilation scenario. |
| --no_force | Optional | When the operator to be compiled depends on other operators, no longer compile other operator binary files |
