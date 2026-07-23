# Environment Deployment

Before operating various [learning tutorials](../../../README.md#学习教程), please first complete basic environment setup and source code download by referring to the following steps, ensuring that NPU driver, firmware and CANN software (`Ascend-cann-toolkit` and `Ascend-cann-ops`) are installed.

## Environment Installation

This project provides multiple ways to set up Ascend environment, please choose as needed.

> **Note**: The meanings of compilation mode and runtime mode mentioned in this document are as follows, please choose according to actual situation.
>
> - Compilation mode: For scenarios where only this project is compiled without running, only need to install CANN toolkit package.
> - Runtime mode: For scenarios where this project is run (compile and run or pure run), need to install driver and firmware, CANN toolkit package, CANN ops package.

| Installation Method | Usage Instructions | Usage Scenario |
| ----- | ------ | ------ |
| CANNLab | One-stop development platform, provides online directly running Ascend environment, no need to manually install.<br>Currently can provide single machine computing power, **default installs latest version CANN package**. | Suitable for developers without Ascend devices. |
| Docker | Docker image is an efficient deployment method, has pre-integrated CANN package and essential dependencies.<br>Currently suitable for Atlas A2, A3 series products, OS supports ubuntu22.04, openeuler24.03. **Default installs latest version CANN package**. | Suitable for developers with Ascend devices, need to quickly set up environment. |
| Manual Installation | Manually install CANN package and basic dependencies, high flexibility. | Suitable for developers with Ascend devices, want to experience manual CANN package installation or experience latest master branch capabilities. |
| Spack | Package management tool, automatically installs dependencies and manages compilation options, supports one-click script or custom configuration.<br>**Automatically installs CANN package and compilation dependencies** (runtime mode requires host machine to pre-install driver and firmware). | Suitable for developers with Ascend devices, need to automatically manage CANN package version, compilation options and dependencies, supports multi-configuration and customized build (especially suitable for HPC environment). |

### Method 1: CANNLab

For developers without Ascend devices, can directly use CANNLab cloud development environment, that is "**One-stop Development Platform**". This platform provides you with online directly running Ascend environment, where essential driver firmware, software packages and dependencies are installed, no need to manually install.

> **Note**: Environment default installs latest version CANN package, pay attention to match with software when downloading source code. For more introduction about development platform, please refer to [CANNLab Guide](https://gitcode.com/org/cann/discussions/54).

1. Enter open source project, click "CANNLab" button, log in with authenticated Huawei Cloud account. If not registered or authenticated, please register and authenticate according to page prompts.
    <!--
   <img src="../figures/cloudIDE.png" alt="Cloud Platform" width="750px" height="85px">
    -->
2. Create NPU environment and configure specifications according to page prompts. After starting cloud development environment, click "`Connect > WebIDE`" to enter one-stop development platform.

   Current open source project resources are default in `/mnt/workspace/gitCode/${gitCode_id}` directory, $\$\{gitCode\_id\}$ represents developer's personal gitCode account.
<!--
   <img src="../figures/webIDE.png" alt="Cloud Platform" width="1000px" height="150px">
-->
### Method 2: Docker Deployment

For developers with Ascend devices, if you want to quickly set up Ascend environment, can use Docker image deployment.

> **Note**:
>
> - Image file is relatively large, download takes certain time, please wait patiently. For docker command option introduction, can query through `docker --help`.
> - Environment default installs latest version CANN package, pay attention to match with software when downloading source code.

1. **Install Driver (Runtime Dependency)**

   Driver is runtime dependency, if only compiling operators, can not install. Use `npu-smi info` to check if there is NPU related information. If not, please refer to "[CANN Quick Installation](https://www.hiascend.com/cann/download)" to complete driver installation:

   - Step 1: Select your product series, CPU architecture, operating system on the page, installation method select "Online Installation (Yum)".
   - Step 2: Follow page guide to complete "Configure User Group", "Install Dependencies & Configure Source", "Install NPU Driver" three steps.
   - Step 3: Execute `npu-smi info`, if NPU device information can be displayed normally, then driver installation is successful.

2. **Download Image**

   - Step 1: Log in to host machine as root user. Ensure Docker engine is installed on host machine (version 1.11.2 or above), use `docker --version` to check Docker version. If not, please refer to [Docker Official Installation Guide](https://docs.docker.com/engine/install/).
   - Step 2: Pull image with pre-integrated CANN software package and `ops-math` required dependencies from [Ascend Image Repository](https://www.hiascend.com/developer/ascendhub/detail/17da20d1c2b6493cb38765adeba85884).

      Example as follows, please replace CANN version number, chip series, operating system, python version and other information yourself. Supported values for each field can be queried on the above Ascend image repository page.

      ```bash
      # Taking cann:9.1.0-beta.1 version as example
      docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel
       ```

      > **Note**: Image tag format is `<CANN version>-<chip series>-<operating system>-<Python version>-devel`. Images with `-devel` suffix are operator development images, containing operator development compilation dependencies.

3. **Run Docker**

    After pulling image, need to start container with specific parameters so that container can access host's Ascend device.

    ```bash
    docker run --name cann_container --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -it swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel bash
    ```

    > **Note**: `--name` is used to specify container name, please replace with a custom name that is easy to identify; if this name is already occupied, container will fail to start.

    | Parameter | Description | Notes |
    | :--- | :--- | :--- |
    | `--name cann_container` | Specify name for container, easy to manage. | Can customize. |
    | `--device /dev/davinci0` | Core: Map host's NPU device card to container, can specify mapping multiple NPU device cards. | Must adjust according to actual situation: `davinci0` corresponds to the 0th NPU card in the system. Please first execute `npu-smi info` command on host, modify this number according to device number displayed in output (such as `NPU 0`, `NPU 1`). |
    | `--device /dev/davinci_manager` | Map NPU device management interface. | - |
    | `--device /dev/devmm_svm` | Map device memory management interface. | - |
    | `--device /dev/hisi_hdc` | Map communication interface between host and device. | - |
    | `-v /usr/local/dcmi:/usr/local/dcmi` | Mount device container management interface (DCMI) related tools and libraries. | - |
    | `-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi` | Mount `npu-smi` tool. | Enable container to directly run this command to query NPU status and performance information. |
    | `-v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/` | Key mount: Map host's NPU driver library to container. | - |
    | `-v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info` | Mount driver version information file. | - |
    | `-v /etc/ascend_install.info:/etc/ascend_install.info` | Mount CANN software installation information file. | - |
    | `-it` | Combination parameter of `-i` (interactive) and `-t` (allocate pseudo terminal). | - |
    | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12-devel` | Specify Docker image to run. | Please ensure this image name and tag (tag) are exactly the same as the image you pulled through `docker pull`. |
    | `bash` | Command executed immediately after container starts. | - |

### Method 3: Manual Installation

For developers with Ascend devices, if you want to manually set up Ascend environment, please refer to the following steps.

#### Install Software

- **Scenario 1: Experience master version capabilities or develop based on master version**

    1. **Install Driver (Runtime Dependency)**

        Driver is runtime dependency, if only compiling operators, can not install. Use `npu-smi info` to check if there is NPU related information. If not, please refer to "[CANN Quick Installation](https://www.hiascend.com/cann/download)" to complete driver installation:

        - Step 1: Select your product series, CPU architecture, operating system on the page, installation method select "Online Installation (Yum)".
        - Step 2: Follow page guide to complete "Configure User Group", "Install Dependencies & Configure Source", "Install NPU Driver" three steps.
        - Step 3: Execute `npu-smi info`, if NPU device information can be displayed normally, then driver installation is successful.

    2. **Install CANN Package**

        Please click [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/), select latest time version, and download corresponding package according to product model and environment architecture. Installation command as follows, for more guidance refer to "[CANN Quick Installation](https://www.hiascend.com/cann/download)".

        - Install CANN toolkit package

           ```bash
           bash ./Ascend-cann-toolkit_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
           ```

        - Install CANN ops package (runtime dependency)

            ops package is runtime dependency, if only compiling operators, can not install this package.

           ```bash
           bash ./Ascend-cann-${soc_name}-ops_${cann_version}_linux-${arch}.run --install --install-path=${install_path}
           ```

        Variable meaning description:

        - $\$\{cann\_version\}$: Represents CANN package version number.
        - $\$\{arch\}$: Represents CPU architecture, can query through `uname -m`, such as aarch64, x86_64.
        - $\$\{soc\_name\}$: Represents NPU model name.
        - $\$\{install\_path\}$: Represents specified installation path, ops package needs to be installed in the same path as toolkit package, root user default installed in `/usr/local/Ascend` directory.

- **Scenario 2: Experience released version capabilities or develop based on released version**

    Please refer to "[CANN Quick Installation](https://www.hiascend.com/cann/download)", select version (only supports CANN 8.5.0 and subsequent versions), and download corresponding package according to product series, CPU architecture, operating system, etc., finally refer to command provided on page to complete installation.

#### Install Basic Dependencies

This project basic dependencies are as follows, note to meet version number requirements.

- python >= 3.7.0 (recommended version <= 3.10)
- gcc/g++ >= 7.3.0
- cmake >= 3.16.0
- pigz (optional, can improve packaging speed after installation, recommended version >= 2.4)
- dos2unix
- make
- patch
- googletest (only depended when executing UT, recommended version [release-1.11.0](https://github.com/google/googletest/releases/tag/release-1.11.0))

Above dependencies can be installed one-click through project script, operation steps as follows:

1. Download source code.

    Download source code matching CANN version, command as follows, replace $\$\{tag\_version\}$ with branch tag name.

    ```bash
    git clone -b ${tag_version} https://gitcode.com/cann/ops-math.git
    ```

2. Install dependencies.

    First install above dependencies one-click through project root directory install_deps.sh, command as follows. If encounter unsupported system, please refer to this file to adapt yourself.

    ```bash
    bash install_deps.sh
    ```

    After installation, continue to install python third-party library dependencies through project root directory requirements.txt, command as follows.

    ```bash
    pip3 install -r requirements.txt
    ```

### Method 4: Spack Installation

Suitable for developers who want to quickly install dependencies and flexibly configure compilation options. Spack will automatically install CANN software package (toolkit and ops) and compilation dependencies, runtime mode still requires host machine to pre-install driver and firmware (if only compiling operators, can not install).

#### Pre-dependencies

Although Spack will automatically install related dependencies, Spack still needs some pre-dependencies to support its own operation:

- python >= 3.7.0 (recommended version <= 3.10)
- gcc/g++ >= 7.3.0
- patch
Can manually install yourself, or refer to [Install Basic Dependencies](./quick_install.md#install-basic-dependencies)

#### Quick Build

Enter project root directory, run one-click script:

```bash
cd ${local_repo_path}/ops-math # ${local_repo_path} is local repository path
source spack/prepare_cann_env.sh
spack install --add cann-ops-math@${cann_version} soc=${soc_name} # ops package is runtime dependency, if only compiling operators, can not install this package.
```

prepare_cann_env.sh will automatically install Spack, set up build environment, pull related dependencies and complete build. Build scenario default is compiling ops-math package. For other build scenarios, please refer to [compile.md](./compile.md#scenarios-using-spack)

#### Artifact Location

**View artifact location:** ```spack location -i cann-ops-math```
Artifact here refers to run package. Actually run package has been automatically installed to `$ASCEND_HOME_PATH`. Run package can also be found in build_out directory under code root directory.

#### More Spack Operations

For more refined use of Spack to control installation process, please refer to [spack_quick_install.md](./spack_quick_install.md)

## Environment Verification

After installing CANN package, need to verify whether environment and driver are normal.

- **Check NPU Device**

    ```bash
    # Run npu-smi, if device information can be displayed normally, then driver is normal
    npu-smi info
    ```

- **Check CANN Version**

    ```bash
    # View CANN toolkit package version information (default path installation)
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_toolkit_install.info # Docker deployment and manual installation scenarios
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_toolkit_install.info # CANNLab scenario
    spack location -i cann-toolkit # Spack installation scenario
    # View CANN ops package version information (default path installation)
    cat /usr/local/Ascend/cann/${arch}-linux/ascend_ops_install.info # Docker deployment and manual installation scenarios
    cat /home/developer/Ascend/cann/${arch}-linux/ascend_ops_install.info # CANNLab scenario
    spack location -i cann-ops # Spack installation scenario
    ```

    Where $\$\{arch\}$ can query current architecture through `uname -m`, such as aarch64, x86_64.

## Environment Variable Configuration

Choose appropriate command as needed to make environment variables effective. Spack scenario has automatically set environment variables, skip this step

```bash
# Default path installation, taking root user as example (non-root user, replace /usr/local with ${HOME})
source /usr/local/Ascend/cann/set_env.sh
# Specified path installation
# source ${install_path}/cann/set_env.sh
```
