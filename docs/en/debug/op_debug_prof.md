# Operator Debugging and Tuning

## Debugging and Positioning (AI Core Operator)

During operator execution, if operator execution failure, accuracy abnormality and other problems occur, you can print information at each stage, such as Kernel intermediate results, for problem analysis and positioning.

### 1. Host-side Log Acquisition Method

* **plog acquisition**

   After program execution ends, by default you can view in "$HOME/ascendc/log". The host log file storage path is as follows:

   ```bash
   $HOME/ascend/log/debug/plog/plog-pid_*.log
   ```

   Enable environment variable ASCEND_SLOG_PRINT_TO_STDOUT to display log directly on screen (1: enable screen printing, 0: disable screen printing). Configuration example:

   ```bash
   export ASCEND_SLOG_PRINT_TO_STDOUT=1
   ```

   For log related introduction, refer to [Log Reference](https://hiascend.com/document/redirect/CannCommunitylogref). For environment variable introduction, refer to [Environment Variable Reference](https://hiascend.com/document/redirect/CannCommunityEnvRef).

* **aclnn exception error information acquisition**

   Obtain exception information during aclnn interface call through aclGetRecentErrMsg interface ([Runtime API](https://hiascend.com/document/redirect/CannCommunityRuntimeApi)). Usage:

   ```bash
   printf(aclGetRecentErrMsg());
   ```

   Print error information example:

   ```bash
   [PID:646612] 2026-01-24-11:53:44.671.727 AclNN_Parameter_Error(EZ1001): Expected a proper Tensor but got null for argument addmmTensor.self.
   ```

### 2. Kernel Debugging

Common debugging methods:

* **printf**

  This interface supports printing Scalar type data, such as integers, characters, booleans, etc. For detailed introduction, refer to [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi) "Operator Debugging API > printf".

  ```c++
  blockLength_ = tilingData->totalLength / AscendC::GetBlockNum();
  tileNum_ = tilingData->tileNum;
  tileLength_ = blockLength_ / tileNum_ / BUFFER_NUM;
  // Print current core calculation Block length
  AscendC::PRINTF("Tiling blockLength is %llu\n", blockLength_);
  ```

* **DumpTensor**

  This interface supports Dump specified Tensor content, and also supports printing custom additional information, such as current line number, etc. For detailed introduction, refer to [Ascend C API](https://hiascend.com/document/redirect/CannCommunityAscendCApi) "Operator Debugging API > DumpTensor".

  ```c++
  AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
  // Print zLocal Tensor information
  DumpTensor(zLocal, 0, 128);
  AscendC::DataCopy(outputGMZ[progress * tileLength_], zLocal, tileLength_);
  ```

For complex scenario problem positioning, such as operator deadlock, GM/UB access out of bounds, etc., you can use **single-step debugging** method. For specific operations, refer to [msDebug](https://www.hiascend.com/document/redirect/CannCommunityToolMsdebug) operator debugging tool.

## Debugging and Positioning (AI CPU Operator)

During operator execution, if operator execution failure, accuracy abnormality and other problems occur, you can print information at each stage, such as Kernel intermediate results, for problem analysis and positioning.

### 1. Host-side Log Acquisition Method

   See AI Core operator [Host-side Log Acquisition Method](#1-host-side-log-acquisition-method)

### 2. Kernel Debugging

Common debugging methods:

* **KERNEL_LOG macro**

  You can print log information during operator execution through the following macro, including DEBUG, INFO, WARN, ERROR level logs.

  ```Cpp
  KERNEL_LOG_DEBUG(fmt, …)      // fmt parameter represents format control string
  KERNEL_LOG_INFO(fmt, …)
  KERNEL_LOG_WARN(fmt, …)
  KERNEL_LOG_ERROR(fmt, …)      // Default print ERROR level log
  ```

  If you need to print non-ERROR level logs, you need to configure environment variable `ASCEND_GLOBAL_LOG_LEVEL` in advance. For specific usage, refer to [Environment Variable Reference](https://hiascend.com/document/redirect/CannCommunityEnvRef).

  Print example:

  ```c++
  Tensor* input0 = ctx.Input(kFirstInputIndex);
  Tensor* input1 = ctx.Input(kSecondInputIndex);
  Tensor* output = ctx.Output(0);

  if (input0 == nullptr || input1 == nullptr || output == nullptr) {
    // Print error information
    KERNEL_LOG_ERROR("Invalid argument");
    return kParamInvalid;
  }

  int64_t num_elements = input0->NumElements();
  // Print input element count
  KERNEL_LOG_INFO("Num of elements is %ld", data_size);
  ```

## Performance Tuning

During operator execution, if execution performance degradation, memory usage abnormality and other problems occur, you can analyze operator execution stage indicator data (such as throughput, memory usage, latency, etc.) through [msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof) performance analysis tool to determine the root cause of the problem and optimize accordingly.

This chapter uses `AddExample` custom operator as an example to introduce the commonly used on-board performance collection and pipeline simulation methods in operator tuning.

**Applicable Scenario Differences:**

* **On-board performance collection**: Suitable for running operators on real NPU hardware, quickly obtaining overall operator performance indicators (such as Kernel latency, Block count, pipeline ratio, etc.), helping to judge whether the operator has performance issues.
* **Pipeline simulation**: Suitable for developers without NPU hardware, or scenarios that need to deeply analyze operator internal instruction-level pipeline bottlenecks and optimize instruction arrangement, providing more detailed instruction-level pipeline analysis than on-board.

### Method 1: On-board Performance Collection

* **Prerequisites**

      After completing operator development and compilation, assuming aclnn interface method is used for invocation, the generated operator executable file (test_aclnn_add_example) is located in the project `examples/add_example/examples/build/bin/`.

* **Collect Performance Data**

      When you need to collect various pipeline indicators of operator on-board execution, you can enter the operator executable file directory and execute the following command:

      ```bash
      msprof op ./test_aclnn_add_example
      ```

      The collection result is in the project `examples/add_example/examples/build/bin/OPPROF_*` directory. After collection, the following information is printed:

      ``` text
      Op Name: AddExample_a1532827238e1555db7b997c7bce2928_high_performance_1
      Op Type: vector
      Task Duration(us): 97.861954
      Block Dim: 8
      Mix Block Dim:
      Device Id: 0
      Pid: 2776181
      Current Freq: 1800
      Rated Freq: 1800
      ```

      Where Task Duration is the current operator Kernel latency, Block Dim is the current operator execution core count.

      For detailed indicators of various operator pipelines, refer to `ArithmeticUtilization` file under `OPPROF_*`, which contains the ratio of various current pipelines. For specific introduction, refer to [msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof) "Performance Data File > msprof op > ArithmeticUtilization (cube and vector type instruction latency and ratio)" section.

### Method 2: Simulation Pipeline Collection

* **Prerequisites**

      After completing operator development and compilation, assuming aclnn interface method is used for invocation, the generated operator executable file (test_aclnn_add_example) is located in the project `examples/add_example/examples/build/bin/`.

* **For Ascend 950PR, use [CANN Simulator](./cann_sim.md) simulation tool, execute simulation command, generate simulation data**

      Execute simulation command, generate simulation data

      ```
      cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
      ```

      The simulation result is in the project `examples/add_example/examples/build/bin/cannsim_*/report/results/kernel_*/core_*` directory. The pipeline related files are:

      ```
      trace_core0.json
      ```

      Enter "chrome://tracing" address in Chrome browser and drag the generated instruction pipeline file (trace_core0.json) to the blank area to open. For specific parameter introduction, refer to CANN Simulator ["Simulation Result Analysis"](./cann_sim.md#仿真结果解析) section.

* **For Atlas A2/A3 series products, use [msProf](https://www.hiascend.com/document/redirect/CannCommunityToolMsprof) tool, execute simulation command, generate simulation data**

      Before using msProf tool for operator simulation tuning, execute the following command to configure environment variables.

      ```bash
      export LD_LIBRARY_PATH=${INSTALL_DIR}/tools/simulator/Ascendxxxyy/lib:$LD_LIBRARY_PATH
      ```

      Please modify the above environment variables according to the actual CANN software package installation path and AI processor model.

      Then enter the operator executable file directory and execute the following command:

      ```bash
      msprof op simulator --output=$PWD/pipeline_auto --kernel-name="AddExample" ./test_aclnn_add_example
      ```

      The collection result is in the project `$PWD/pipeline_auto/OPPROF_**` directory.
      The pipeline related file path is `OPPROF**/simulator/visualize_data.bin`. You can use [MindStudio Insight](https://www.hiascend.com/document/redirect/MindStudioInsight) tool "Basic Operations > Import Data" section to view how to import pipeline data.
