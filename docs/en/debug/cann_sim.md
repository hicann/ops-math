# Introduction

CANN Simulator is a SoC-level chip simulation tool designed for operator development scenarios, used to analyze accuracy and performance data (such as instruction execution status, etc.) of AI tasks running on AI simulators at various stages. This tool helps users perform deep performance tuning, enabling R&D personnel to obtain verification results and performance feedback almost identical to real chips even when chip resources are unavailable or scarce.

# Main Functions

This tool maintains binary compatibility with on-board execution (the same kernel can execute on both simulation and AI processors), with main uses as follows:

* Accuracy Simulation: Outputs bit-level accuracy results to assist users in completing operator accuracy verification.
* Performance Simulation: Outputs instruction pipeline diagrams to assist users in locating operator performance bottleneck issues.

# Preparation Before Use

## Usage Constraints

* Tool recommended environment configuration: CPU 16 cores, memory 32GB or above.
* All example paths in this document need to ensure the running user has read or read-write permissions.
* For security and minimum privilege considerations, it is recommended to execute this tool with regular user privileges, avoid using root or other high-privilege accounts.
* This tool depends on CANN software package. Before use, please install CANN software package first. No need to install driver and firmware. Execute CANN's set_env.sh environment variable file through source command. For security, please do not modify environment variables involved in set_env.sh after executing source command.
* Users should follow the principle of minimum privilege. For example, files input to the tool require that other users cannot write. In some function scenarios with stricter security requirements, it is also necessary to ensure that input files cannot be written by group users.
* This tool is a development tool and is not recommended for use in production environments.
* The simulation function of the tool only supports single-card scenarios and cannot simulate multi-card environments. Only 0 card can be set in the code. Modifying the visible card number will cause simulation failure.
* The simulation environment only supports AI Core compute operators (does not support MC2 and HCCL type operators).
* CANN Simulator tool is currently in the preview version stage, only supports Ascend950PR chip. It is recommended that the simulator running environment be configured with 16-core CPU and 32GB or above memory.
* Currently does not support arm environment simulation.

## Environment Preparation

CANN Simulator is integrated in the CANN toolkit package. Refer to [Environment Deployment](../install/quick_install.md) to complete software package installation.

# Quick Start

Below uses [add_examples](../../../examples/add_example/) as an example to explain operator simulation in detail.

## Operator Compilation

* Refer to [Operator Invocation](../invocation/quick_op_invocation.md) to complete add_example operator compilation and installation.

```bash
# Note: Enter the project root directory and execute the following compilation command. The command is for reference only. For details, see the operator invocation instructions.
bash build.sh --pkg --soc=Ascend950 --vendor_name=custom --ops=add_example
# Install custom operator package
./build_out/cann-ops-math-${vendor_name}_linux-${arch}.run
```

* Refer to [aclnn Invocation](../invocation/quick_op_invocation.md) to complete test_aclnn_add_example.cpp compilation and generate executable file test_aclnn_add_example.

## Execute Simulation Command

```bash
cannsim record ./test_aclnn_add_example -s Ascend950 --gen-report
```

The simulation tool execution log file is in the examples/add_example/examples/build/bin/cannsim_* directory, and the execution log file is cannsim.log.

From the simulation tool log file, you can see the print information in the example:

```bash
add_example first input[0] is: 1.000000, second input[0] is: 1.000000, result[0] is: 2.000000
add_example first input[1] is: 1.000000, second input[1] is: 1.000000, result[1] is: 2.000000
add_example first input[2] is: 1.000000, second input[2] is: 1.000000, result[2] is: 2.000000
add_example first input[3] is: 1.000000, second input[3] is: 1.000000, result[3] is: 2.000000
add_example first input[4] is: 1.000000, second input[4] is: 1.000000, result[4] is: 2.000000
add_example first input[5] is: 1.000000, second input[5] is: 1.000000, result[5] is: 2.000000
add_example first input[6] is: 1.000000, second input[6] is: 1.000000, result[6] is: 2.000000
```

## View Performance Pipeline

The simulation performance pipeline file is in the project `examples/add_example/examples/build/bin/cannsim_*/report/results/kernel_*/core_*` directory. The pipeline related files are:

```bash
trace_core0.json
```

Enter "chrome://tracing" address in Chrome browser and drag the generated instruction pipeline file (trace_core0.json) to the blank area to open. For specific parameter introduction, refer to the "Simulation Result Analysis" section.

# Simulation Execution Instructions

## Command Function

Execute applications in simulation environment.

## Command Format

cannsim record [options] user_app

## Parameter Description

Table 1 Simulation Execution Parameter Description

| Parameter | Optional/Required | Description |
| --- | --- | --- |
|-s or --soc-version | Required | Specify the simulation target chip version (such as: Ascend950).|
|-o or --output | Optional | The path where generated files are located, can be configured as absolute path or relative path, and the user executing the tool needs to have read-write permission. If no path is specified, data is saved in the current directory by default.|
|-g or --gen-report | Optional | Enable whether to perform automatic analysis after simulation completion and generate analysis report. Default is no automatic analysis.|
|-u or --user-option | Optional | User-defined operator parameters, passed to operator program in command line option form.|
|-n or --core-id | Optional | AI Core that enables logging during simulation, format same as report -n: 'all', '0-2,12-14', '5'. Default all on; when used with -g and not specified, falls back to core 0.|
|user_app|Required|Operator program or command to run (such as ./app, python train.py, bash run.sh).|

## Usage Example

1. Complete operator development and compilation.
2. Execute simulation command, can refer to the following usage examples.

    ```bash
    Method 1: Enable simulation and save output to ./output directory, /path/to/app is operator program
    $ cannsim record /path/to/app -o ./output -s Ascend950

    Method 2: Enable simulation and generate report for subsequent performance analysis
    $ cannsim record /path/to/app -o ./output -s Ascend950 --gen-report
    ```

3. After command completion, a folder named "cannsim_{timestamp}_${user_app}" will be generated in the default path or specified "output" directory. The structure example is as follows:

    ```text
    ├─cannsim_{timestamp}_${user_app}
    ├── cannsim.log
    ```

4. Users can obtain operator execution results and perform accuracy comparison. Results are displayed in cannsim.log. Example as follows:

    The following output is only an example of Ascend C single operator direct invocation accuracy comparison result. Due to version differences, please refer to actual output.

    ```text
    INFO:root:[INFO] compare data case[ case001]
    INFO:root:---------------RESULT---------------
    INFO:root:['case_name', 'wrong_num', 'total_num', 'result', 'task_duration']
    INFO:root:[' case001', 0, 65536, 'Success']
    ```

5. View operator instruction pipeline diagram, refer to simulation result analysis.

# Simulation Result Analysis Instructions

## Command Function

Generate visualized instruction pipeline diagram.

## Command Format

cannsim report [options]

## Parameter Description

Table 1 Simulation Result Analysis Parameter Description

| Parameter | Optional/Required | Description |
| --- | --- | --- |
|-e or --export | Required | Simulation execution result directory, specified to cannsim_{timestamp}_${user_app} level, can be configured as absolute path or relative path, and the executing user needs to have read-write permission.|
|-o or --output | Optional | Instruction pipeline diagram output directory, can be configured as absolute path or relative path, and the executing user needs to have read-write permission. If no path is specified, default is same as export directory.|
|-n or --core-id | Optional | Specify the core ID for generating instruction pipeline, supported formats: 'all', '0-2,12-14', '5'. If not specified, default generates instruction pipeline for core 0.|
|-f or --object-file | Optional | Device object file path, used to assist in generating reports.|

## Usage Example

1. Refer to simulation execution operator simulation, compare output examples, ensure corresponding results execute correctly.
2. Execute simulation result analysis command, can refer to the following execution use cases.

    ```bash
    Generate performance analysis report in current directory (default only analyze core 0)
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app}

    Generate performance analysis reports for core 0, core 1, core 11, core 12 in specified directory
    cannsim report -e /path/to/cannsim_{timestamp}_${user_app} -o /path/to/report -n '0-1, 11-12'
    ```

3. After command execution, corresponding pipeline files will be generated in the directory configured by output. The file format is json format. Output result example is as follows:

    ```text
    trace_core0.json
    trace_core1.json
    ...
    ```

4. Simulation result viewing
    Enter "chrome://tracing" address in Chrome browser and drag the generated instruction pipeline file (trace.json) to the blank area to open. Press keyboard shortcuts (W: zoom in, S: zoom out, A: move left, D: move right) to view.
    <!--
    ![Instruction Pipeline Diagram](../figures/指令流水图.png)
    -->
    Table 2 Key Field Description

    | Field Name | Field Meaning |
    | --- | --- |
    |VECTOR|Vector computation unit.|
    |SCALAR|Scalar computation unit.|
    |Cube|Matrix multiplication computation unit.|
    |MTE1|Data transfer pipeline, data transfer direction is: L1 ->{L0A/L0B, UBUF}.|
    |MTE2|Data transfer pipeline, data transfer direction is: {DDR/GM, L2} ->{L1, L0A/B, UBUF}.|
    |MTE3|Data transfer pipeline, data transfer direction is: UBUF -> {DDR/GM, L2, L1}、L1->{DDR/L2}.|
    |FIXP|Data transfer pipeline, data transfer direction is: FIXPIPE L0C -> OUT/L1.|
    |FLOWCTRL|Control flow instruction.|
    |ICACHELOAD|View missed ICache.|

# Query Help Information

## Command Function

Query tool help information.

## Command Format

Query tool help information:

```bash
cannsim --help
```

Query tool record subcommand help information:

```bash
cannsim record --help
```

Query tool report subcommand help information:

 ```bash
 cannsim report --help
 ```

## Parameter Description

None

## Usage Example

1. Log in to Host side server.
2. Execute the following command.

    ```bash
    cannsim --help
    ```

## Output Description

```bash
usage: cannsim [-h] {record,report} ...

Command-line tool for performance simulation analysis on Ascend hardware.

positional arguments:
  {record,report}  Available commands
    record         Run user application in AscendOps simulation environment
    report         Generate performance analysis reports

options:
  -h, --help       show this help message and exit
```
