# Operator Invocation

## Usage Instructions

- Prerequisite: Before operator invocation, please refer to this project README to complete environment preparation and source code download. This will not be repeated here.

- Invocation Scope: Built-in operator list available for invocation see [Operator List](../../zh//op_list.md), also supports invoking custom operators (such as contributed operators under experimental directory).

- Invocation Scenario: Please choose appropriate operator invocation scheme according to actual scenario requirements.

   | Invocation Scenario | Scenario Description | Features |
   |--------|----|-----|
   |[Quick Operator Invocation](#quick-operator-invocation)|Suitable for scenarios to quickly experience and verify operator functionality.|**No need to build invocation project**, can directly invoke operator examples based on source code compilation package and project script build.sh.|
   |[Business Application Integrated Operator](#business-application-integrated-operator)|Suitable for flexibly integrating operators into actual business applications.|**Need to build invocation project yourself**, manually create invocation script/CMake project, flexibly implement operator compilation and running.|

- Invocation Method: Currently mainly provides the following operator invocation methods, please choose as needed.

  | Invocation Method | Description |
  |--------|----|
  |[PyTorch API](#pytorch-api)|Register operator Kernel to PyTorch native framework, implement operator invocation in a way similar to Torch native API.|
  |[aclnn API](#aclnn-api)|Provide corresponding C language API for operator (prefix aclnn), no need to provide IR definition, implement direct API operator invocation.|
  |[GE Graph Mode](#ge-graph-mode)|Through operator IR (Intermediate Representation) definition, implement operator invocation in graph composition way.|

## Quick Operator Invocation

**If you need to quickly experience or verify existing operator functionality in the project, can refer to this chapter's simplest invocation method, execute operator examples through build.sh**.

This method features no need to build invocation project (i.e., create compilation/running scripts, etc.), simple and easy to operate.

> **Note**: For Ascend 950PR products, can execute operator examples through Simulator simulation tool, see [Simulation Guide](../debug/op_debug_prof.md#method-2-simulation-pipeline-collection).

**Step 1**: Environment Preparation. Before invoking operator, please first ensure your environment has installed CANN-toolkit package and compiled operator package.

**Step 2**: Execute existing operator examples in the project.

- Execute operator examples based on **custom operator package**, command as follows:

    ```bash
    bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}] [--soc=${soc_version}] [--experimental]
    # Taking Abs operator example execution as example
    # bash build.sh --run_example abs eager cust --vendor_name=custom
    # Taking Abs operator experimental execution as example
    # bash build.sh --experimental --run_example abs eager cust --vendor_name=custom
    ```

  - $\$\{op\}$: Represents operator to execute, operator name lowercase underscore form, such as abs.
  - $\$\{mode\}$: Represents invocation method, currently supports eager (aclnn invocation), graph (graph mode invocation).
  - $\$\{pkg_mode\}$: Represents package mode, currently only supports cust, i.e., custom operator package.
  - $\$\{vendor\_name\}$ (optional): Consistent with built custom operator package setting, default name is custom.
  - $\$\{soc_version\}$ (optional): Represents NPU model.
  - $\$\{experimental\}$ (optional): Represents executing operators saved by user in experimental contribution directory.

    Note: When $\$\{mode\}$ is graph, do not specify $\$\{pkg_mode\}$ and $\$\{vendor\_name\}$

- Execute operator examples based on **ops-math package**, command as follows:

    ```bash
    bash build.sh --run_example ${op} ${mode} [--soc=${soc_version}]
    # Taking Abs operator example execution as example
    # bash build.sh --run_example abs eager
    ```

  - $\$\{op\}$: Represents operator to execute, operator name lowercase underscore form, such as abs.
  - $\$\{mode\}$: Represents operator execution mode, currently supports eager (aclnn invocation), graph (graph mode invocation).
  - $\$\{soc_version\}$ (optional): Represents NPU model.

- Execute operator examples based on **ops-math static library**:

    1. **Prerequisites**

        ops-math static library depends on ops-legacy static library, prepare above static libraries, extract and move all lib64, include directories to unified directory $\$\{static\_lib\_path\}$.

        > Note: ops-legacy static library ```cann-${soc_name}-ops-legacy-static_${cann_version}_linux-${arch}.tar.gz``` can be obtained by clicking [Download Link](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master), ops-math static library not yet provided as software package, please generate through local compilation.

    2. **Create run.sh**

        Create run.sh file in the same directory as operator to execute `examples/test_aclnn_${op_name}.cpp`.

        Taking Abs operator executing test_aclnn_abs.cpp as example, example as follows:

        ```bash
        # Static library file path
        static_lib_path=""

        # Environment variable takes effect
        if [ -n "$ASCEND_INSTALL_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
        elif [ -n "$ASCEND_HOME_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
        else
            _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
        fi

        source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

        # Compile executable file
        g++ test_aclnn_abs.cpp \
        -I ${static_lib_path}/include \
        -L ${static_lib_path}/lib64 \
        -I ${_ASCEND_INSTALL_PATH}/include \
        -I ${_ASCEND_INSTALL_PATH}/include/aclnnop \
        -L ${_ASCEND_INSTALL_PATH}/lib64 \
        -Wl,--allow-multiple-definition \
        -Wl,--start-group -lcann_math_static -lcann_legacy_static -Wl,--end-group -lgraph -lgraph_base \
        -lpthread -lmmpa -lmetadef -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform \
        -ldl -lc_sec -lnnopbase -lruntime -lerror_manager -lunified_dlog \
        -o test_aclnn_abs   # Replace with actual operator executable file name

        # Execute program
        ./test_aclnn_abs
        ```

        $\$\{static\_lib\_path\}$ represents static library unified placement path; $\$\{ASCEND\_INSTALL\_PATH\}$ has been configured through environment variable, represents CANN toolkit package installation path; final executable file name please replace with **actual operator executable file name**.

        Where lcann\_math\_static, lcann\_legacy\_static represent static library files depended by operator, obtained from static library unified placement path $\$\{static\_lib\_path\}$;
        lgraph, lmetadef etc. represent underlying library files depended by operator, can be obtained in CANN toolkit package.

    3. **Execute run.sh**

        ```bash
        bash run.sh
        ```

**Step 3**: Check Execution Result.

Operator example will print result after execution, taking Abs operator result as example:

  ```text
  abs result[0] is: 1.000000
  abs result[1] is: 1.000000
  abs result[2] is: 1.000000
  abs result[3] is: 2.000000
  abs result[4] is: 2.000000
  abs result[5] is: 2.000000
  abs result[6] is: 3.000000
  abs result[7] is: 3.000000
  ```

## Business Application Integrated Operator

**If you need to integrate operators into actual business applications, can refer to this chapter to build invocation project yourself. Through custom invocation script/CMake project, etc., implement operator compilation and running**.

This method features manually building invocation project, high scenario flexibility, strong portability.

Different invocation methods correspond to different compilation projects. Currently supports PyTorch API, aclnn API, GE graph mode invocation methods. Invocation methods and practices please refer to below, please choose as needed.

### PyTorch API

This method registers operator Kernel to PyTorch native framework, making it can be directly invoked like native Torch API.

For specific invocation principle and process, please refer to [examples/fast_kernel_launch_example](../../../examples/fast_kernel_launch_example/README.md).

### aclnn API

#### Invocation Process

To facilitate operator invocation, Host side provides operator corresponding C language API (i.e., API with aclnn prefix) to implement operator invocation, no need to provide operator IR (Intermediate Representation) definition. aclnn API invocation process is as follows:
<!--
![Schematic Diagram](../figures/aclnn调用.png)
-->
#### Compilation and Running

> Note: During operation, if log prompts to set environment variable, please operate according to prompt.

1. Environment Preparation. Before compilation and running, please first ensure your environment has installed CANN-toolkit package and compiled operator package.

2. Create Invocation Script.

    In any directory of the environment, create new invocation cpp script, name can be customized (such as `${test_aclnn_op_name}.cpp`).

   For easy understanding, taking `AddExample` operator as example, invocation script as follows, for reference only, full code see [test_aclnn_add_example.cpp](../../../examples/add_example/examples/test_aclnn_add_example.cpp).

   ```Cpp
   int main()
   {
       // 1.Call acl for device/stream initialization
       int32_t deviceId = 0;
       aclrtStream stream;
       auto ret = Init(deviceId, &stream);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

       // 2.Construct input and output, need to customize construction according to API interface
       aclTensor* selfX = nullptr;
       void* selfXDeviceAddr = nullptr;
       std::vector<int64_t> selfXShape = {32, 4, 4, 4};
       std::vector<float> selfXHostData(2048, 1);
       ret = CreateAclTensor(selfXHostData, selfXShape, &selfXDeviceAddr, aclDataType::ACL_FLOAT, &selfX);
       CHECK_RET(ret == ACL_SUCCESS, return ret);

       aclTensor* selfY = nullptr;
       void* selfYDeviceAddr = nullptr;
       std::vector<int64_t> selfYShape = {32, 4, 4, 4};
       std::vector<float> selfYHostData(2048, 1);
       ret = CreateAclTensor(selfYHostData, selfYShape, &selfYDeviceAddr, aclDataType::ACL_FLOAT, &selfY);
       CHECK_RET(ret == ACL_SUCCESS, return ret);

       aclTensor* out = nullptr;
       void* outDeviceAddr = nullptr;
       std::vector<int64_t> outShape = {32, 4, 4, 4};
       std::vector<float> outHostData(2048, 1);
       ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
       CHECK_RET(ret == ACL_SUCCESS, return ret);

       // 3.Call CANN operator library API, need to modify to specific Api name
       uint64_t workspaceSize = 0;
       aclOpExecutor* executor;

       // 4.Call aclnnAddExample first stage interface
       ret = aclnnAddExampleGetWorkspaceSize(selfX, selfY, out, &workspaceSize, &executor);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExampleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

       // Allocate device memory according to workspaceSize calculated by first stage interface
       void* workspaceAddr = nullptr;
       if (workspaceSize > static_cast<uint64_t>(0)) {
           ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
           CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
       }

       // 5.Call aclnnAddExample second stage interface
       ret = aclnnAddExample(workspaceAddr, workspaceSize, executor, stream);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddExample failed. ERROR: %d\n", ret); return ret);

       // 6.(Fixed writing) Synchronize waiting for task execution to complete
       ret = aclrtSynchronizeStream(stream);
       CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

       // 7.Get output value, copy result from device side memory to host side, need to modify according to specific API interface definition
       PrintOutResult(outShape, &outDeviceAddr);

       // 8.Release aclTensor, need to modify according to specific API interface definition
       aclDestroyTensor(selfX);
       aclDestroyTensor(selfY);
       aclDestroyTensor(out);

       // 9.Release device resources
       aclrtFree(selfXDeviceAddr);
       aclrtFree(selfYDeviceAddr);
       aclrtFree(outDeviceAddr);
       if (workspaceSize > static_cast<uint64_t>(0)) {
           aclrtFree(workspaceAddr);
       }
       aclrtDestroyStream(stream);
       aclrtResetDevice(deviceId);

       // 10. acl de-initialization
       aclFinalize();
       return 0;
   }
   ```

3. Create CMakeLists.txt file.

    Create CMakeLists.txt file in the same directory as `${test_aclnn_op_name}.cpp`. Note that compilation scripts differ when invoking custom operators (such as experimental directory) and standard project operators (built-in operators). Example as follows, for reference only, please modify according to actual situation.

    - **Invoke Custom Operator**: Depends on custom operator package

        ```bash
        cmake_minimum_required(VERSION 3.14)
        # Set project name
        project(ACLNN_EXAMPLE)

        # Set C++ compilation standard
        add_compile_options(-std=c++11)

        # Set compilation output directory to bin folder under current directory
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

        # Set compilation options for debug and release modes
        set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
        set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

        # Add executable file (Custom: replace with actual invoked operator *.cpp file)
        add_executable(${test_aclnn_op_name}
        ${test_aclnn_op_name}.cpp)

        # ASCEND_PATH (If CANN package path is incorrect, please modify according to actual path)
        if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
            set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
        else()
            set(ASCEND_PATH "/usr/local/Ascend/cann")
        endif()

        # Get custom operator package name, when multiple custom operator packages exist, only one will be used
        set(VENDORS_DIR "${ASCEND_PATH}/opp/vendors")
        file(GLOB CUSTOM_DIRS "${VENDORS_DIR}/*")
        foreach(CUSTOM_DIR ${CUSTOM_DIRS})
            if(IS_DIRECTORY ${CUSTOM_DIR})
                set(TARGET_SUBDIR ${CUSTOM_DIR})
            endif()
        endforeach()

        if(NOT DEFINED TARGET_SUBDIR)
            message(FATAL_ERROR "Custom operator package not found in path ${ASCEND_PATH}")
        endif()

        # Set header file path
        set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
        include_directories(
            ${INCLUDE_BASE_DIR}
            ${TARGET_SUBDIR}/op_api/include
        )
        include_directories(
            ${INCLUDE_BASE_DIR}
        )

        # Link required dynamic libraries (Custom: replace with actual operator executable file)
        target_link_libraries(${test_aclnn_op_name} PRIVATE
            ${ASCEND_PATH}/lib64/libascendcl.so
            ${ASCEND_PATH}/lib64/libnnopbase.so
            ${TARGET_SUBDIR}/op_api/lib/libcust_opapi.so      # Link custom operator library file
        )
        target_link_options(${test_aclnn_op_name} PRIVATE
            "-Wl,-rpath,${TARGET_SUBDIR}/op_api/lib"
        )

        # Install target file to bin directory (Custom: replace with actual operator executable file)
        install(TARGETS ${test_aclnn_op_name} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        ```

    - **Invoke Standard Operator (Built-in Operator)**: Depends on op-math full package

        ```bash
        cmake_minimum_required(VERSION 3.14)
        # Set project name
        project(ACLNN_EXAMPLE)

        # Set C++ compilation standard
        add_compile_options(-std=c++11)

        # Set compilation output directory to bin folder under current directory
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")

        # Set compilation options for debug and release modes
        set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
        set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

        # Add executable file (Custom: replace with actual invoked operator *.cpp file)
        add_executable(${test_aclnn_op_name}
        ${test_aclnn_op_name}.cpp)

        # ASCEND_PATH (If CANN package path is incorrect, please modify according to actual path)
        if(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
            set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
        else()
            set(ASCEND_PATH "/usr/local/Ascend/cann")
        endif()

        # Set header file path
        set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
        include_directories(
            ${INCLUDE_BASE_DIR}
            ${ASCEND_PATH}/include/aclnnop
        )

        # Link required dynamic libraries (Custom: replace with actual operator executable file)
        target_link_libraries(${test_aclnn_op_name} PRIVATE
            ${ASCEND_PATH}/lib64/libascendcl.so
            ${ASCEND_PATH}/lib64/libnnopbase.so
            ${ASCEND_PATH}/lib64/libopapi_math.so            # Link built-in operator library file
        )

        # Install target file to bin directory (Custom: replace with actual operator executable file)
        install(TARGETS ${test_aclnn_op_name} DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
        ```

4. Create run.sh file.

    Create run.sh file in the same directory as `${test_aclnn_op_name}.cpp`. Taking `AddExample` operator as example, example as follows, please modify according to actual situation.

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                     # Actual CANN package installation path
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

    rm -rf build
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # Execute build command
    make
    cd bin
    ./${test_aclnn_op_name}                                     # Replace with actual operator executable file name
    ```

5. Run run.sh file.
   Execute the following command in the path where run.sh file is located:

   ```bash
   bash run.sh
   ```

    Default generates executable file ${test_aclnn_op_name} in current execution path `/build/bin`. Running result taking test\_aclnn\_add\_ example as example:

   ```bash
   mean result[2046] is 2.000000
   mean result[2047] is 2.000000
   ```

### GE Graph Mode

#### Invocation Process

This method invokes operator in graph composition way based on operator GE IR (Intermediate Representation) definition. Invocation process is as follows:
<!--
![Schematic Diagram](../figures/IR调用.png)
-->
#### Compilation and Running

> Note: During operation, if log prompts to set environment variable, please operate according to prompt.

1. Environment Preparation. Before compilation and running, please first ensure your environment has installed CANN-toolkit package and compiled operator package.

2. Create Invocation Script.

   In target operator `examples` directory, create new invocation script test\_geir\_\$\{op\_name\}.cpp, $\$\{op\_name\}$ represents target operator name. Taking `AddExample` operator as example, invocation script as follows, for reference only, full code see [test_geir_add_example.cpp](../../../examples/add_example/examples/test_geir_add_example.cpp).

   ```CPP
   int main() {
       // 1.Create graph object
       Graph graph(graphName);

       // 2.Graph global compilation option initialization
       Status ret = ge::GEInitialize(globalOptions);

       // 3.Create AddExample operator instance
       auto add1 = op::AddExample("add1");

       // 4.Define graph input output vectors
       std::vector<Operator> inputs{};
       std::vector<Operator> outputs{};

       // 5.Prepare input data
       std::vector<int64_t> xShape = {32,4,4,4};
       // Macro expansion way to handle variable assignment
       ADD_INPUT(1, x1, inDtype, xShape);
       ADD_INPUT(2, x2, inDtype, xShape);
       ADD_OUTPUT(1, y, inDtype, xShape);

       outputs.push_back(add1);

       // 6.Set graph object's input operator and output operator
       graph.SetInputs(inputs).SetOutputs(outputs);

       // 7.Create session object
       ge::Session* session = new Session(buildOptions);

       // 8. session add graph
       ret = session->AddGraph(graphId, graph, graphOptions);

       // 9.Run graph
       ret = session->RunGraph(graphId, input, output);

       // 10.Release resources
       GEFinalize();

       return 0;
   }
   ```

3. Create CMakeLists.txt file.

   Create CMakeLists.txt file in the same directory as test\_geir\_\$\{op\_name\}.cpp. Taking `AddExample` operator as example, example as follows, please modify according to actual situation.

    ```bash
   cmake_minimum_required(VERSION 3.14)

   # Set project name
   project(GE_IR_EXAMPLE)

   if(NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
       get_filename_component(ASCEND_PATH $ENV{ASCEND_OPP_PATH} DIRECTORY)
   elseif(NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
       set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
   else()
       set(ASCEND_PATH "/usr/local/Ascend/cann")
   endif()

   set(FWK_INCLUDE_DIR "${ASCEND_PATH}/compiler/include")

   message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

   file(GLOB files CONFIGURE_DEPENDS
        test_geir_add_example.cpp
   )

   # Add executable file (Please replace with actual operator executable file)
   add_executable(test_geir_add_example ${files})

   find_library(GRAPH_LIBRARY_DIR libgraph.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GE_RUNNER_LIBRARY_DIR libge_runner.so "${ASCEND_PATH}/compiler/lib64/stub")
   find_library(GRAPH_BASE_LIBRARY_DIR libgraph_base.so "${ASCEND_PATH}/compiler/lib64")

   # Link required dynamic libraries
   target_link_libraries(test_geir_add_example PRIVATE
        ${GRAPH_LIBRARY_DIR}
        ${GE_RUNNER_LIBRARY_DIR}
        ${GRAPH_BASE_LIBRARY_DIR}
   )

   # Set header file path
   target_include_directories(test_geir_add_example PRIVATE
        ${FWK_INCLUDE_DIR}/graph/
        ${FWK_INCLUDE_DIR}/ge/
        ${ASCEND_PATH}/opp/built-in/op_proto/inc/
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ASCEND_PATH}/compiler/include
   )
    ```

4. Create run.sh script.

   Create run.sh file in the same directory as test\_geir\_\$\{op\_name\}.cpp. Taking `AddExample` operator as example, example as follows, please modify according to actual situation.

    ```bash
    if [ -n "$ASCEND_INSTALL_PATH" ]; then                      # Actual CANN package installation path
        _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
    elif [ -n "$ASCEND_HOME_PATH" ]; then
        _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
    fi

    source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

    rm -rf build
    mkdir -p build
    cd build
    cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE  # Execute build command
    make
    ./test_geir_add_example                                     # Replace with actual operator executable file name
    ```

5. Run run.sh script.
    Execute the following command in the path where run.sh file is located:

    ```bash
    bash run.sh
    ```

    Default generates executable file test\_geir\_add\_example in current execution path `/build/bin`, running result as follows:

    ```bash
    INFO - [XIR]: Finalize ir graph session success
    ```
