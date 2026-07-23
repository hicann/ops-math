# Compile and Run Samples

## Prerequisites

- If you need to compile and execute operator API, please ensure basic environment has been set up, including driver, firmware, CANN software package, ops package, etc.
- For operator API invocation process and compilation running operation details, please refer to [Application Development (C&C++)](https://hiascend.com/document/redirect/CannCommunityCppInferWizard) "Single Operator Invocation > Single Operator API Execution > Call aclnn Interface Example Code".

## Preparation Before Compilation

This chapter takes development and runtime environment co-location scenario as example, i.e., machine with AI processor serves as both development environment and runtime environment. In this scenario, code development and code execution are on the same machine. Here taking **Abs operator** as example, other operators' invocation logic, process, compilation script are roughly the same as Abs operator. Please modify API invocation script (\*.cpp) and compilation script (CMakeLists) according to actual situation.

- **Sample Code**

   Known Abs operator function is to calculate absolute value of each element in tensor, calculation formula is: y<sub>i</sub>=|x<sub>i</sub>|. You can get invocation example code from aclnnAbs document, and name code file as "**test\_abs.cpp**".

- **CMakeLists File**

    CMake file example is as follows, please modify according to actual situation:

    ```bash
    # Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

    # CMake lowest version requirement
    cmake_minimum_required(VERSION 3.14)

    # Set project name
    project(ACLNN_EXAMPLE)

    # Compile options
    add_compile_options(-std=c++11)

    # Set compilation options
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "./bin")
    set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

    # Set executable file name (such as opapi_test), and specify directory of operator file *.cpp to run
    add_executable(opapi_test
                   test_abs.cpp)

    # Set ASCEND_PATH (CANN software package directory, please modify according to actual path) and INCLUDE_BASE_DIR (header file directory)
    if(NOT "$ENV{ASCEND_CUSTOM_PATH}" STREQUAL "")
        set(ASCEND_PATH $ENV{ASCEND_CUSTOM_PATH})
    else()
        set(ASCEND_PATH "/usr/local/Ascend/cann")
    endif()
    set(INCLUDE_BASE_DIR "${ASCEND_PATH}/include")
    include_directories(
        ${INCLUDE_BASE_DIR}
        ${INCLUDE_BASE_DIR}/aclnn
    )

    # Set linked library file path
    target_link_libraries(opapi_test PRIVATE
                          ${ASCEND_PATH}/lib64/libascendcl.so
                          ${ASCEND_PATH}/lib64/libnnopbase.so
                          ${ASCEND_PATH}/lib64/libopapi_math.so)

    # Executable file is in bin directory under CMakeLists file directory
    install(TARGETS opapi_test DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
    ```

    For operators that combine collective communication and MatMul computation fusion and parallel, collectively called communication-computation fusion operators (abbreviated as MC2 operators), including AllGatherMatmul, AlltoAllAllGatherBatchMatMul, BatchMatMulReduceScatterAlltoAll, MatmulAllReduce, MatmulAllReduceAddRmsNorm, MatmulReduceScatter, etc. When invoking this type of operator API, generally involves multi-threading and HCCL (Huawei Collective Communication Library, collective communication library), therefore CMake file needs to additionally import the following content, otherwise cannot compile successfully.

  ```text
  # Set linked library file path
  find_package(Threads REQUIRED)
  target_link_libraries(opapi_test PRIVATE
                        ${ASCEND_PATH}/lib64/libascendcl.so
                        ${ASCEND_PATH}/lib64/libnnopbase.so
                        ${ASCEND_PATH}/lib64/libopapi_math.so
                        ${ASCEND_PATH}/lib64/libhccl.so      # Collective communication library file
                        ${CMAKE_THREAD_LIBS_INIT})           # Library file depended by multi-threading
  ```

  Where "find_package(Threads REQUIRED)" is CMake command for finding thread library, can automatically link header files or indirectly dependent library files that thread library depends on.

## Compilation and Running

  1. Prepare operator invocation code (\*.cpp) and compilation script (CMakeLists.txt) in advance.
  2. Configure environment variables.

     After installing CANN software, use CANN runtime user to login environment, execute the following command to make environment variables effective.

        ```bash
        source ${INSTALL_DIR}/set_env.sh
        ```

     Where ${INSTALL_DIR} is CANN software installation file storage path, please replace according to actual situation.

  3. Compile and run.

      - Enter CMakeLists.txt directory, execute the following command, create build directory to store generated compilation files.

          ```bash
          mkdir -p build
          ```

      - Enter build directory, execute cmake command to compile, then execute make command to generate executable file.

        ```bash
        cd build
        cmake ../ -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
        make
        ```

        After successful compilation, opapi_test executable file will be generated in bin folder under build directory.

      - Enter bin directory, run executable file opapi_test.

        ```bash
        cd bin
        ./opapi_test
        ```

        Taking Abs operator running result as example, result example after running is as follows:

        ```bash
        result[0] is: 1.000000
        result[1] is: 1.000000
        result[2] is: 1.000000
        result[3] is: 2.000000
        result[4] is: 2.000000
        result[5] is: 2.000000
        result[6] is: 3.000000
        result[7] is: 3.000000
        ```

        If execution result reports error, expected result does not appear, can use aclGetRecentErrMsg interface to get specific error information.
        Calling aclnnAbsGetWorkspaceSize error getting exception information example is as follows:

        ```bash
        // self is nullptr
        ret = aclnnAbsGetWorkspaceSize(self, out, &workspaceSize, &executor);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAbsGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
        ```

        Above constructing null pointer problem getting error information example is as follows:

        ```bash
        aclnnAbsGetWorkspaceSize failed. ERROR: 161001
        [ERROR msg][PID:xxxx] xxx(timestamp) AclNN_Parameter_Error(EZ1001): Expected a value of type [aclTensor] for argument [self] but instead found nullptr.
        ```
