# Project Directory

> Some directories listed in this chapter are optional, please refer to actual deliverables. Especially **single operator directory**, deliverables vary in different scenarios, specific description as follows:
>
> - If op_host directory is missing, it may be calling other operator op_host implementation, see the operator op_api or op_graph directory source code implementation for calling logic; or Kernel may not have Ascend C implementation temporarily, if needed, welcome developers to refer to [Contribution Guide](../../../CONTRIBUTING.md) to supplement and contribute this operator.
> - If op_kernel directory is missing, it may be calling other operator op_kernel implementation, see the operator op_api or op_graph directory source code implementation for calling logic; or Kernel may not have Ascend C implementation temporarily, if needed, welcome developers to refer to [Contribution Guide](../../../CONTRIBUTING.md) to supplement and contribute this operator.
> - If op_api directory is missing, it means this operator does not support aclnn invocation temporarily.
> - If op_graph directory is missing, it means this operator does not support graph mode invocation temporarily.

Project full directory hierarchy introduction is as follows:

```text
├── cmake                                               # Project compilation directory
│   ├── aclnn_ops_math.h.in                             # aclnn summary header file template
│   └── ...
├── common                                              # Project common header files and common code
│   ├── CMakeLists.txt
│   ├── inc                                             # Common header file directory
│   └── src                                             # Common code directory
├── experimental                                        # User-defined operator storage directory
│   ├── conversion                                      # Optional, user-developed conversion class operator directory
│   │   └── CMakeLists.txt
│   ├── math                                            # Optional, user-developed math class operator directory
│   │   └── CMakeLists.txt
│   ├── random                                          # Optional, user-developed random class operator directory
│   │   └── CMakeLists.txt
|   └── CMakeLists.txt
├── ${op_class}                                         # Operator classification, such as conversion, math, random class operators
│   ├── ${op_name}                                         # Operator project directory, ${op_name} represents operator name (lowercase underscore form)
│   │   ├── CMakeLists.txt                              # Operator CMakeLists.txt entry
│   │   ├── README.md                                   # Operator introduction document
│   │   ├── docs                                        # Operator document directory
│   │   │   └── aclnn${OpName}.md                       # Operator aclnn interface introduction document, ${OpName} represents operator name (upper camel case form)
│   │   ├── examples                                    # Operator invocation example directory
│   │   │   ├── test_aclnn_${op_name}.cpp               # Operator invocation example through aclnn
│   │   │   └── test_geir_${op_name}.cpp                # Operator invocation example through geir
│   │   ├── op_graph                                    # Graph fusion related implementation
│   │   │   ├── CMakeLists.txt                          # op_graph side CMakeLists.txt file
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType file, implements operator data type deduction
│   │   │   ├── ${op_name}_proto.h                      # Operator prototype definition, used for graph optimization and fusion phase to identify operator
│   │   │   └── fusion_pass                             # Operator fusion rule directory
│   │   ├── op_host                                     # Host-side implementation
│   │   │   ├── config                                  # Optional, binary configuration file, if not configured project automatically generates
│   │   │   │   ├── ${soc_version}                      # Binary information configured by operator on NPU, ${soc_version} represents NPU model
│   │   │   │   │   ├── ${op_name}_binary.json          # Operator binary configuration file
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # Operator SimplifiedKey configuration information
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # Operator information library, defines basic operator information, such as name, input/output, data type, etc.
│   │   │   ├── ${op_name}_infershape.cpp               # Optional, InferShape implementation, deduces output shape according to operator shape, if not configured output shape is same as input shape
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # Optional, Tiling optimization for certain sub-scenarios, ${sub_case} represents sub-scenario, such as ${op_name}_tiling_arch35 is optimization for arch35 architecture, if no such file indicates this operator has no specific Tiling strategy for corresponding sub-scenario
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # Optional, header file used for Tiling implementation in ${sub_case} sub-scenario
│   │   │   ├── ${op_name}_tiling.cpp                   # Optional, if no such file indicates no Tiling implementation in corresponding scenario (divides tensor into multiple small blocks, distinguishes data types for parallel computation)
│   │   │   ├── ${op_name}_tiling.h                     # Optional, header file used for Tiling implementation
│   │   ├── op_api                                      # Optional, operator aclnn implementation file directory, if not configured project automatically generates
│   │   │   ├── aclnn_${op_name}.cpp                    # Operator aclnn interface implementation file
│   │   │   ├── aclnn_${op_name}.h                      # Operator aclnn interface implementation header file
│   │   │   ├── ${op_name}.cpp                          # Operator l0 interface implementation file
│   │   │   ├── ${op_name}.h                            # Operator l0 interface implementation header file
│   │   │── op_kernel                                   # AI Core operator Device-side Kernel implementation
│   │   │   ├── ${sub_case}                             # Optional, directory used by ${sub_case} sub-scenario
│   │   │   │   ├── ${op_name}_${model}.h               # Operator kernel implementation file, ${model} represents user-defined file name suffix, usually Tiling template name
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # Optional, TilingKey file, defines Key of Tiling strategy, identifies different partitioning methods, if not configured indicates this operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}_tiling_data.h                # Optional, TilingData file, stores Tiling strategy related configuration information, such as block size, parallelism, if not configured indicates this operator has no corresponding Tiling strategy
│   │   │   ├── ${op_name}.cpp                          # Kernel entry file, contains main function and scheduling logic
│   │   │   └── ${op_name}.h                            # Kernel implementation file, defines Kernel header file, contains function declarations, structure definitions, logic implementation
│   │   │── op_kernel_aicpu                             # Optional, AI CPU operator Device-side Kernel implementation
│   │   │   ├── ${op_name}_aicpu.cpp                    # Kernel entry file, contains main function and scheduling logic
│   │   │   └── ${op_name}_aicpu.h                      # Kernel header file, contains function declarations, structure definitions, logic implementation
│   │   └── tests                                       # Operator test case directory
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # Optional, UT test cases, develop corresponding cases according to actual situation
│   │           ├── CMakeLists.txt                      # UT case CMakeLists.txt file
│   │           ├── graph_plugin                        # graph_plugin test case directory
│   │           │   ├── CMakeLists.txt
│   │           │   └── fusion_pass                     # Fusion rule test case directory
│   │           │       └── CMakeLists.txt
│   │           ├── op_host                             # op_host test case directory
│   │           │   ├── CMakeLists.txt
│   │           │   ├── ${op_name}_regbase_tiling.h
│   │           │   ├── op_api                          # op_api test case directory
│   │           │   │   ├── CMakeLists.txt
│   │           │   │   └── test_aclnn_${op_name}.cpp   # Operator aclnn test case file
│   │           │   ├── test_${op_name}_${sub_case}.cpp # op_host test case file in ${sub_case} sub-scenario
│   │           │   ├── test_${op_name}.cpp             # op_host test case file
│   │           │   ├── test_${op_name}_infershape.cpp  # Operator InferShape test case file
│   │           │   └── test_${op_name}_tiling.cpp      # Operator Tiling test case file
│   │           └── op_kernel                           # op_kernel test case directory
│   │               ├── CMakeLists.txt
│   │               │── test_${op_name}.cpp             # Operator Kernel test case file
│   │               └── ${op_name}_data                 # Optional, data comparison and generation scripts depended by op_kernel test cases, if not configured need to manually implement in corresponding cases
│   │                   ├── compare_data.py             # Data script
│   │                   └── gen_data.py                 # Data generation script
│   └── ...
├── docs                                                # Project related document directory
├── examples                                            # End-to-end operator development and invocation examples
│   ├── add_example                                     # AI Core operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator usage example directory
│   │   ├── op_graph                                    # Operator graph composition related directory
│   │   ├── op_host                                     # Operator information library, Tiling, InferShape related implementation directory
│   │   ├── op_kernel                                   # Operator Kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── add_example_aicpu                               # AI CPU operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator usage example directory
│   │   ├── op_graph                                    # Operator graph composition related directory
│   │   ├── op_host                                     # Operator information library, InferShape related implementation
│   │   ├── op_kernel_aicpu                             # Operator Kernel directory
│   │   └── tests                                       # Operator test case directory
│   ├── add_example_c_api                               # C API operator example directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── examples                                    # Operator usage example directory
│   │   ├── op_graph                                    # Operator graph composition related directory
│   │   ├── op_host                                     # Operator information library, Tiling, InferShape related implementation directory
│   │   └── op_kernel                                   # Operator Kernel directory
│   ├── CMakeLists.txt
│   ├── fast_kernel_launch_example                       # Lightweight, high-performance operator development project template
│   │   ├── ascend_ops                                  # Example operator implementation directory
│   │   ├── cmake                                       # Build related cmake script directory
│   │   ├── CMakeLists.txt                              # Operator compilation configuration file
│   │   ├── csrc                                        # C/C++ extension source code directory
│   │   ├── README.md                                   # Lightweight, high-performance operator development project description material
│   │   ├── requirements.txt
│   │   ├── setup.py                                    # Build script
│   │   └── tests                                       # Test case directory
│   └── README.md                                       # Project example introduction document
├── scripts                                             # Script directory, contains custom operator, Kernel build related configuration files
├── tests                                               # Project-level test directory
│   ├── requirements.txt                                # Third-party components depended by test cases
│   └── ut                                              # UT case project
│       ├── CMakeLists.txt                              # UT project CMakeLists.txt script
│       ├── common                                      # Common code used in UT project
│       ├── op_api                                      # op_api test project
│       ├── op_host                                     # op_host test project
│       └── op_kernel                                   # op_kernel test project
├── CMakeLists.txt                                      # Project CMakeLists.txt entry
├── CONTRIBUTING.md                                     # Project contribution guide file
├── LICENSE                                             # Project open source license information
├── OAT.xml                                             # Configuration script, used by repository tools, used to check if License is standard
├── README.md                                           # Project general introduction document
├── SECURITY.md                                         # Project security statement file
├── build.sh                                            # Project compilation script
├── classify_rule.yaml                                  # Component division information
├── install_deps.sh                                     # Project install dependency package script
├── requirements.txt                                    # Project third-party dependency packages
└── version.cmake                                       # Project version information
```
