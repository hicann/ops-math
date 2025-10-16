# 项目目录

## 目录介绍

详细目录介绍说明如下：
```
├── build.sh                                            # 项目工程编译脚本
├── install_deps.sh                                     # 安装依赖包脚本
├── cmake                                               # 项目工程编译目录
├── CMakeLists.txt                                      # ops-math工程cmakelist入口
├── classify_rule.yaml                                  # 组件划分信息
├── common                                              # 项目公共头文件和公共代码
│   ├── CMakeLists.txt
│   ├── inc                                             # 公共头文件目录
│   └── src                                             # 公共代码目录
├── CONTRIBUTING.md                                     # 贡献指南文档
├── conversion                                          # conversion类算子
├── docs                                                # 项目文档介绍目录
├── examples                                            # 端到端算子开发和调用示例
│   ├── add_example                                     # AI Core算子示例目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件 
│   │   ├── examples                                    # 算子使用示例目录
│   │   ├── op_graph                                    # 算子构图相关目录
│   │   ├── op_host                                     # 算子信息库、Tiling、InferShape相关实现目录
│   │   ├── op_kernel                                   # 算子kernel目录
│   │   └── tests                                       # 算子测试用例目录
│   ├── add_example_aicpu                               # AI CPU算子示例目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── examples                                    # 算子使用示例目录
│   │   ├── op_graph                                    # 算子构图相关目录
│   │   ├── op_host                                     # 算子信息库、InferShape相关实现
│   │   ├── op_kernel_aicpu                             # 算子kernel目录
│   │   └── tests                                       # 算子测试用例目录
│   ├── CMakeLists.txt
│   ├── fast_kernel_launch_example                      # 轻量级，高性能的算子开发工程模板
│   │   ├── ascend_ops                                  # 示例算子实现目录
│   │   ├── CMakeLists.txt                              # 算子编译配置文件
│   │   ├── README.md                                   # 轻量级，高性能的算子开发工程说明资料
│   │   ├── requirements.txt
│   │   └── setup.py                                    # 构建脚本
│   └── README.md                                       # 示例工程详细说明文档
├── math                                                # math类算子
│   ├${op_name}                                         # 算子工程目录，${op_name}表示实际的算子名（小写下划线形式）
│   │   ├── CMakeLists.txt                              # 算子cmakelist入口
│   │   ├── README.md                                   # 算子说明资料
│   │   ├── docs                                        # 算子aclnn资料目录
│   │   │   └── aclnn${OpName}.md                       # 算子aclnn接口说明资料，${OpName}表示实际的算子（大驼峰形式）
│   │   ├── examples                                    # 算子接口调用示例目录
│   │   │   ├── test_aclnn_${op_name}.cpp               # 算子aclnn调用示例
│   │   │   └── test_geir_${op_name}.cpp                # 算子geir调用示例
│   │   ├── op_graph                                    # 图融合相关实现
│   │   │   ├── CMakeLists.txt                          # op_graph侧cmakelist文件
│   │   │   ├── ${op_name}_graph_infer.cpp              # InferDataType文件，实现算子数据类型推导
│   │   │   ├── ${op_name}_proto.h                      # 算子原型定义，用于图优化和融合阶段识别算子
│   │   │   └── fusion_pass                             # 算子融合规则目录
│   │   ├── op_host                                     # Host侧实现
│   │   │   ├── CMakeLists.txt                          # Host侧cmakelist文件
│   │   │   ├── config                                  # 可选，二进制配置文件，如果没有配置则工程会自动生成
│   │   │   │   ├── ${soc_version}                      # 算子在对应NPU上配置的二进制信息，${soc_version}表示NPU型号
│   │   │   │   │   ├── ${op_name}_binary.json          # 算子二进制配置文件
│   │   │   │   │   └── ${op_name}_simplified_key.ini   # 算子simplified_key配置信息
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_def.cpp                      # 算子信息库，定义算子基本信息，如名称、输入输出、数据类型等
│   │   │   ├── ${op_name}_infershape.cpp               # 可选，InferShape实现，实现算子形状推导输出shape，如没配置则输出shape与输入shape一样
│   │   │   ├── ${op_name}_tiling_${sub_case}.cpp       # 可选，针对某些子场景下的tiling优化，${sub_case}表示子场景，如：${op_name}_tiling_arch35是针对arch35架构的优化，如没有该类文件则表明该算子没有对应子场景的特定tiling策略
│   │   │   ├── ${op_name}_tiling_${sub_case}.h         # 可选，${sub_case}场景下Tiling实现用的头文件
│   │   │   ├── ${op_name}_tiling.cpp                   # 可选，如不使用该文件表明对应场景下无tiling优化，Tiling实现，将张量划分为多个小块，区分数据类型进行并行计算
│   │   │   ├── ${op_name}_tiling.h                     # 可选，Tiling实现用的头文件
│   │   │   └── op_api                                  # 可选，算子aclnn接口实现目录，如未提供则表示此算子的aclnn接口会让工程自动生成
│   │   │       ├── aclnn_${op_name}.cpp                # 算子aclnn接口实现文件
│   │   │       ├── aclnn_${op_name}.h                  # 算子aclnn接口实现头文件
│   │   │       ├── ${op_name}.cpp                      # 算子l0接口实现文件
│   │   │       ├── ${op_name}.h                        # 算子l0接口实现头文件
│   │   │       └── CMakeLists.txt
│   │   │── op_kernel                                   # Device侧Kernel实现
│   │   │   ├── ${sub_case}                             # 可选，${sub_case}对应的子场景使用的目录，无相应的场景不需要配本目录
│   │   │   │   ├── ${op_name}_${model}.h               # 算子kernel实现文件，${model}表示用户自定义文件名后缀，通常为tiling模板名
│   │   │   │   └── ...
│   │   │   ├── ${op_name}_tiling_key.h                 # 可选，Tilingkey文件，定义Tiling策略的Key，标识不同的划分方式，如没配置则表明该算子无相应的tiling策略
│   │   │   ├── ${op_name}_tiling_data.h                # 可选，Tilingdata文件，存储Tiling策略相关的配置数据，如块大小、并行度，如没配置则表明该算子无相应的tiling策略
│   │   │   ├── ${op_name}.cpp                          # Kernel入口文件，包含主函数和调度逻辑
│   │   │   └── ${op_name}.h                            # Kernel实现文件，定义Kernel头文件，包含函数声明、结构定义、逻辑实现
│   │   └── tests                                       # 算子测试用例
│   │       ├── CMakeLists.txt
│   │       └── ut                                      # 可选，ut测试用例目录，根据实际情况开发相应的用例
│   │           ├── CMakeLists.txt                      # ut用例cmakelist文件
│   │           ├── graph_plugin                        # grap_plugin测试用例目录
│   │           │   ├── CMakeLists.txt
│   │           │   └── fusion_pass                     # 融合规则测试用例目录
│   │           │       └── CMakeLists.txt
│   │           ├── op_host                             # op_host测试用例目录
│   │           │   ├── CMakeLists.txt
│   │           │   ├── ${op_name}_regbase_tiling.h
│   │           │   ├── op_api                          # op_api测试用例目录
│   │           │   │   ├── CMakeLists.txt
│   │           │   │   └── test_aclnn_${op_name}.cpp   # 算子op_api测试用例文件
│   │           │   ├── test_${op_name}_${sub_case}.cpp # ${sub_case}场景下的op_host测试用例
│   │           │   ├── test_${op_name}.cpp             # op_host测试用例
│   │           │   ├── test_${op_name}_infershape.cpp  # 算子infershape测试用例文件
│   │           │   └── test_${op_name}_tiling.cpp      # 算子tiling测试用例文件
│   │           └── op_kernel                           # op_kernel测试用例目录
│   │               ├── CMakeLists.txt
│   │               │── test_${op_name}.cpp             # 算子kernel测试用例
│   │               └── ${op_name}_data                 # 可选，op_kernel测试用例中使用的数据生成及对比脚本，如没配置则需要在对应的用例中实现相应的功能
│   │                   ├── compare_data.py             # 数据比较脚本
│   │                   └── gen_data.py                 # 数据生成脚本
│   └── ...
├── OAT.xml                                             # 配置脚本，代码仓工具使用，用于检查license是否规范
├── random                                              # random类算子
├── README.md                                           # 项目工程资料文档
├── LICENSE                                             # 开源声明信息
├── requirements.txt                                    # 项目需要的第三方依赖包
├── scripts                                             # 脚本目录，包含自定义算子、kernel构建相关配置文件
├── SECURITY.md                                         # 项目安全声明文档
├── tests                                               # 测试工程目录
│   ├── requirements.txt                                # 测试用例依赖的第三方组件
│   └── ut                                              # UT用例工程
│       ├── CMakeLists.txt                              # ut工程的cmakelist脚本
│       ├── common                                      # ut工程中使用的公共代码
│       ├── op_api                                      # op_api测试工程
│       ├── op_host                                     # op_host测试工程
│       └── op_kernel                                   # op_kernel测试工程
└── version.info                                        # 版本信息
```

