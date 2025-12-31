cmake_minimum_required(VERSION 3.14)
project(Col2ImTilingTest)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 查找依赖包
find_package(GTest REQUIRED)

# 假设你的项目结构如下：
# - include/ (头文件)
# - src/ (源文件)
# - test/ (测试文件)

# 包含华为 CANN 头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/local/Ascend/ascend-toolkit/latest/include
    /usr/local/Ascend/ascend-toolkit/latest/include/ge
    /usr/local/Ascend/ascend-toolkit/latest/include/gert
)

# 链接库目录
link_directories(
    /usr/local/Ascend/ascend-toolkit/latest/lib64
)

# 添加测试可执行文件
add_executable(col2im_tiling_test
    test/main.cpp  # 你的测试主文件
    test/col2im_tiling_test.cpp  # 你提供的测试文件
    # 其他源文件...
)

# 链接库
target_link_libraries(col2im_tiling_test
    GTest::gtest
    GTest::gtest_main
    ascend_pass  # 华为 CANN 库
    ge           # 图引擎库
    gert         # Tiling 相关库
    pthread
    dl
)

# 启用测试
enable_testing()
add_test(NAME Col2ImTilingTest COMMAND col2im_tiling_test)