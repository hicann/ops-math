# AdjacentDifference op_api UT 覆盖率报告

## 结论

**op_api/adjacent_difference.cpp 行覆盖率: 0% → 100% (7/7 行), 函数覆盖率 100% (1/1)**

## 根因

原测试文件 `tests/ut/op_api/test_adjacent_difference_l0.cpp` 命名不符合 CMake 收集规则:
`cmake/ut.cmake` 中 `file(GLOB OPAPI_CASES_SRC ${MODULE_DIR}/test_aclnn_*.cpp)` 只收集
`test_aclnn_` 前缀的用例文件, 因此原用例从未被编译进 `math_op_api_ut` 可执行文件,
函数 `l0op::AdjacentDifference` 无任何 gcda 生成, 覆盖率为 0。

## 修复

新增 `tests/ut/op_api/test_aclnn_adjacent_difference.cpp`(符合 glob 规则), 共 19 个用例:

| 用例类别 | 数量 | 说明 |
|---------|------|------|
| 输入 dtype 全覆盖 × 输出 INT32 | 11 | FLOAT/FLOAT16/BF16/INT8/INT16/INT32/INT64/UINT8/UINT16/UINT32/UINT64 |
| 输入 dtype 代表性 × 输出 INT64 | 3 | FLOAT/INT32/INT64 |
| shape 边界 | 5 | 1D/5D/空 tensor {0,3}/单元素 {1}/非连续 stride |

每个用例校验: 输出非空、输出 shape 与输入一致 (GetViewShape)、输出 dtype 为指定 yDtype (GetDataType)。

## 验证环境

- CANN: 9.2.0-beta.1 (Ascend 950PR, arch35)
- 构建: `bash build.sh -u --opapi --ops=adjacent_difference --soc=ascend950 --cov --noexec`
- 执行: `./build/tests/ut/op_api/math_op_api_ut --gtest_filter=*AdjacentDifference*`
  → 19 tests, 19 passed, 0 failed
- 覆盖率采集: lcov 2.0 (coverage.info 为本地中间产物, 不入库; 原始数据已摘录至下方逐行覆盖明细)

## 逐行覆盖明细 (adjacent_difference.cpp)

| 行号 | 命中次数 | 内容 |
|------|---------|------|
| 29 | 19 | 函数入口 `l0op::AdjacentDifference` |
| 31 | 19 | OP_LOGI |
| 32 | 19 | L0_DFX |
| 34 | 19 | AllocTensor(x->GetViewShape(), yDtype) |
| 36 | 19 | ADD_TO_LAUNCHER_LIST_AICORE (stub 模式) |
| 37 | 19 | OP_LOGI Launch finish |
| 38 | 19 | return y |
