#include <gtest/gtest.h>
#include <iostream>
#include "infershape_context_faker.h"
#include "infershape_case_executor.h"

class ReduceMeanWithCountInfershape : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceMeanWithCountInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceMeanWithCountInfershape TearDown" << std::endl;
    }
};

// ============================================================================
// Helper: Build StorageShape from a vector of dims
// ============================================================================
static gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape ss;
    ss.MutableOriginShape().SetDimNum(dims.size());
    ss.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        ss.MutableOriginShape().SetDim(i, dims[i]);
        ss.MutableStorageShape().SetDim(i, dims[i]);
    }
    return ss;
}

// ============================================================================
// Helper: Build InfershapeContextPara for ReduceMeanWithCount
//   Input: input (1 tensor)
//   Output: mean_result, count_result (2 tensors)
//   Attrs: axis (ListInt, index 0), keepdim (Bool, index 1)
// ============================================================================
static gert::InfershapeContextPara MakeCtx(
    const std::vector<int64_t>& inputDims,
    ge::DataType dtype,
    const std::vector<int64_t>& axis,
    bool keepdim)
{
    gert::StorageShape inShape = MakeStorageShape(inputDims);
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {
        {inShape, dtype, ge::FORMAT_ND},
    };
    // Two outputs: mean_result and count_result (placeholder shapes, will be inferred)
    gert::StorageShape outPlaceholder;
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {
        {outPlaceholder, dtype, ge::FORMAT_ND},
        {outPlaceholder, ge::DT_INT64, ge::FORMAT_ND},
    };
    // Attrs: axis (ListInt), keepdim (Bool)
    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(axis)},
        {"keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepdim)},
    };
    return gert::InfershapeContextPara("ReduceMeanWithCount", inputs, outputs, attrs);
}

// ============================================================================
// Test: single axis reduction, keepdim=false
// input: [4, 6, 8], axis=[1], keepdim=false -> output: [4, 8]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, single_axis_keepdim_false)
{
    auto ctx = MakeCtx({4, 6, 8}, ge::DT_FLOAT, {1}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 8},  // mean_result
        {4, 8},  // count_result
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: single axis reduction, keepdim=true
// input: [4, 6, 8], axis=[1], keepdim=true -> output: [4, 1, 8]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, single_axis_keepdim_true)
{
    auto ctx = MakeCtx({4, 6, 8}, ge::DT_FLOAT, {1}, true);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 1, 8},
        {4, 1, 8},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: multi-axis reduction, keepdim=false
// input: [2, 3, 4, 5], axis=[1,3], keepdim=false -> output: [2, 4]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, multi_axis_keepdim_false)
{
    auto ctx = MakeCtx({2, 3, 4, 5}, ge::DT_FLOAT, {1, 3}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4},
        {2, 4},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: multi-axis reduction, keepdim=true
// input: [2, 3, 4, 5], axis=[1,3], keepdim=true -> output: [2, 1, 4, 1]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, multi_axis_keepdim_true)
{
    auto ctx = MakeCtx({2, 3, 4, 5}, ge::DT_FLOAT, {1, 3}, true);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 1, 4, 1},
        {2, 1, 4, 1},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: all-axis reduction (empty axis list), keepdim=false
// input: [2, 3, 4], axis=[], keepdim=false -> output: [] (scalar)
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, all_axis_keepdim_false)
{
    auto ctx = MakeCtx({2, 3, 4}, ge::DT_FLOAT, {}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {},  // scalar
        {},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: all-axis reduction, keepdim=true
// input: [2, 3, 4], axis=[], keepdim=true -> output: [1, 1, 1]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, all_axis_keepdim_true)
{
    auto ctx = MakeCtx({2, 3, 4}, ge::DT_FLOAT, {}, true);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {1, 1, 1},
        {1, 1, 1},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: negative axis normalization
// input: [2, 3, 4], axis=[-1], keepdim=false -> output: [2, 3]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, negative_axis)
{
    auto ctx = MakeCtx({2, 3, 4}, ge::DT_FLOAT, {-1}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 3},
        {2, 3},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: first axis reduction
// input: [4, 6, 8], axis=[0], keepdim=false -> output: [6, 8]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, first_axis_keepdim_false)
{
    auto ctx = MakeCtx({4, 6, 8}, ge::DT_FLOAT, {0}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {6, 8},
        {6, 8},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: last axis reduction
// input: [4, 6, 8], axis=[2], keepdim=false -> output: [4, 6]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, last_axis_keepdim_false)
{
    auto ctx = MakeCtx({4, 6, 8}, ge::DT_FLOAT, {2}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4, 6},
        {4, 6},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: 2D tensor reduction
// input: [10, 20], axis=[0], keepdim=false -> output: [20]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, two_dim_reduce_axis0)
{
    auto ctx = MakeCtx({10, 20}, ge::DT_FLOAT, {0}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {20},
        {20},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: FP16 dtype
// input: [4, 8], axis=[1], keepdim=false -> output: [4]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, fp16_dtype)
{
    auto ctx = MakeCtx({4, 8}, ge::DT_FLOAT16, {1}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {4},
        {4},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: duplicate axes should be deduplicated
// input: [2, 3, 4], axis=[1, 1], keepdim=false -> output: [2, 4]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, duplicate_axes)
{
    auto ctx = MakeCtx({2, 3, 4}, ge::DT_FLOAT, {1, 1}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4},
        {2, 4},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}

// ============================================================================
// Test: consecutive multi-axis (merge adjacent R dims)
// input: [2, 3, 4, 5], axis=[1,2], keepdim=false -> output: [2, 5]
// ============================================================================
TEST_F(ReduceMeanWithCountInfershape, consecutive_multi_axis)
{
    auto ctx = MakeCtx({2, 3, 4, 5}, ge::DT_FLOAT, {1, 2}, false);
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 5},
        {2, 5},
    };
    ExecuteTestCase(ctx, ge::GRAPH_SUCCESS, expectOutputShape);
}
