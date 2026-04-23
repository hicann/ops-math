#include <iostream>
#include <cstring>
#include <cmath>
#include <gtest/gtest.h>
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "reduce_mean_with_count_tiling_data.h"

namespace ReduceMeanWithCountUT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "ReduceMeanWithCount";

// Platform parameters for Ascend950
static const uint64_t MAX_AIV_NUM = 40;
static const uint64_t UB_SIZE = 393216;  // 384KB typical for Ascend950
static const uint64_t TILING_DATA_MAX_SIZE = 4096;

struct ReduceMeanWithCountCompileInfo {
} compileInfo;

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
// Helper: create TilingContextPara for ReduceMeanWithCount
// ============================================================================
static gert::TilingContextPara MakeTilingCtx(
    const std::vector<int64_t>& inputShape,
    ge::DataType dtype,
    const std::vector<int64_t>& meanOutShape,
    const std::vector<int64_t>& countOutShape,
    const std::vector<int64_t>& axis,
    bool keepdim,
    uint64_t coreNum = MAX_AIV_NUM,
    uint64_t ubSize = UB_SIZE)
{
    gert::StorageShape inSS = MakeStorageShape(inputShape);
    gert::StorageShape meanSS = MakeStorageShape(meanOutShape);
    gert::StorageShape countSS = MakeStorageShape(countOutShape);

    std::vector<gert::TilingContextPara::TensorDescription> inputs = {
        {inSS, dtype, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputs = {
        {meanSS, dtype, ge::FORMAT_ND},
        {countSS, ge::DT_INT64, ge::FORMAT_ND},
    };
    std::vector<gert::TilingContextPara::OpAttr> attrs = {
        {"axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(axis)},
        {"keepdim", Ops::Math::AnyValue::CreateFrom<bool>(keepdim)},
    };
    return gert::TilingContextPara(
        OP_NAME, inputs, outputs, attrs, &compileInfo,
        coreNum, ubSize, TILING_DATA_MAX_SIZE);
}

// ============================================================================
// Helper: extract ReduceMeanWithCountTilingData from raw tiling bytes
// ============================================================================
static ReduceMeanWithCountTilingData ExtractTilingData(const TilingInfo& info) {
    ReduceMeanWithCountTilingData td;
    EXPECT_GE(info.tilingDataSize, sizeof(ReduceMeanWithCountTilingData));
    std::memcpy(&td, info.tilingData.get(), sizeof(ReduceMeanWithCountTilingData));
    return td;
}

// ============================================================================
// Test class
// ============================================================================
class ReduceMeanWithCountTilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ReduceMeanWithCountTilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ReduceMeanWithCountTilingTest TearDown." << std::endl;
    }
};

// ============================================================================
// TK0: AR full-load, FP32, single axis last dim
// input: [10, 64], axis=[1], keepdim=false
// Expected: a1Length=10, rLength=64, a0Length=1, countResult=64
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk0_fp32_single_axis_last_dim)
{
    auto ctx = MakeTilingCtx(
        {10, 64}, ge::DT_FLOAT, {10}, {10}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 10UL);
    EXPECT_EQ(td.rLength, 64UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 64);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 64.0f);
    EXPECT_EQ(td.outputLength, 10UL);
    EXPECT_GT(td.usedCoreNum, 0);
}

// ============================================================================
// TK0: AR full-load, FP32, all-axis reduction
// input: [2, 3, 4], axis=[], keepdim=false
// Expected: a1Length=1, rLength=24 (since all dims are reduce), a0Length=1
// countResult=24
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk0_fp32_all_axis_reduction)
{
    auto ctx = MakeTilingCtx(
        {2, 3, 4}, ge::DT_FLOAT, {}, {}, {}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // All dims reduce: merged = [R:24], a1=1, r=24, a0=1
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 24UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 24);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 24.0f);
    EXPECT_EQ(td.outputLength, 1UL);
}

// ============================================================================
// TK0: AR full-load, FP32, first axis reduction
// input: [4, 6, 8], axis=[0], keepdim=false
// Merged: [R:4, A:48] -> R before A -> a1=1, r=4, a0=48
// But a0!=1, so ARA mode... Let's check the scene logic.
// Actually: tags = [R, A, A], merged=[R:4, A:48], size=2
// merged[0]=R, merged[1]=A => a1=1, r=4, a0=48 => ARA mode
// So this should NOT be TK0. Let me use a pure AR case instead.
// ============================================================================
// Replaced with: input: [8, 64], axis=[1], keepdim=false (pure AR)
TEST_F(ReduceMeanWithCountTilingTest, tk0_fp32_reduce_last_dim)
{
    auto ctx = MakeTilingCtx(
        {8, 64}, ge::DT_FLOAT, {8}, {8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 8UL);
    EXPECT_EQ(td.rLength, 64UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 64);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 64.0f);
    EXPECT_EQ(td.outputLength, 8UL);
    EXPECT_GT(td.usedCoreNum, 0);
    EXPECT_GE(td.tilesPerCore, 1UL);
}

// ============================================================================
// TK0: AR full-load, FP16, axis merging with consecutive reduce dims
// input: [4, 3, 5, 8], axis=[1, 2], keepdim=false
// tags: [A, R, R, A], merged: [A:4, R:15, A:8] -> ARA pattern
// Actually this is ARA since a0=8, not AR.
// Let me pick: input: [4, 3, 5], axis=[1, 2], keepdim=false
// tags: [A, R, R], merged: [A:4, R:15], a1=4, r=15, a0=1 => AR
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk0_fp16_consecutive_reduce_dims)
{
    auto ctx = MakeTilingCtx(
        {4, 3, 5}, ge::DT_FLOAT16, {4}, {4}, {1, 2}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 15UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 15);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 15.0f);
    EXPECT_EQ(td.outputLength, 4UL);
}

// ============================================================================
// TK0: AR full-load, FP32, large A1 multi-core split
// input: [100, 32], axis=[1], keepdim=false
// a1=100, r=32, a0=1 => AR mode
// usedCoreNum should be > 1 for 100 rows
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk0_fp32_multicore_split)
{
    auto ctx = MakeTilingCtx(
        {100, 32}, ge::DT_FLOAT, {100}, {100}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 100UL);
    EXPECT_EQ(td.rLength, 32UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_GT(td.usedCoreNum, 1);
    // tilesPerCore * (usedCoreNum-1) + tailCoreTiles = a1Length
    EXPECT_EQ(td.tilesPerCore * (td.usedCoreNum - 1) + td.tailCoreTiles, td.a1Length);
    EXPECT_EQ(td.countResult, 32);
}

// ============================================================================
// Axis merging: size-1 dims should be removed
// input: [4, 1, 8], axis=[2], keepdim=false
// shapeDims = [4, 1, 8], tags = [A, A, R]
// Removing size-1 dim(1): merged = [A:4, R:8]
// a1=4, r=8, a0=1 => AR
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, axis_merging_skip_size1_dims)
{
    auto ctx = MakeTilingCtx(
        {4, 1, 8}, ge::DT_FLOAT, {4, 1}, {4, 1}, {2}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 8UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 8);
}

// ============================================================================
// Axis merging: adjacent same-type dims merged
// input: [2, 3, 4, 5], axis=[2, 3], keepdim=false
// tags: [A, A, R, R], merged: [A:6, R:20]
// a1=6, r=20, a0=1 => AR
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, axis_merging_adjacent_same_type)
{
    auto ctx = MakeTilingCtx(
        {2, 3, 4, 5}, ge::DT_FLOAT, {2, 3}, {2, 3}, {2, 3}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 6UL);
    EXPECT_EQ(td.rLength, 20UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 20);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 20.0f);
}

// ============================================================================
// countResult/invCount: verify for various count values
// input: [5, 7], axis=[1], keepdim=false -> countResult=7
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, count_result_and_inv_count)
{
    auto ctx = MakeTilingCtx(
        {5, 7}, ge::DT_FLOAT, {5}, {5}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.countResult, 7);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 7.0f);
}

// ============================================================================
// ARA mode: reduce middle axis
// input: [4, 6, 8], axis=[1], keepdim=false
// tags: [A, R, A], merged: [A:4, R:6, A:8]
// a1=4, r=6, a0=8 => ARA mode
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, ara_mode_reduce_middle_axis)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 8}, ge::DT_FLOAT, {4, 8}, {4, 8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 8UL);
    EXPECT_EQ(td.countResult, 6);
    EXPECT_GT(td.tileA0Len, 0UL);
}

// ============================================================================
// keepdim=true: verify outputLength
// input: [4, 8], axis=[1], keepdim=true -> output shape [4, 1], outputLength=4
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, keepdim_true_output_length)
{
    auto ctx = MakeTilingCtx(
        {4, 8}, ge::DT_FLOAT, {4, 1}, {4, 1}, {1}, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 8UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.outputLength, 4UL);
    EXPECT_EQ(td.countResult, 8);
}

// ============================================================================
// BF16 dtype: verify typeSize=2 affects alignment
// input: [10, 32], axis=[1], keepdim=false
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, bf16_dtype_tiling)
{
    auto ctx = MakeTilingCtx(
        {10, 32}, ge::DT_BF16, {10}, {10}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 10UL);
    EXPECT_EQ(td.rLength, 32UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 32);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 32.0f);
}

// ============================================================================
// Single element input: [1], axis=[0] -> all reduce
// countResult=1, outputLength=1
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, single_element)
{
    auto ctx = MakeTilingCtx(
        {1}, ge::DT_FLOAT, {}, {}, {0}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.countResult, 1);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f);
    EXPECT_EQ(td.outputLength, 1UL);
}

// ============================================================================
// Negative axis in tiling
// input: [4, 8], axis=[-1], keepdim=false -> same as axis=[1]
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, negative_axis_tiling)
{
    auto ctx = MakeTilingCtx(
        {4, 8}, ge::DT_FLOAT, {4}, {4}, {-1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 8UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 8);
}

// ============================================================================
// tmpBufSize should be non-zero
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tmp_buf_size_nonzero)
{
    auto ctx = MakeTilingCtx(
        {8, 64}, ge::DT_FLOAT, {8}, {8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_GT(td.tmpBufSize, 0UL);
}

// ============================================================================
// Workspace size should be set
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, workspace_is_set)
{
    auto ctx = MakeTilingCtx(
        {8, 64}, ge::DT_FLOAT, {8}, {8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    // WS_SYS_SIZE = 0, so workspace[0] should be 0
    if (!info.workspaceSizes.empty()) {
        EXPECT_EQ(info.workspaceSizes[0], 0UL);
    }
}

// ============================================================================
// ============================================================================
// ===  Iteration 2: Tiling branch coverage (TK1 AR col-split, TK2 ARA)  =====
// ============================================================================
// ============================================================================

// ============================================================================
// TK1: AR col-split - large R forces col-split mode
// input: [2, 100000], axis=[1], keepdim=false, FP32
// rLength=100000, rLengthAlign=100000 (already 8-aligned)
// ubNeeded = 2*100000*4 + 2*32 + 4096 = 804160 > 393216 => col-split
// chunkR calculation (FP32, elemPerBlock=8):
//   overhead = 32 + 4096 = 4128
//   chunkR = FloorAlign((393216-4128)/4, 8) = FloorAlign(97272, 8) = 97272
//   After 3 refinement iterations, tmpBufSize=4096 (constant), chunkR stays 97272
//   chunkR(97272) < rLength(100000), so chunkR = 97272
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_fp32_large_r_colsplit)
{
    auto ctx = MakeTilingCtx(
        {2, 100000}, ge::DT_FLOAT, {2}, {2}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 100000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    // Must be col-split mode
    EXPECT_GT(td.chunkR, 0UL);         // chunkR > 0 indicates col-split
    EXPECT_LT(td.chunkR, td.rLength);  // chunkR < rLength (split happened)
    // chunkR must be 8-aligned (FP32 elemPerBlock)
    EXPECT_EQ(td.chunkR % 8, 0UL);
    EXPECT_EQ(td.countResult, 100000);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 100000.0f);
}

// ============================================================================
// TK1: AR col-split - verify chunkR precise value (FP32)
// input: [1, 100000], axis=[1], FP32
// Iter3: tmpBufSize is now dynamically computed via ComputeReduceBufSize.
// For count=96720 (FP32), blocks = ceil(96720/64) = 1512, bytes = 1512*4 = 6048
// aligned32 = 6048, + 256 = 6304. So tmpBufSize = 6304.
// chunkR refinement converges at 96720 (8-aligned).
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_fp32_chunkr_precise_value)
{
    auto ctx = MakeTilingCtx(
        {1, 100000}, ge::DT_FLOAT, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // Iter3 expected value (dynamic tmpBufSize)
    EXPECT_EQ(td.chunkR, 96720UL);
    EXPECT_EQ(td.rLength, 100000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR % 8, 0UL);  // 8-aligned for FP32
    EXPECT_LT(td.chunkR, td.rLength);
    // tmpBufSize should be dynamically computed, not hardcoded 4096
    EXPECT_EQ(td.tmpBufSize, 6304UL);
}

// ============================================================================
// TK1: AR col-split - numChunks calculation
// input: [1, 200000], axis=[1], FP32
// chunkR = 97272, rLength = 200000
// numChunks = CeilDiv(200000, 97272) = 3
// (The kernel uses rLength/chunkR; we verify the tiling params are consistent)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_fp32_numchunks_calculation)
{
    auto ctx = MakeTilingCtx(
        {1, 200000}, ge::DT_FLOAT, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_LT(td.chunkR, td.rLength);
    // Verify numChunks = CeilDiv(rLength, chunkR) >= 2
    uint64_t numChunks = (td.rLength + td.chunkR - 1) / td.chunkR;
    EXPECT_GE(numChunks, 2UL);
    // Verify chunkR * (numChunks - 1) < rLength
    EXPECT_LT(td.chunkR * (numChunks - 1), td.rLength);
}

// ============================================================================
// TK1: AR col-split with FP16 - larger elemPerBlock
// FP16: typeSize=2, elemPerBlock=16
// rLength=200000, rLengthAlign=200000
// ubNeeded = 2*200000*2 + 2*32 + 4096 = 804160 > 393216 => col-split
// overhead = 32 + 4096 = 4128
// chunkR = FloorAlign((393216-4128)/2, 16) = FloorAlign(194544, 16) = 194544
// 194544 < 200000, so chunkR = 194544
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_fp16_colsplit)
{
    auto ctx = MakeTilingCtx(
        {3, 200000}, ge::DT_FLOAT16, {3}, {3}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 3UL);
    EXPECT_EQ(td.rLength, 200000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_LT(td.chunkR, td.rLength);
    // chunkR must be 16-aligned (FP16 elemPerBlock)
    EXPECT_EQ(td.chunkR % 16, 0UL);
}

// ============================================================================
// TK1: AR col-split with multi-core
// input: [80, 100000], axis=[1], FP32
// a1Length=80, usedCoreNum should be >1, split along A1 rows
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_fp32_colsplit_multicore)
{
    auto ctx = MakeTilingCtx(
        {80, 100000}, ge::DT_FLOAT, {80}, {80}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 80UL);
    EXPECT_EQ(td.rLength, 100000UL);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_GT(td.usedCoreNum, 1);
    // tilesPerCore * (usedCoreNum-1) + tailCoreTiles = a1Length
    EXPECT_EQ(td.tilesPerCore * (td.usedCoreNum - 1) + td.tailCoreTiles, td.a1Length);
}

// ============================================================================
// TK1: Boundary - R exactly at threshold (FP32)
// Threshold: ubNeeded = 2 * rLengthAlign * 4 + 64 + 4096 <= 393216
//   => rLengthAlign <= (393216 - 4160) / 8 = 48632
// So R=48632 should be AR full-load (chunkR=0), R=48640 should be col-split
// (48640 is the next 8-aligned value > 48632)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_fp32_boundary_fullload)
{
    // R=48632 => ubNeeded = 2*48632*4 + 64 + 4096 = 389216 + 4160 = 393216 (exactly fits)
    // Actually let's compute: 2*48632*4 = 389056, 389056 + 64 + 4096 = 393216
    // Wait: 48632 * 8 = 389056, + 64 + 4096 = 393216 = ubSize => fits
    auto ctx = MakeTilingCtx(
        {1, 48632}, ge::DT_FLOAT, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // Should be full-load (chunkR = 0)
    EXPECT_EQ(td.chunkR, 0UL);
    EXPECT_EQ(td.rLength, 48632UL);
}

TEST_F(ReduceMeanWithCountTilingTest, tk1_fp32_boundary_colsplit)
{
    // R=48640 (next 8-aligned value) => rLengthAlign = 48640
    // ubNeeded = 2*48640*4 + 64 + 4096 = 389120 + 4160 = 393280 > 393216
    auto ctx = MakeTilingCtx(
        {1, 48640}, ge::DT_FLOAT, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // Should be col-split (chunkR > 0)
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_EQ(td.rLength, 48640UL);
}

// ============================================================================
// TK1: Large R with merged multi-axis reduction (still AR mode)
// input: [4, 500, 200], axis=[1, 2], keepdim=false
// tags: [A, R, R], merged: [A:4, R:100000]
// a1=4, r=100000, a0=1 => AR col-split
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_multiaxis_merge_to_colsplit)
{
    auto ctx = MakeTilingCtx(
        {4, 500, 200}, ge::DT_FLOAT, {4}, {4}, {1, 2}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 100000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_EQ(td.countResult, 100000);
}

// ============================================================================
// TK2: ARA full-load - tileA0Len calculation
// input: [4, 6, 8], axis=[1], keepdim=false
// merged: [A:4, R:6, A:8] => a1=4, r=6, a0=8
// FP32: a0TileBase = 32/4 = 8
// Trying candidate = 8:
//   candidateBytes=32, alignedBytes=32, alignedCols=8
//   inBufSize=2*6*8*4=384, outBufSize=2*8*4=64
//   countBufSize=ceil(8*8/32)*32=64, tmpBufSize=4096
//   totalUB=384+64+4096+64=4608 <= 393216 => OK, bestTileA0=8
// tileA0Len=8 (= a0Length, so no further splitting needed)
// a0Outer = CeilDiv(8, 8) = 1
// totalTiles = 4 * 1 = 4
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_fp32_basic_ara_tiling)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 8}, ge::DT_FLOAT, {4, 8}, {4, 8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 8UL);
    EXPECT_EQ(td.tileA0Len, 8UL);     // a0 fits entirely
    EXPECT_EQ(td.chunkR, 0UL);        // Not AR col-split
    EXPECT_EQ(td.countResult, 6);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 6.0f);
}

// ============================================================================
// TK2: ARA full-load - a0Outer calculation with larger a0
// input: [2, 10, 128], axis=[1], keepdim=false
// merged: [A:2, R:10, A:128] => a1=2, r=10, a0=128
// FP32: a0TileBase = 8
// Search for best tileA0Len:
//   candidate=8: alignedCols=8, inBuf=2*10*8*4=640, outBuf=64, countBuf=64, tmp=4096
//     total=4864 => OK
//   candidate=16: alignedCols=16, inBuf=1280, outBuf=128, countBuf=128, tmp=4096
//     total=5632 => OK
//   ... continues up to candidate=128
//   candidate=128: alignedCols=128, inBuf=10240, outBuf=1024, countBuf=1024, tmp=4096
//     total=16384 => OK
// tileA0Len=128 (all fits), a0Outer=CeilDiv(128,128)=1
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_fp32_a0outer_is_1_when_fits)
{
    auto ctx = MakeTilingCtx(
        {2, 10, 128}, ge::DT_FLOAT, {2, 128}, {2, 128}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 10UL);
    EXPECT_EQ(td.a0Length, 128UL);
    // a0 should all fit in UB
    EXPECT_EQ(td.tileA0Len, 128UL);
    EXPECT_EQ(td.countResult, 10);
}

// ============================================================================
// TK2: ARA full-load - tileA0Len < a0Length (needs A0 tiling)
// input: [2, 1000, 4096], axis=[1], keepdim=false
// merged: [A:2, R:1000, A:4096] => a1=2, r=1000, a0=4096
// FP32: a0TileBase = 8
// For candidate c:
//   alignedCols = CeilAlign(c*4, 32)/4 = c (since c is multiple of 8)
//   inBuf = 2 * 1000 * c * 4 = 8000*c
//   outBuf = 2 * c * 4 = 8*c
//   countBuf = ceil(c*8/32)*32 = c*8 (when c>=4)
//   tmpBuf = 4096
//   total = 8000c + 8c + 8c + 4096 = 8016c + 4096
//   8016c + 4096 <= 393216 => c <= (393216-4096)/8016 = 48.5 => c=48
// tileA0Len = 48
// a0Outer = CeilDiv(4096, 48) = 86
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_fp32_a0_tiling_needed)
{
    auto ctx = MakeTilingCtx(
        {2, 1000, 4096}, ge::DT_FLOAT, {2, 4096}, {2, 4096}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 1000UL);
    EXPECT_EQ(td.a0Length, 4096UL);
    // tileA0Len should be < a0Length
    EXPECT_GT(td.tileA0Len, 0UL);
    EXPECT_LT(td.tileA0Len, td.a0Length);
    // tileA0Len should be a multiple of a0TileBase (8 for FP32)
    EXPECT_EQ(td.tileA0Len % 8, 0UL);
    // Expected: 48
    EXPECT_EQ(td.tileA0Len, 48UL);
    // a0Outer = CeilDiv(4096, 48) = 86
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    EXPECT_EQ(a0Outer, 86UL);
    // totalTiles = a1Length * a0Outer = 2 * 86 = 172
    uint64_t totalTiles = td.a1Length * a0Outer;
    EXPECT_EQ(totalTiles, 172UL);
    EXPECT_EQ(td.countResult, 1000);
}

// ============================================================================
// TK2: ARA full-load - alignedCols for non-block-aligned a0
// input: [3, 8, 10], axis=[1], keepdim=false
// merged: [A:3, R:8, A:10] => a1=3, r=8, a0=10
// FP32: a0TileBase = 8
// candidate=8: candidateBytes=32, alignedBytes=32, alignedCols=8
//   inBuf=2*8*8*4=512, outBuf=64, countBuf=64, tmp=4096 => total=4736 <= 393216
// candidate=10 NOT tested because loop increments by a0TileBase=8, so next is 16
// candidate=16: candidateBytes=64, alignedBytes=64, alignedCols=16
//   inBuf=2*8*16*4=1024, outBuf=128, countBuf=128, tmp=4096 => total=5376 <= 393216
// ... but 16 > a0Length(10), so loop stops at 10
// Wait: loop condition is `candidate <= a0Length`, and candidate goes 8, 16
// candidate=16 > 10, so loop breaks after bestTileA0=8
// Then: tileA0Len = 8, capped: 8 < 10, so stays 8
// a0Outer = CeilDiv(10, 8) = 2
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_fp32_alignedcols_non_aligned_a0)
{
    auto ctx = MakeTilingCtx(
        {3, 8, 10}, ge::DT_FLOAT, {3, 10}, {3, 10}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 3UL);
    EXPECT_EQ(td.rLength, 8UL);
    EXPECT_EQ(td.a0Length, 10UL);
    EXPECT_EQ(td.tileA0Len, 8UL);
    // a0Outer = CeilDiv(10, 8) = 2
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    EXPECT_EQ(a0Outer, 2UL);
    // totalTiles = 3 * 2 = 6
    EXPECT_EQ(td.countResult, 8);
}

// ============================================================================
// TK2: ARA full-load - FP16 a0TileBase = 16
// input: [4, 6, 24], axis=[1], keepdim=false
// merged: [A:4, R:6, A:24] => a1=4, r=6, a0=24
// FP16: a0TileBase = 32/2 = 16
// candidate=16: candidateBytes=32, alignedBytes=32, alignedCols=16
//   inBuf=2*6*16*2=384, outBuf=2*16*2=64, countBuf=max(4,16)*8=128 ceil32=128, tmp=4096
//   total=4672 <= 393216 => OK
// candidate=24 is not tested (loop increments by 16, next is 32)
// candidate=32 > 24, breaks => bestTileA0=16
// tileA0Len = 16, a0Outer = CeilDiv(24, 16) = 2
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_fp16_a0tilebase)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 24}, ge::DT_FLOAT16, {4, 24}, {4, 24}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 24UL);
    // FP16 a0TileBase = 16
    EXPECT_EQ(td.tileA0Len, 16UL);
    // a0Outer = CeilDiv(24, 16) = 2
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    EXPECT_EQ(a0Outer, 2UL);
}

// ============================================================================
// TK2: ARA mode multi-core tiling
// input: [8, 4, 256], axis=[1], keepdim=false
// merged: [A:8, R:4, A:256] => a1=8, r=4, a0=256
// FP32: a0TileBase=8
// Many candidates fit; a0Outer likely 1-4
// totalTiles = a1 * a0Outer; multi-core split distributes totalTiles
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_fp32_multicore_tiling)
{
    auto ctx = MakeTilingCtx(
        {8, 4, 256}, ge::DT_FLOAT, {8, 256}, {8, 256}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 8UL);
    EXPECT_EQ(td.rLength, 4UL);
    EXPECT_EQ(td.a0Length, 256UL);
    EXPECT_GT(td.tileA0Len, 0UL);
    EXPECT_GT(td.usedCoreNum, 0);
    // tilesPerCore*(usedCoreNum-1)+tailCoreTiles = totalTiles = a1*a0Outer
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    uint64_t totalTiles = td.a1Length * a0Outer;
    EXPECT_EQ(td.tilesPerCore * (td.usedCoreNum - 1) + td.tailCoreTiles, totalTiles);
}

// ============================================================================
// Scene auto-detection: non-innermost axis reduction -> ARA
// input: [4, 6, 8], axis=[0], keepdim=false
// tags: [R, A, A], merged: [R:4, A:48]
// finalMerged[0]=R, finalMerged[1]=A => a1=1, r=4, a0=48 => ARA mode
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, scene_non_innermost_axis_is_ara)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 8}, ge::DT_FLOAT, {6, 8}, {6, 8}, {0}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 4UL);
    EXPECT_EQ(td.a0Length, 48UL);
    EXPECT_GT(td.tileA0Len, 0UL); // ARA mode has tileA0Len > 0
    EXPECT_EQ(td.countResult, 4);
}

// ============================================================================
// Scene auto-detection: middle axis reduction -> ARA pattern
// input: [3, 5, 7, 4], axis=[1, 2], keepdim=false
// tags: [A, R, R, A], merged: [A:3, R:35, A:4]
// ARA pattern: a1=3, r=35, a0=4
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, scene_middle_axes_reduce_ara)
{
    auto ctx = MakeTilingCtx(
        {3, 5, 7, 4}, ge::DT_FLOAT, {3, 4}, {3, 4}, {1, 2}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 3UL);
    EXPECT_EQ(td.rLength, 35UL);
    EXPECT_EQ(td.a0Length, 4UL);
    EXPECT_GT(td.tileA0Len, 0UL);
    EXPECT_EQ(td.countResult, 35);
}

// ============================================================================
// Scene auto-detection: complex multi-axis with merge -> RAR path
// input: [2, 3, 4, 5, 6], axis=[0, 2, 4], keepdim=false
// tags: [R, A, R, A, R], merged/finalMerged: [R:2, A:3, R:4, A:5, R:6]
// Iter3: size=5 starting with R-A-R -> RAR branch is chosen:
//   a1=1, rLength=finalMerged[0]*finalMerged[2]=2*4=8, a0=finalMerged[1]=3
//   Trailing absorb: i=3 (A:5) -> a0*=5 =15;  i=4 (R:6) -> rLength*=6 =48
// Final: a1=1, r=48, a0=15 => ARA mode
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, scene_complex_multiaxis_rar_path)
{
    auto ctx = MakeTilingCtx(
        {2, 3, 4, 5, 6}, ge::DT_FLOAT, {3, 5}, {3, 5}, {0, 2, 4}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 48UL);
    EXPECT_EQ(td.a0Length, 15UL);
    EXPECT_EQ(td.countResult, 48);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 48.0f);
    // ARA mode: tileA0Len > 0
    EXPECT_GT(td.tileA0Len, 0UL);
}

// ============================================================================
// Scene: purely-reduce tensor (all dims are R)
// input: [100000], axis=[0], keepdim=false
// merged: [R:100000], a1=1, r=100000, a0=1 => AR mode
// R is large enough to trigger col-split
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, scene_pure_reduce_large_r)
{
    auto ctx = MakeTilingCtx(
        {100000}, ge::DT_FLOAT, {}, {}, {0}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 100000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_GT(td.chunkR, 0UL); // col-split
    EXPECT_EQ(td.countResult, 100000);
}

// ============================================================================
// TK0 vs TK1 transition: confirm AR full-load when R fits in UB
// input: [10, 1024], axis=[1], FP32
// rLengthAlign=1024, ubNeeded=2*1024*4+64+4096=12352 < 393216 => full-load
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk0_fullload_medium_r)
{
    auto ctx = MakeTilingCtx(
        {10, 1024}, ge::DT_FLOAT, {10}, {10}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 10UL);
    EXPECT_EQ(td.rLength, 1024UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 0UL); // Full-load, no col-split
}

// ============================================================================
// TK2: ARA with keepdim=true
// input: [4, 6, 8], axis=[1], keepdim=true
// output shape: [4, 1, 8], outputLength = 4*1*8 = 32
// merged: [A:4, R:6, A:8] => a1=4, r=6, a0=8
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_ara_keepdim_true)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 8}, ge::DT_FLOAT, {4, 1, 8}, {4, 1, 8}, {1}, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 8UL);
    EXPECT_EQ(td.outputLength, 32UL);
    EXPECT_GT(td.tileA0Len, 0UL);
}

// ============================================================================
// TK1: AR col-split with BF16
// BF16: typeSize=2, elemPerBlock=16
// input: [2, 200000], axis=[1], keepdim=false
// ubNeeded = 2*200000*2 + 64 + 4096 = 804160 > 393216 => col-split
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk1_bf16_colsplit)
{
    auto ctx = MakeTilingCtx(
        {2, 200000}, ge::DT_BF16, {2}, {2}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 200000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_GT(td.chunkR, 0UL);
    // chunkR must be 16-aligned (BF16 elemPerBlock)
    EXPECT_EQ(td.chunkR % 16, 0UL);
    EXPECT_EQ(td.countResult, 200000);
}

// ============================================================================
// TK2: ARA mode - verify tileA0Len capping to a0Length
// input: [1, 4, 8], axis=[1], keepdim=false
// merged: [A:1, R:4, A:8] => but A:1 is removed (size-1 dim removal)
// Actually: rank=3, shapeDims=[1,4,8], tags=[A,R,A]
// size-1 removal: dim0 is A with size 1 => removed
// merged: [(R:4), (A:8)], finalMerged: [R:4, A:8]
// finalMerged size=2, [0]=R, [1]=A => a1=1, r=4, a0=8 => ARA mode
// tileA0Len capped to min of computed and a0Length
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk2_ara_size1_a1_removed)
{
    auto ctx = MakeTilingCtx(
        {1, 4, 8}, ge::DT_FLOAT, {1, 8}, {1, 8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 4UL);
    EXPECT_EQ(td.a0Length, 8UL);
    EXPECT_GT(td.tileA0Len, 0UL);
    EXPECT_LE(td.tileA0Len, td.a0Length);
}

// ============================================================================
// ============================================================================
// ===  Iteration 3: Full coverage (FP16 TK3/TK4/TK5, BF16 TK6/TK7/TK8,  =====
// ===  dynamic tmpBufSize, cast/fp32 buffer accounting, merging boundary) ===
// ============================================================================
// ============================================================================

// ============================================================================
// TK3: FP16 AR full-load
// input: [10, 32], axis=[1], FP16 (typeSize=2, elemPerBlock=16)
// rLengthAlign = 32, rLengthAlignFP32 = 32
// needCast=true
//   inBuf  = 2*32*2 = 128
//   outBuf = 64
//   tmpBuf = ComputeReduceBufSize(32, 4) = max(ceil(32/64)*64*4+256, 4096) = 4096
//   castBuf = 32*4 = 128
//   fp32Res = 32
// ubNeeded = 128+64+4096+128+32 = 4448 <= 393216 => full-load (chunkR=0)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk3_fp16_ar_fullload_small)
{
    auto ctx = MakeTilingCtx(
        {10, 32}, ge::DT_FLOAT16, {10}, {10}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 10UL);
    EXPECT_EQ(td.rLength, 32UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 0UL);           // Full-load
    EXPECT_EQ(td.tmpBufSize, 4096UL);    // Dynamic tmpBufSize (minimum floor)
    EXPECT_EQ(td.countResult, 32);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 32.0f);
    EXPECT_EQ(td.outputLength, 10UL);
}

// ============================================================================
// TK3: FP16 AR full-load - medium R (1024)
// input: [4, 1024], axis=[1], FP16
// rLengthAlign=1024, inBuf=2*1024*2=4096, outBuf=64, tmpBuf=4096
// castBuf=1024*4=4096, fp32Res=32
// ubNeeded = 4096+64+4096+4096+32 = 12384 <= 393216 => full-load
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk3_fp16_ar_fullload_medium)
{
    auto ctx = MakeTilingCtx(
        {4, 1024}, ge::DT_FLOAT16, {4}, {4}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 1024UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 0UL);
    EXPECT_EQ(td.tmpBufSize, 4096UL);
    EXPECT_EQ(td.countResult, 1024);
}

// ============================================================================
// TK3: FP16 AR full-load - boundary (below col-split threshold)
// For FP16 with cast: ubNeeded ~ 2*rAlign*2 + 64 + 4096 + rAlign*4 + 32 = 8*rAlign + 4192
// 8*rAlign + 4192 <= 393216 => rAlign <= 48628 => round to 48624 (16-aligned)
// R=45056 (16-aligned, inside): ubNeeded = 8*45056+4192 = 364640 < 393216 => full-load
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk3_fp16_boundary_fullload)
{
    auto ctx = MakeTilingCtx(
        {1, 45056}, ge::DT_FLOAT16, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.chunkR, 0UL);  // Full-load
    EXPECT_EQ(td.rLength, 45056UL);
}

// ============================================================================
// TK4: FP16 AR col-split - boundary transition
// R=49152 (16-aligned, just above threshold) => col-split
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk4_fp16_boundary_colsplit)
{
    auto ctx = MakeTilingCtx(
        {1, 49152}, ge::DT_FLOAT16, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_EQ(td.rLength, 49152UL);
    // chunkR must be 16-aligned (FP16 elemPerBlock)
    EXPECT_EQ(td.chunkR % 16, 0UL);
}

// ============================================================================
// TK4: FP16 AR col-split - large R with precise chunkR
// input: [3, 200000], axis=[1], FP16
// Iter3 dynamic computation:
//   perElemBytes = 2 + 4 = 6
//   Iterative refinement converges at chunkR=64800
//   tmpBufSize for count=64800 FP32: ceil(64800/64)*4 = 1013*4=4052, aligned32=4064, +256=4320
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk4_fp16_colsplit_precise)
{
    auto ctx = MakeTilingCtx(
        {3, 200000}, ge::DT_FLOAT16, {3}, {3}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 3UL);
    EXPECT_EQ(td.rLength, 200000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 64800UL);         // Precise value
    EXPECT_EQ(td.chunkR % 16, 0UL);        // 16-aligned for FP16
    EXPECT_LT(td.chunkR, td.rLength);
    // tmpBufSize dynamically > 4096 (for large chunkR, cast path)
    EXPECT_EQ(td.tmpBufSize, 4320UL);
    EXPECT_EQ(td.countResult, 200000);
}

// ============================================================================
// TK4: FP16 AR col-split - multi-core
// input: [40, 200000], FP16
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk4_fp16_colsplit_multicore)
{
    auto ctx = MakeTilingCtx(
        {40, 200000}, ge::DT_FLOAT16, {40}, {40}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 40UL);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_EQ(td.tilesPerCore * (td.usedCoreNum - 1) + td.tailCoreTiles, td.a1Length);
}

// ============================================================================
// TK5: FP16 ARA full-load - a0TileBase = 16
// input: [4, 6, 24], axis=[1], FP16
// merged: [A:4, R:6, A:24], a1=4, r=6, a0=24
// a0TileBase=16, candidate=16: alignedCols=16
//   inBuf=2*6*16*2=384, outBuf=64, castBuf=6*16*4=384, fp32Res=64, countBuf=128, tmp=4096
//   total=5120 <= UB
// candidate=32 > a0=24, breaks => bestTileA0=16
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk5_fp16_ara_fullload)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 24}, ge::DT_FLOAT16, {4, 24}, {4, 24}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 24UL);
    EXPECT_EQ(td.tileA0Len, 16UL);       // FP16 a0TileBase=16
    EXPECT_EQ(td.chunkR, 0UL);
    EXPECT_EQ(td.countResult, 6);
}

// ============================================================================
// TK5: FP16 ARA - large A0 fits entirely
// input: [2, 100, 256], axis=[1], FP16
// a0=256, all candidates fit: final tileA0Len=256, a0Outer=1
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk5_fp16_ara_large_a0_fits)
{
    auto ctx = MakeTilingCtx(
        {2, 100, 256}, ge::DT_FLOAT16, {2, 256}, {2, 256}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 100UL);
    EXPECT_EQ(td.a0Length, 256UL);
    EXPECT_EQ(td.tileA0Len, 256UL);     // All fits
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    EXPECT_EQ(a0Outer, 1UL);
}

// ============================================================================
// TK5: FP16 ARA - a0 tiling needed (cast buffer doubles UB cost)
// input: [2, 1000, 4096], axis=[1], FP16
// For candidate c: inBuf=2*1000*c*2=4000c, castBuf=1000*c*4=4000c
//   total dominating terms ~ 8016c + overhead => c <= 48
// Same as FP32 case: tileA0Len = 48
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk5_fp16_ara_a0_tiling)
{
    auto ctx = MakeTilingCtx(
        {2, 1000, 4096}, ge::DT_FLOAT16, {2, 4096}, {2, 4096}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 1000UL);
    EXPECT_EQ(td.a0Length, 4096UL);
    EXPECT_EQ(td.tileA0Len, 48UL);
    EXPECT_EQ(td.tileA0Len % 16, 0UL);   // FP16 a0TileBase=16
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    EXPECT_EQ(a0Outer, 86UL);
}

// ============================================================================
// TK5: FP16 ARA keepdim=true
// input: [4, 6, 24], axis=[1], keepdim=true
// output shape: [4, 1, 24], outputLength = 4*1*24 = 96
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk5_fp16_ara_keepdim_true)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 24}, ge::DT_FLOAT16, {4, 1, 24}, {4, 1, 24}, {1}, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 24UL);
    EXPECT_EQ(td.outputLength, 96UL);
    EXPECT_GT(td.tileA0Len, 0UL);
}

// ============================================================================
// TK6: BF16 AR full-load
// input: [10, 32], axis=[1], BF16 (same footprint as FP16)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk6_bf16_ar_fullload_small)
{
    auto ctx = MakeTilingCtx(
        {10, 32}, ge::DT_BF16, {10}, {10}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 10UL);
    EXPECT_EQ(td.rLength, 32UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 0UL);
    EXPECT_EQ(td.tmpBufSize, 4096UL);
    EXPECT_EQ(td.countResult, 32);
}

// ============================================================================
// TK6: BF16 AR full-load - medium R
// input: [4, 1024], axis=[1], BF16 (same layout as FP16)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk6_bf16_ar_fullload_medium)
{
    auto ctx = MakeTilingCtx(
        {4, 1024}, ge::DT_BF16, {4}, {4}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 1024UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 0UL);
}

// ============================================================================
// TK7: BF16 AR col-split - large R with precise chunkR
// input: [2, 200000], axis=[1], BF16
// Same as FP16 col-split: chunkR=64800, tmpBufSize=4320
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk7_bf16_colsplit_precise)
{
    auto ctx = MakeTilingCtx(
        {2, 200000}, ge::DT_BF16, {2}, {2}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 200000UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.chunkR, 64800UL);
    EXPECT_EQ(td.chunkR % 16, 0UL);      // 16-aligned BF16
    EXPECT_EQ(td.tmpBufSize, 4320UL);
    EXPECT_EQ(td.countResult, 200000);
}

// ============================================================================
// TK7: BF16 AR col-split - multicore
// input: [80, 100000], axis=[1], BF16
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk7_bf16_colsplit_multicore)
{
    auto ctx = MakeTilingCtx(
        {80, 100000}, ge::DT_BF16, {80}, {80}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 80UL);
    EXPECT_EQ(td.rLength, 100000UL);
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_EQ(td.chunkR % 16, 0UL);
    EXPECT_GT(td.usedCoreNum, 1);
    EXPECT_EQ(td.tilesPerCore * (td.usedCoreNum - 1) + td.tailCoreTiles, td.a1Length);
}

// ============================================================================
// TK8: BF16 ARA full-load
// input: [4, 6, 24], axis=[1], BF16 (same footprint as FP16 ARA)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk8_bf16_ara_fullload)
{
    auto ctx = MakeTilingCtx(
        {4, 6, 24}, ge::DT_BF16, {4, 24}, {4, 24}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 6UL);
    EXPECT_EQ(td.a0Length, 24UL);
    EXPECT_EQ(td.tileA0Len, 16UL);
    EXPECT_EQ(td.chunkR, 0UL);
}

// ============================================================================
// TK8: BF16 ARA - a0 tiling needed
// input: [2, 1000, 4096], axis=[1], BF16 (same as FP16)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk8_bf16_ara_a0_tiling)
{
    auto ctx = MakeTilingCtx(
        {2, 1000, 4096}, ge::DT_BF16, {2, 4096}, {2, 4096}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 1000UL);
    EXPECT_EQ(td.a0Length, 4096UL);
    EXPECT_EQ(td.tileA0Len, 48UL);
    EXPECT_EQ(td.tileA0Len % 16, 0UL);
}

// ============================================================================
// TK8: BF16 ARA - keepdim and multi-core
// input: [16, 8, 128], axis=[1], BF16, keepdim=false
// merged: [A:16, R:8, A:128], a1=16, r=8, a0=128 => ARA
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tk8_bf16_ara_multicore)
{
    auto ctx = MakeTilingCtx(
        {16, 8, 128}, ge::DT_BF16, {16, 128}, {16, 128}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 16UL);
    EXPECT_EQ(td.rLength, 8UL);
    EXPECT_EQ(td.a0Length, 128UL);
    EXPECT_GT(td.tileA0Len, 0UL);
    EXPECT_GT(td.usedCoreNum, 0);
    // totalTiles = a1 * a0Outer
    uint64_t a0Outer = (td.a0Length + td.tileA0Len - 1) / td.tileA0Len;
    uint64_t totalTiles = td.a1Length * a0Outer;
    EXPECT_EQ(td.tilesPerCore * (td.usedCoreNum - 1) + td.tailCoreTiles, totalTiles);
}

// ============================================================================
// tmpBufSize dynamic: verify it is floored at 4096 for small R (not zero)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tmp_buf_size_small_r_floor)
{
    auto ctx = MakeTilingCtx(
        {8, 16}, ge::DT_FLOAT, {8}, {8}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // Small count, still at least 4096
    EXPECT_EQ(td.tmpBufSize, 4096UL);
}

// ============================================================================
// tmpBufSize dynamic: verify growth with larger chunk (FP32 col-split with R=500000)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tmp_buf_size_grows_with_chunk)
{
    // Small-R full-load case: tmp=4096
    auto ctxSmall = MakeTilingCtx(
        {1, 256}, ge::DT_FLOAT, {1}, {1}, {1}, false);
    TilingInfo infoSmall;
    ASSERT_TRUE(ExecuteTiling(ctxSmall, infoSmall));
    auto tdSmall = ExtractTilingData(infoSmall);
    EXPECT_EQ(tdSmall.tmpBufSize, 4096UL);

    // Large col-split case: tmp should grow beyond 4096 since chunkR >= ~96720
    auto ctxLarge = MakeTilingCtx(
        {1, 100000}, ge::DT_FLOAT, {1}, {1}, {1}, false);
    TilingInfo infoLarge;
    ASSERT_TRUE(ExecuteTiling(ctxLarge, infoLarge));
    auto tdLarge = ExtractTilingData(infoLarge);
    // chunkR=96720 => tmpBufSize=6304 > 4096
    EXPECT_GT(tdLarge.tmpBufSize, 4096UL);
    EXPECT_EQ(tdLarge.tmpBufSize, 6304UL);
}

// ============================================================================
// tmpBufSize dynamic: FP16 path uses FP32-based size (count = rLengthAlignFP32)
// input: [1, 100000], FP16 => col-split, tmp based on chunkR aligned to FP32
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, tmp_buf_size_fp16_uses_fp32_count)
{
    auto ctx = MakeTilingCtx(
        {1, 100000}, ge::DT_FLOAT16, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // FP16 with cast, col-split: chunkR ~ 64800, tmp ~ 4320
    EXPECT_GT(td.chunkR, 0UL);
    EXPECT_GT(td.tmpBufSize, 4096UL);
}

// ============================================================================
// UB accounting: FP16/BF16 full-load fits because castBuf is accounted
// input: [1, 40960] FP16 - should full-load (ubNeeded ~ 332K < 393K)
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, fp16_ub_accounting_castbuf_counted)
{
    auto ctx = MakeTilingCtx(
        {1, 40960}, ge::DT_FLOAT16, {1}, {1}, {1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    // Should still full-load (ubNeeded < UB_SIZE)
    EXPECT_EQ(td.chunkR, 0UL);
    EXPECT_EQ(td.rLength, 40960UL);
}

// ============================================================================
// Axis merging: scalar-like input, rank=1 with all-axis reduce
// input: [5], axis=[] or [0] => all reduce, countResult=5
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, axis_merging_rank1_fp16)
{
    auto ctx = MakeTilingCtx(
        {5}, ge::DT_FLOAT16, {}, {}, {}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 5UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 5);
}

// ============================================================================
// Axis merging: rank=1 BF16
// input: [100], axis=[0]
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, axis_merging_rank1_bf16)
{
    auto ctx = MakeTilingCtx(
        {100}, ge::DT_BF16, {}, {}, {0}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 100UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 100);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 100.0f);
}

// ============================================================================
// Axis merging: BF16 with size-1 dims (verify dtype doesn't affect merging)
// input: [1, 8, 1, 16], axis=[1, 3], keepdim=false
// shapeDims [1,8,1,16], tags [A, R, A, R]
// Size-1 removal: [R:8, R:16] => merged [R:128]
// a1=1, r=128, a0=1
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, axis_merging_bf16_with_size1)
{
    auto ctx = MakeTilingCtx(
        {1, 8, 1, 16}, ge::DT_BF16, {1, 1}, {1, 1}, {1, 3}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 128UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 128);
}

// ============================================================================
// Axis merging boundary: negative axis with FP16
// input: [4, 8], axis=[-1], FP16 => same as axis=[1]
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, negative_axis_fp16)
{
    auto ctx = MakeTilingCtx(
        {4, 8}, ge::DT_FLOAT16, {4}, {4}, {-1}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 4UL);
    EXPECT_EQ(td.rLength, 8UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 8);
}

// ============================================================================
// Axis merging boundary: single element FP16 / BF16
// input: [1], axis=[0]
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, single_element_fp16)
{
    auto ctx = MakeTilingCtx(
        {1}, ge::DT_FLOAT16, {}, {}, {0}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.countResult, 1);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f);
    EXPECT_EQ(td.outputLength, 1UL);
}

TEST_F(ReduceMeanWithCountTilingTest, single_element_bf16)
{
    auto ctx = MakeTilingCtx(
        {1}, ge::DT_BF16, {}, {}, {0}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.countResult, 1);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f);
    EXPECT_EQ(td.outputLength, 1UL);
}

// ============================================================================
// Axis merging boundary: 4D BF16 with ARA pattern
// input: [2, 3, 4, 5], axis=[1, 2], BF16
// tags: [A, R, R, A], merged: [A:2, R:12, A:5] => a1=2, r=12, a0=5 => ARA
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, axis_merging_bf16_4d_ara)
{
    auto ctx = MakeTilingCtx(
        {2, 3, 4, 5}, ge::DT_BF16, {2, 5}, {2, 5}, {1, 2}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 2UL);
    EXPECT_EQ(td.rLength, 12UL);
    EXPECT_EQ(td.a0Length, 5UL);
    EXPECT_EQ(td.countResult, 12);
    EXPECT_GT(td.tileA0Len, 0UL);
}

// ============================================================================
// Axis merging boundary: FP16 all-axis reduction
// input: [2, 3, 4], axis=[], keepdim=false, FP16
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, fp16_all_axis_reduction)
{
    auto ctx = MakeTilingCtx(
        {2, 3, 4}, ge::DT_FLOAT16, {}, {}, {}, false);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 24UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.countResult, 24);
    EXPECT_FLOAT_EQ(td.invCount, 1.0f / 24.0f);
}

// ============================================================================
// Axis merging boundary: BF16 keepdim=true all-axis
// input: [2, 3, 4], axis=[], keepdim=true => output shape [1,1,1], outputLength=1
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, bf16_all_axis_keepdim_true)
{
    auto ctx = MakeTilingCtx(
        {2, 3, 4}, ge::DT_BF16, {1, 1, 1}, {1, 1, 1}, {}, true);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(ctx, info));

    auto td = ExtractTilingData(info);
    EXPECT_EQ(td.a1Length, 1UL);
    EXPECT_EQ(td.rLength, 24UL);
    EXPECT_EQ(td.a0Length, 1UL);
    EXPECT_EQ(td.outputLength, 1UL);
    EXPECT_EQ(td.countResult, 24);
}

// ============================================================================
// Cross-dtype sanity: same shape, three dtypes produce same a1/r/a0
// input: [8, 64], axis=[1], verify a1/r/a0/countResult identical across FP32/FP16/BF16
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, cross_dtype_shape_invariant)
{
    TilingInfo infoFp32, infoFp16, infoBf16;
    auto ctxFp32 = MakeTilingCtx({8, 64}, ge::DT_FLOAT, {8}, {8}, {1}, false);
    auto ctxFp16 = MakeTilingCtx({8, 64}, ge::DT_FLOAT16, {8}, {8}, {1}, false);
    auto ctxBf16 = MakeTilingCtx({8, 64}, ge::DT_BF16, {8}, {8}, {1}, false);
    ASSERT_TRUE(ExecuteTiling(ctxFp32, infoFp32));
    ASSERT_TRUE(ExecuteTiling(ctxFp16, infoFp16));
    ASSERT_TRUE(ExecuteTiling(ctxBf16, infoBf16));

    auto td32 = ExtractTilingData(infoFp32);
    auto td16 = ExtractTilingData(infoFp16);
    auto tdBf = ExtractTilingData(infoBf16);

    EXPECT_EQ(td32.a1Length, td16.a1Length);
    EXPECT_EQ(td32.a1Length, tdBf.a1Length);
    EXPECT_EQ(td32.rLength, td16.rLength);
    EXPECT_EQ(td32.rLength, tdBf.rLength);
    EXPECT_EQ(td32.a0Length, td16.a0Length);
    EXPECT_EQ(td32.a0Length, tdBf.a0Length);
    EXPECT_EQ(td32.countResult, td16.countResult);
    EXPECT_EQ(td32.countResult, tdBf.countResult);
    EXPECT_FLOAT_EQ(td32.invCount, td16.invCount);
    EXPECT_FLOAT_EQ(td32.invCount, tdBf.invCount);
}

// ============================================================================
// Regression guard: every dtype/mode combination produces a sane tiling
// Verifies that a1, r, a0, countResult > 0 and multi-core split invariant holds
// ============================================================================
TEST_F(ReduceMeanWithCountTilingTest, sanity_all_dtype_AR_modes)
{
    std::vector<ge::DataType> dtypes = {ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16};
    for (auto dt : dtypes) {
        // AR full-load
        auto ctxFull = MakeTilingCtx({8, 64}, dt, {8}, {8}, {1}, false);
        TilingInfo infoFull;
        ASSERT_TRUE(ExecuteTiling(ctxFull, infoFull));
        auto tdFull = ExtractTilingData(infoFull);
        EXPECT_EQ(tdFull.a1Length, 8UL);
        EXPECT_EQ(tdFull.rLength, 64UL);
        EXPECT_EQ(tdFull.chunkR, 0UL);     // Full-load
        EXPECT_GT(tdFull.usedCoreNum, 0);

        // AR col-split
        auto ctxCol = MakeTilingCtx({2, 200000}, dt, {2}, {2}, {1}, false);
        TilingInfo infoCol;
        ASSERT_TRUE(ExecuteTiling(ctxCol, infoCol));
        auto tdCol = ExtractTilingData(infoCol);
        EXPECT_EQ(tdCol.a1Length, 2UL);
        EXPECT_EQ(tdCol.rLength, 200000UL);
        EXPECT_GT(tdCol.chunkR, 0UL);
        EXPECT_LT(tdCol.chunkR, tdCol.rLength);

        // ARA full-load
        auto ctxAra = MakeTilingCtx({4, 6, 8}, dt, {4, 8}, {4, 8}, {1}, false);
        TilingInfo infoAra;
        ASSERT_TRUE(ExecuteTiling(ctxAra, infoAra));
        auto tdAra = ExtractTilingData(infoAra);
        EXPECT_EQ(tdAra.a1Length, 4UL);
        EXPECT_EQ(tdAra.rLength, 6UL);
        EXPECT_EQ(tdAra.a0Length, 8UL);
        EXPECT_GT(tdAra.tileA0Len, 0UL);
    }
}

} // namespace ReduceMeanWithCountUT
