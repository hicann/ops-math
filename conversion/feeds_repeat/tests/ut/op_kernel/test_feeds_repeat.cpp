/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_host/feeds_repeat_tiling.h"
#include "data_utils.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

extern "C" __global__ __aicore__ void feeds_repeat(
    GM_ADDR feeds, GM_ADDR feeds_repeat_times, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);
class feeds_repeat_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "feeds_repeat_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "feeds_repeat_test TearDown\n" << endl;
    }
};

TEST_F(feeds_repeat_test, test_case_fp32_int32)
{
    optiling::FeedsRepeatCompileInfo compileInfo = {48, 196608};
    
    gert::TilingContextPara tilingContextPara("FeedsRepeat",
                                              {{{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                               {{{1,}, {1,}}, ge::DT_INT32, ge::FORMAT_ND}},
                                              {{{{1, 1, 1, 8}, {1, 1, 1, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},},
                                              {gert::TilingContextPara::OpAttr("output_feeds_size", Ops::Math::AnyValue::CreateFrom<int64_t>(1))},
                                              &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);


    size_t feeds_size = 1 * 1 * 1 * 8 * sizeof(float);
    size_t feeds_repeat_times_size = 1 * sizeof(int32_t);
    size_t y_size = 1 * 1 * 1 * 8 * sizeof(float);

    uint8_t* feeds = (uint8_t*)AscendC::GmAlloc(feeds_size);
    uint8_t* feeds_repeat_times = (uint8_t*)AscendC::GmAlloc(feeds_repeat_times_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    uint32_t blockDim = 1;
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(feeds_repeat, blockDim, feeds, feeds_repeat_times, y, workspace, tiling);

    AscendC::GmFree(feeds);
    AscendC::GmFree(feeds_repeat_times);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}