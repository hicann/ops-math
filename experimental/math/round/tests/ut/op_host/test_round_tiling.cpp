 /**
  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
  * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
  * CANN Open Software License Agreement Version 2.0 (the "License").
  * Please refer to the License for details. You may not use this file except in compliance with the License.
  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
  * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
  * See LICENSE in the root of the software repository for the full text of the License.
  */

 #include <iostream>
 #include <gtest/gtest.h>
 #include "round_tiling.h"
 #include "../../../op_kernel/round_tiling_data.h"
 #include "../../../op_kernel/round_tiling_key.h"
 #include "tiling_context_faker.h"
 #include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class RoundTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "RoundTiling SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "RoundTiling TearDown" << endl;
    }
};

TEST_F(RoundTiling, ascend9101_test_tiling_fp16_001)
{
    optiling::RoundCompileInfo compileInfo = {16}; 

    gert::TilingContextPara tilingContextPara(
        "Round",
        {
            {{{2,8}, {2,8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // input
        },
        {
            {{{2,8}, {2,8}}, ge::DT_FLOAT16, ge::FORMAT_ND}, // output
        },
        {
            gert::TilingContextPara::OpAttr("decimals", Ops::Math::AnyValue::CreateFrom<int64_t>(0))
        },
        &compileInfo
    );

    uint64_t expectTilingKey = 0; // 根据实际逻辑可能非零，若无 key 则为 0
    
    /*
        示例数据，本次以2*8维度数据为例，按照hosy侧经简单计算可得
        smallCoreDataNum-->16
        bigCoreDataNum----> 0
        finalBigTileNum---> 0
        finalSmallTileNum-> 1
        tileDataNum------->16368
        smallTailDataNum--->16
        bigTailDataNum---> 0
        tailBlockNum------> 0
        decimals---------->1
    */
    string expectTilingData = "16 0 0 1 16368 16 0 0 1 "; // 对应round_tiling_data.h中的RoundTilingData
    std::vector<size_t> expectWorkspaces = {16777216};  

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
