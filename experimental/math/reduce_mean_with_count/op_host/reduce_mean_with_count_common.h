/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

#ifndef OPS_MATH_REDUCE_MEAN_WITH_COUNT_COMMON_H
#define OPS_MATH_REDUCE_MEAN_WITH_COUNT_COMMON_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ops_reduce_mean_with_count {

template <typename AxisListPtrT>
inline std::vector<int64_t> NormalizeAxes(AxisListPtrT axisListPtr, int64_t rank)
{
    std::vector<int64_t> axes;
    if (axisListPtr == nullptr || axisListPtr->GetSize() == 0) {
        for (int64_t i = 0; i < rank; i++) {
            axes.push_back(i);
        }
    } else {
        const int64_t* axisData = axisListPtr->GetData();
        for (size_t i = 0; i < axisListPtr->GetSize(); i++) {
            int64_t ax = axisData[i];
            if (ax < 0) {
                ax += rank;
            }
            axes.push_back(ax);
        }
    }
    std::sort(axes.begin(), axes.end());
    axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
    return axes;
}

inline bool IsReduceDim(const std::vector<int64_t>& axes, int64_t d)
{
    for (size_t a = 0; a < axes.size(); a++) {
        if (axes[a] == d) {
            return true;
        }
    }
    return false;
}

} // namespace ops_reduce_mean_with_count

#endif // OPS_MATH_REDUCE_MEAN_WITH_COUNT_COMMON_H
