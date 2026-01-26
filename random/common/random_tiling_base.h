/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file random_tiling_base.h
 * \brief
 */
#ifndef RANDOM_TILING_BASE_H
#define RANDOM_TILING_BASE_H

#include <random>
#include <chrono>
#include <thread>
#include <cstdint>

namespace optiling {

static inline std::mt19937_64& GetGlobalRng() {
    static std::mt19937_64 rng([]() -> uint64_t {
        auto now =std::chrono::high_resolution_clock::now();
        uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
        ).count();

        seed ^= std::hash<std::thread::id>()(std::this_thread::get_id());
        return seed;
    }());

    return rng;
}


inline uint64_t New64() {
    return GetGlobalRng()();
}

} // namespace optiling
#endif // RANDOM_TILING_BASE_H