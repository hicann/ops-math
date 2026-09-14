# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Select the 950 entry instead of the default 910 hans_encode.cpp. The UT
# framework searches alternate entry names only in op_kernel subdirectories;
# this root-level apt entry is therefore also listed explicitly below.
list(PREPEND KERNEL_SRC_LIST "HansEncode ascend950 hans_encode_apt")
AddOpsTestCase(
    OP_NAME hans_encode
    SOC_VERSION "ascend950"
    OTHER_COMPILE_OPTIONS "-D__CCE_UT_TEST__ -DHANS_ENCODE_DEBUG=1"
    TILING_SRC_FILES "${hans_encode_tiling_files}"
    UT_SRC_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../../op_kernel/hans_encode_apt.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_hans_encode_950.cpp")
