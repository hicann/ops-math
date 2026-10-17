# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------------------------------------

# 设置算子定义时支持的芯片类型
set(SUPPORT_COMPUTE_UNIT "ascend950" "ascend350" "ascend960dt" "ascend960pr")
# 设置每种芯片类型对应的 tiling 文件目录，即采用op_host目录下哪个文件夹下的tiling文件编译
# A5/A6 tiling 逻辑一致（key/counter 元数据下发，kernel 从 GM 读取 seed/offset），
# 仅 A6 (DV121) 核间对齐粒度差异通过运行时架构判断隔离，统一使用 arch35
set(SUPPORT_TILING_DIR "arch35" "arch35" "arch35" "arch35")
add_all_modules_sources(OPTYPE stateless_drop_out_gen_mask ACLNNTYPE aclnn_exclude COMPUTE_UNIT ${SUPPORT_COMPUTE_UNIT} TILING_DIR ${SUPPORT_TILING_DIR} DEPENDENCIES random_common DISABLE_IN_OPP TRUE)
