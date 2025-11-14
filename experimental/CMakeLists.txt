# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
set(COMMON_L0_SRCS
  ${CMAKE_SOURCE_DIR}/math/cast/op_host/op_api/cast.cpp
  ${CMAKE_SOURCE_DIR}/conversion/contiguous/op_host/op_api/contiguous.cpp
  ${CMAKE_SOURCE_DIR}/conversion/reshape/op_host/op_api/reshape.cpp
  ${CMAKE_SOURCE_DIR}/conversion/slice/op_host/op_api/slice.cpp
  ${CMAKE_SOURCE_DIR}/conversion/as_strided/op_host/op_api/as_strided.cpp
  ${CMAKE_SOURCE_DIR}/conversion/view_copy/op_host/op_api/view_copy.cpp
  ${CMAKE_SOURCE_DIR}/conversion/strided_slice/op_host/op_api/strided_slice.cpp
  ${CMAKE_SOURCE_DIR}/conversion/broadcast_to/op_host/op_api/broadcast_to.cpp
  ${CMAKE_SOURCE_DIR}/conversion/tensor_move/op_host/op_api/tensor_move.cpp
  ${CMAKE_SOURCE_DIR}/conversion/trans_data/op_host/op_api/transdata.cpp
  ${CMAKE_SOURCE_DIR}/conversion/transpose/op_host/op_api/transpose.cpp
  ${CMAKE_SOURCE_DIR}/conversion/pad/op_host/op_api/pad.cpp)

add_opapi_modules()
target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${COMMON_L0_SRCS})

add_subdirectory(math)
add_subdirectory(conversion)
add_subdirectory(random)