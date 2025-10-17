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

file(GLOB CURRENT_SOURCE_DIRS LIST_DIRECTORIES true ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_SOURCE_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()


# ��Ҫ��Tiling�������ļ����ӵ�CMakeLists.txt��
# set(elewise_common_tiling_files
#         ${CANN_ROOT}/ops/built-in/op_tiling/runtime/elewise_tiling.cc
#         )
# �����Լ���tiling�ļ�·��
# set(lin_space_tiling_files
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/lin_space_tiling.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/lin_space_tiling_arch35.cpp
#     ${elewise_common_tiling_files}
#     )
# ʹ��AddOpTestCase
# param1���������ƣ���kernel��ʽ����
# param2��soc�汾������Էֺŷָ������磺"ascend910_9599;AscendB1"
# param3���Զ������ѡ�һ����д���Ե�һ�ֵ�������������ϣ�����Ҫ������ַ��������磺"-DDTYPE_X=float"�����ʹ�ÿո�ָ������磺"-DDTYPE_X=float -DDTYPE_Y=float"
# param4������������������tilingԴ���ļ�
if (UT_TEST_ALL OR FAST_OP_UT)
    # AddOpTestCase(lin_space "ascend910B1;ascend910_9599" "-DDTYPE_X=float" "${lin_space_tiling_files}")
endif()
