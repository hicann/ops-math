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
function(gen_common_symbol)
  add_library(
    ${COMMON_NAME} SHARED
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    )

  target_link_libraries(
    ${COMMON_NAME}
    PRIVATE c_sec -Wl,--no-as-needed register -Wl,--as-needed exe_graph tiling_api
    )

  install(
    TARGETS ${COMMON_NAME}
    LIBRARY DESTINATION ${COMMON_LIB_INSTALL_DIR}
    )
  install(
    DIRECTORY ${OPS_MATH_COMMON_INC_HEADERS}
    DESTINATION ${COMMON_INC_INSTALL_DIR}
    )
endfunction()

# ophost shared
function(gen_ophost_symbol)
  add_library(
    ${OPHOST_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
    $<$<TARGET_EXISTS:${OPHOST_NAME}_aicpu_objs>:$<TARGET_OBJECTS:${OPHOST_NAME}_aicpu_objs>>
    $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    )

  target_link_libraries(
    ${OPHOST_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
            tiling_api
    )

  target_link_directories(${OPHOST_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  if(NOT ENABLE_CUSTOM)
    install(
      TARGETS ${OPHOST_NAME}
      LIBRARY DESTINATION ${OPHOST_LIB_INSTALL_PATH}
      )
  endif()
endfunction()

# ######################################################################################################################
# merge ops proto headers in aclnn/aclnn_inner/aclnn_exc to a total proto file srcpath: ${ASCEND_AUTOGEN_PATH} generate
# outpath: ${CMAKE_BINARY_DIR}/tbe/graph
# ######################################################################################################################
function(merge_graph_headers)
  set(oneValueArgs TARGET OUT_DIR)
  cmake_parse_arguments(MGPROTO "" "${oneValueArgs}" "" ${ARGN})
  get_target_property(proto_headers ${GRAPH_PLUGIN_NAME}_proto_headers INTERFACE_SOURCES)
  add_custom_command(
    OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_math.h
    COMMAND
      ${ASCEND_PYTHON_EXECUTABLE} ${CMAKE_SOURCE_DIR}/scripts/util/merge_proto.py ${proto_headers} --output-file
      ${MGPROTO_OUT_DIR}/ops_proto_math.h
    )
  add_custom_command(
    OUTPUT ${MGPROTO_OUT_DIR}/ops_proto_math.cpp
    COMMAND ${CMAKE_COMMAND} -E copy
      ${MGPROTO_OUT_DIR}/ops_proto_math.h
      ${MGPROTO_OUT_DIR}/ops_proto_math.cpp
    DEPENDS ${MGPROTO_OUT_DIR}/ops_proto_math.h
    )
  add_custom_target(${MGPROTO_TARGET} ALL DEPENDS ${MGPROTO_OUT_DIR}/ops_proto_math.h ${MGPROTO_OUT_DIR}/ops_proto_math.cpp)
endfunction()

# graph_plugin shared
function(gen_opgraph_symbol)
  add_library(
    ${OPGRAPH_NAME} SHARED
    $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
    )
  merge_graph_headers(TARGET merge_ops_proto OUT_DIR ${ASCEND_GRAPH_CONF_DST})
  add_dependencies(${OPGRAPH_NAME} merge_ops_proto)

  target_sources(
    ${OPGRAPH_NAME} PRIVATE ${ASCEND_GRAPH_CONF_DST}/ops_proto_math.cpp
    )

  target_link_libraries(
    ${OPGRAPH_NAME}
    PRIVATE $<BUILD_INTERFACE:intf_pub_cxx17>
            c_sec
            -Wl,--no-as-needed
            register
            $<$<TARGET_EXISTS:opsbase>:opsbase>
            -Wl,--as-needed
            -Wl,--whole-archive
            rt2_registry_static
            -Wl,--no-whole-archive
    )

  target_link_directories(${OPGRAPH_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64)

  install(
    TARGETS ${OPGRAPH_NAME}
    LIBRARY DESTINATION ${OPGRAPH_LIB_INSTALL_DIR}
    )

  install(
    FILES ${ASCEND_GRAPH_CONF_DST}/ops_proto_math.h
    DESTINATION ${OPGRAPH_INC_INSTALL_DIR}
    OPTIONAL
    )
endfunction()

function(gen_opapi_symbol)
  # opapi shared
  add_library(
    ${OPAPI_NAME} SHARED
    $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
    $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )

  target_link_libraries(
    ${OPAPI_NAME}
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    PRIVATE c_sec nnopbase
    $<$<TARGET_EXISTS:opsbase>:opsbase>
    )

  install(
    TARGETS ${OPAPI_NAME}
    LIBRARY DESTINATION ${ACLNN_LIB_INSTALL_DIR}
    )
endfunction()

function(gen_cust_opapi_symbol)
  # op_api
  npu_op_library(cust_opapi ACLNN)
  target_sources(
    cust_opapi
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_opapi_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_opapi_obj>>
           $<$<TARGET_EXISTS:opbuild_gen_aclnn_all>:$<TARGET_OBJECTS:opbuild_gen_aclnn_all>>
    )
  target_link_libraries(
    cust_opapi
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    $<$<TARGET_EXISTS:opsbase>:opsbase>
    )
endfunction()

function(gen_cust_optiling_symbol)
  # op_tiling
  if(NOT TARGET ${OPHOST_NAME}_tiling_obj)
    return()
  endif()
  npu_op_library(cust_opmaster TILING)
  target_sources(
    cust_opmaster
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_tiling_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_tiling_obj>>
           $<$<TARGET_EXISTS:${COMMON_NAME}_obj>:$<TARGET_OBJECTS:${COMMON_NAME}_obj>>
    )
  target_link_libraries(
    cust_opmaster
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    PRIVATE $<$<TARGET_EXISTS:opsbase>:opsbase>
    )
endfunction()

function(gen_cust_proto_symbol)
  # op_proto
  if(NOT TARGET ${OPHOST_NAME}_infer_obj)
    return()
  endif()
  npu_op_library(cust_proto GRAPH)
  target_sources(
    cust_proto
    PUBLIC $<$<TARGET_EXISTS:${OPHOST_NAME}_infer_obj>:$<TARGET_OBJECTS:${OPHOST_NAME}_infer_obj>>
           $<$<TARGET_EXISTS:${GRAPH_PLUGIN_NAME}_obj>:$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>>
    )
  target_link_libraries(
    cust_proto
    PUBLIC $<BUILD_INTERFACE:intf_pub_cxx17>
    PRIVATE $<$<TARGET_EXISTS:opsbase>:opsbase>
    )
  file(GLOB_RECURSE proto_headers ${ASCEND_AUTOGEN_PATH}/*_proto.h)
  install(
    FILES ${proto_headers}
    DESTINATION ${OPPROTO_INC_INSTALL_DIR}
    OPTIONAL
    )
endfunction()

function(gen_cust_aicpu_json_symbol)
  get_property(ALL_AICPU_JSON_FILES GLOBAL PROPERTY AICPU_JSON_FILES)
  if(NOT ALL_AICPU_JSON_FILES)
    message(STATUS "No aicpu json files to merge, skipping.")
    return()
  endif()

  set(MERGED_JSON ${CMAKE_BINARY_DIR}/cust_aicpu_kernel.json)
  add_custom_command(
    OUTPUT ${MERGED_JSON}
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/util/merge_aicpu_info_json.sh ${CMAKE_SOURCE_DIR} ${MERGED_JSON} ${ALL_AICPU_JSON_FILES}
    DEPENDS ${ALL_AICPU_JSON_FILES}
    COMMENT "Merging Json files into ${MERGED_JSON}"
    VERBATIM
  )
  add_custom_target(merge_aicpu_json ALL DEPENDS ${MERGED_JSON})
  install(
    FILES ${MERGED_JSON}
    DESTINATION ${CUST_AICPU_KERNEL_CONFIG}
    OPTIONAL
  )
endfunction()

function(gen_cust_aicpu_kernel_symbol)
  if(NOT AICPU_CUST_OBJ_TARGETS)
    message(STATUS "No aicpu cust obj targets found, skipping.")
    return()
  endif()

  set(ARM_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  set(ARM_SO_OUTPUT ${CMAKE_BINARY_DIR}/libcust_aicpu_kernels.so)

  set(ALL_OBJECTS "")
  foreach(tgt IN LISTS AICPU_CUST_OBJ_TARGETS)
    list(APPEND ALL_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  endforeach()

  message(STATUS "Linking cust_aicpu_kernels with ARM toolchain: ${ARM_CXX_COMPILER}")
  message(STATUS "Objects: ${ALL_OBJECTS}")
  message(STATUS "Output: ${ARM_SO_OUTPUT}")

  add_custom_command(
    OUTPUT ${ARM_SO_OUTPUT}
    COMMAND ${ARM_CXX_COMPILER} -shared ${ALL_OBJECTS}
      -Wl,--whole-archive
      ${ASCEND_DIR}/ops_base/lib64/libaicpu_context.a
      ${ASCEND_DIR}/ops_base/lib64/libbase_ascend_protobuf.a
      -Wl,--no-whole-archive
      -Wl,-Bsymbolic
      -Wl,--exclude-libs=libbase_ascend_protobuf.a
      -s
      -o ${ARM_SO_OUTPUT}
    DEPENDS ${AICPU_CUST_OBJ_TARGETS}
    COMMENT "Linking cust_aicpu_kernels.so using ARM toolchain"
  )
  add_custom_target(cust_aicpu_kernels ALL DEPENDS ${ARM_SO_OUTPUT})

  install(
    FILES ${ARM_SO_OUTPUT}
    DESTINATION ${CUST_AICPU_KERNEL_IMPL}
    OPTIONAL
  )
endfunction()

function(gen_norm_symbol)
  gen_common_symbol()

  gen_ophost_symbol()

  gen_opgraph_symbol()

  gen_opapi_symbol()
endfunction()

function(gen_cust_symbol)
  gen_ophost_symbol()

  gen_cust_opapi_symbol()

  gen_cust_optiling_symbol()

  gen_cust_proto_symbol()

  gen_cust_aicpu_json_symbol()

  gen_cust_aicpu_kernel_symbol()
endfunction()
