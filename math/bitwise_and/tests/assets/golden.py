#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""TTK TestSpec for BitwiseAnd (kernel / aclnn / e2e), golden 统一以 torch.bitwise_and 为标杆.

通路支持: kernel/geir ✅(arch35 实现); aclnn ✅(4 个 C API, 随 libcust_opapi.so 部署,
Ascend 950PR 实测 173/173); e2e ✅(torch.bitwise_and 函数形式, 经 torch_npu 派发).

三层 dtype 契约(golden 逐层对齐, 判据均为 binary_equal -- 按位运算无舍入路径):
1. kernel: 仅 8 整型(def 无 bool), x1/x2 必须同 dtype (tiling 否则 GRAPH_FAILED)
   → 越界 dtype / 混合 dtype 直接 ValueError, 不走 torch 提升.
2. aclnn: bool + 8 整型; Tensor 版按 op::PromoteType 提升({bool,int8,uint8,int16,
   int32,int64} 全 36 组合已实测与 torch 提升一致; uint16/32/64 混合 → DT_UNDEFINED,
   golden 同拒). Scalar 版(RegBase/950) self dtype 优先 ≡ torch 标量 weak-typing;
   float/complex 标量拒绝(CanCast false). Inplace == 普通版取 out=selfRef, 结果
   cast 回 self dtype (按位与与补码截断可交换, 值等价已随机压力验证). bool → API
   走 LogicalAnd ≡ torch.bitwise_and(bool,bool).
3. e2e: 契约即 torch 语义(框架默认 CPU 同 API), 不叠加 CANN 限制.
   已知限制(TTK 8167baa): torch.Tensor.bitwise_and_ 方法形式因双重载回退
   alias(torch.bitwise_and) 解析出 out kwarg 并注入, 设备端调用 TypeError
   (框架调用构造问题, golden 未执行); 注册保留待上游修复, 验证用函数形式.

三方(GPU)标杆: 每个 Spec 均按 TestSpec 规范声明 third_party={"torch": _XxxCompose},
  由 TTK 远端 GPU 以设备侧 torch.Tensor 执行(与 CPU golden 同 API、不同设备, 互为
  印证)。cross_check 档位下参与 NPU/golden/third_party 三方比对; 本算子判据为
  binary_equal(位运算无舍入路径), 故此处仅作声明, 不改变既有精度口径。
"""

import numpy as np
import torch

__spec__ = {
    # kernel 与 geir 共用蛇形名注册
    "bitwise_and": "BitwiseAndKernelSpec",
    "aclnnBitwiseAndTensor": "AclnnBitwiseAndTensorSpec",
    "aclnnBitwiseAndScalar": "AclnnBitwiseAndScalarSpec",
    "aclnnInplaceBitwiseAndTensor": "AclnnInplaceBitwiseAndTensorSpec",
    "aclnnInplaceBitwiseAndScalar": "AclnnInplaceBitwiseAndScalarSpec",
    "aclnnBitwiseAndTensorOut": "AclnnBitwiseAndTensorOutSpec",
    "aclnnInplaceBitwiseAndTensorOut": "AclnnInplaceBitwiseAndTensorOutSpec",
    "torch.bitwise_and": "TorchBitwiseAndSpec",
    "torch.Tensor.bitwise_and": "TorchBitwiseAndSpec",
    "torch.Tensor.bitwise_and_": "TorchBitwiseAndInplaceSpec",
}

# legacy 注册(消费方迁移到 TestSpec 期间保留)
__golden__ = {
    "kernel": {"bitwise_and": "bitwise_and_golden"},
    "aclnn": {
        "aclnnBitwiseAndTensor": "aclnn_bitwise_and_tensor_golden",
        "aclnnBitwiseAndScalar": "aclnn_bitwise_and_scalar_golden",
        "aclnnInplaceBitwiseAndTensor": "aclnn_inplace_bitwise_and_tensor_golden",
        "aclnnInplaceBitwiseAndScalar": "aclnn_inplace_bitwise_and_scalar_golden",
        "aclnnBitwiseAndTensorOut": "aclnn_bitwise_and_tensor_out_golden",
        "aclnnInplaceBitwiseAndTensorOut": "aclnn_inplace_bitwise_and_tensor_out_golden",
    },
    "e2e": {
        "torch.bitwise_and": "torch_bitwise_and_golden",
        "torch.Tensor.bitwise_and_": "torch_bitwise_and_inplace_golden",
    },
}

# kernel 契约: bitwise_and_def.cpp 的 8 整型(无 bool)
_KERNEL_INT_DTYPES = frozenset(
    map(
        np.dtype,
        ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"),
    )
)

# aclnn 契约: kPromoteTypesLookup 中可自由互提升的集合(与 torch 提升实测一致)
_API_PROMOTABLE_DTYPES = (
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
)

# kPromoteTypesLookup u2/u4/u8 行: 与任何异 dtype 组合均为 DT_UNDEFINED (API 拒绝)
_API_SAME_DTYPE_ONLY = (torch.uint16, torch.uint32, torch.uint64)

_KERNEL_TOLERANCE = {
    dtype: {"standard": "binary_equal"}
    for dtype in (
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    )
}
# aclnn/e2e 额外支持 bool (API 层 LogicalAnd 分支)
_API_TOLERANCE = {**_KERNEL_TOLERANCE, "bool": {"standard": "binary_equal"}}


def _to_torch(array):
    """numpy → torch (无损; torch 为统一标杆)."""
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array)


def _as_torch(tensor):
    """入参归一为 torch.Tensor (aclnn/e2e 常态即 torch; numpy 兜底)."""
    if isinstance(tensor, np.ndarray):
        return _to_torch(tensor)
    return tensor


def _wrap_scalar_to(other, ref):
    """aclnn Scalar 契约: 整型标量回绕到 self dtype (RegBase self 优先 ≡ torch weak-typing;
    0-dim tensor 按 ConvertToTensor(scalar, self_dtype) 语义 cast); float/complex 标量
    属类型错误, 抛 TypeError (API 侧对应 CanCast false → ACLNN_ERR_PARAM_INVALID,
    torch 侧对应 NotImplementedError; TTK 统一按 GOLDEN_FAILURE 处理)."""
    if torch.is_tensor(other):
        if other.is_floating_point() or other.is_complex():
            raise TypeError(
                f"aclnnBitwiseAndScalar rejects floating/complex scalars "
                f"(CanCast({other.dtype}->int) is false, ACLNN_ERR_PARAM_INVALID); "
                f"got scalar dtype {other.dtype}"
            )
        return other.to(ref.dtype)
    if isinstance(other, float):
        raise TypeError(
            "aclnnBitwiseAndScalar rejects floating scalars "
            "(ACLNN_ERR_PARAM_INVALID); torch raises NotImplementedError -- "
            f"got Python float {other!r}"
        )
    return other


def _check_api_promotable(dt_a, dt_b):
    """镜像 op::PromoteType: 在 C++ host 检查会报 ACLNN_ERR_PARAM_INVALID 处抛
    ValueError (诊断信息对齐 API), 其余交给 torch 原生提升(已验证与 CANN 表一致)."""
    if dt_a == dt_b:
        return
    if dt_a in _API_SAME_DTYPE_ONLY or dt_b in _API_SAME_DTYPE_ONLY:
        raise ValueError(
            f"op::PromoteType yields DT_UNDEFINED for {dt_a} x {dt_b}: "
            "uint16/uint32/uint64 only combine with the same dtype "
            "(aclnn host check: 'can not promote dtype', ACLNN_ERR_PARAM_INVALID; "
            "torch promotion for uint16/32/64 is likewise unsupported)"
        )
    if dt_a not in _API_PROMOTABLE_DTYPES or dt_b not in _API_PROMOTABLE_DTYPES:
        raise ValueError(
            f"aclnn BitwiseAnd supports bool + the 8 integer dtypes only "
            f"(DTYPE_SUPPORT_LIST), got {dt_a} x {dt_b}"
        )


def _cast_into_out(result, out):
    """复刻 l0op::Cast(result, out.dtype) + ViewCopy; int→int 截断回绕两侧一致,
    →bool 按 CanCast 拒绝."""
    if out.dtype == torch.bool and result.dtype != torch.bool:
        raise ValueError(
            f"CanCast({result.dtype}->bool) is false: aclnn host check rejects "
            "non-bool results into a bool out tensor (ACLNN_ERR_PARAM_INVALID)"
        )
    out.copy_(result)
    return out


def _compute_api_tensor(self_t, other_t):
    """aclnn Tensor 契约: CANN 提升(支持集内 == torch 提升)."""
    _check_api_promotable(self_t.dtype, other_t.dtype)
    return torch.bitwise_and(self_t, other_t)


def _compute_api_scalar(self_t, other):
    """aclnn Scalar 契约(RegBase/950): self dtype 优先(weak scalar)."""
    return torch.bitwise_and(self_t, _wrap_scalar_to(other, self_t))


# 三方(GPU)标杆: 由 TTK 远端 GPU 执行, 与 golden(CPU) 同 API、不同设备, 互为印证。
# 形参名按各入口对位: kernel=x1/x2(def.cpp), aclnn=头文件形参(self/selfRef/other/out),
# e2e=torch 形参(input/other)。aclnn 的 self 条目不能作为方法形参名(Python 保留名),
# 与同仓 logical_not/neg/cast 一致以 self_ 承接; 未消费条目由 *args/**kwargs 吸收。


class _BitwiseAndKernelCompose:
    """三方(GPU)标杆: bitwise_and kernel(8 整型, x1/x2 同 dtype)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, x1=None, x2=None, *args, **kwargs):
        del args, kwargs
        return [torch.bitwise_and(_as_torch(x1), _as_torch(x2))]


class _BitwiseAndTensorCompose:
    """三方(GPU)标杆: aclnnTensor/aclnnBitwiseAndTensorOut(self, other, out)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, self_=None, other=None, out=None, *args, **kwargs):
        del args, kwargs
        result = _compute_api_tensor(_as_torch(self_), _as_torch(other))
        return [_cast_into_out(result, _as_torch(out))] if out is not None else [result]


class _BitwiseAndScalarCompose:
    """三方(GPU)标杆: aclnnBitwiseAndScalar(self, other:Scalar, out)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, self_=None, other=None, out=None, *args, **kwargs):
        del args, kwargs
        result = _compute_api_scalar(_as_torch(self_), other)
        return [_cast_into_out(result, _as_torch(out))] if out is not None else [result]


class _BitwiseAndInplaceTensorCompose:
    """三方(GPU)标杆: aclnnInplaceBitwiseAndTensor(selfRef, other)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, selfRef=None, other=None, out=None, *args, **kwargs):
        del out, args, kwargs
        result = _compute_api_tensor(_as_torch(selfRef), _as_torch(other))
        return [result.to(selfRef.dtype)]


class _BitwiseAndInplaceScalarCompose:
    """三方(GPU)标杆: aclnnInplaceBitwiseAndScalar(selfRef, other:Scalar)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, selfRef=None, other=None, out=None, *args, **kwargs):
        del out, args, kwargs
        result = _compute_api_scalar(_as_torch(selfRef), other)
        return [result.to(selfRef.dtype)]


class _TorchBitwiseAndCompose:
    """三方(GPU)标杆: torch.bitwise_and(input, other) 函数形式(e2e)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, input=None, other=None, out=None, *args, **kwargs):
        del out, args, kwargs
        return [torch.bitwise_and(_as_torch(input), _as_torch(other))]


class _TorchBitwiseAndInplaceCompose:
    """三方(GPU)标杆: torch.Tensor.bitwise_and_(self, other) 方法形式(e2e)。"""

    def __init__(self, **kwargs):
        del kwargs

    def __call__(self, self_=None, other=None, *args, **kwargs):
        del args, kwargs
        result = _as_torch(self_).clone()
        result.bitwise_and_(_as_torch(other))
        return [result]


class BitwiseAndKernelSpec:
    """Kernel/GEIR: numpy 入参, torch 计算; 契约源 def.cpp + tiling.cpp (同 dtype, 8 整型)."""

    @staticmethod
    def golden(x1, x2, **kwargs):
        del kwargs
        for name, arr in (("x1", x1), ("x2", x2)):
            if arr.dtype not in _KERNEL_INT_DTYPES:
                raise ValueError(
                    f"bitwise_and kernel supports the 8 integer dtypes of "
                    f"bitwise_and_def.cpp (no bool/float), got {name}.dtype={arr.dtype}"
                )
        if x1.dtype != x2.dtype:
            raise ValueError(
                f"bitwise_and kernel requires identical input dtypes "
                f"(tiling returns GRAPH_FAILED otherwise), got x1={x1.dtype}, x2={x2.dtype}"
            )
        # numpy 式广播 == BroadcastBaseTiling 规则, torch 逐位一致
        return [torch.bitwise_and(_to_torch(x1), _to_torch(x2)).numpy()]

    third_party = {"torch": _BitwiseAndKernelCompose}
    tolerance = _KERNEL_TOLERANCE


class AclnnBitwiseAndTensorSpec:
    """ACLNN: 签名随 aclnnBitwiseAndTensorGetWorkspaceSize(self, other, out).
    流水建模: PromoteType → BitwiseAnd/LogicalAnd → Cast 入 out."""

    @staticmethod
    def golden(selfT, other, out=None, **kwargs):
        del kwargs
        selfT = _as_torch(selfT)
        other = _as_torch(other)
        result = _compute_api_tensor(selfT, other)
        if out is not None:
            return [_cast_into_out(result, _as_torch(out))]
        return [result]

    third_party = {"torch": _BitwiseAndTensorCompose}
    tolerance = _API_TOLERANCE


class AclnnBitwiseAndScalarSpec:
    """ACLNN: 签名随 aclnnBitwiseAndScalarGetWorkspaceSize(self, other:Scalar, out).
    RegBase 标量语义: self dtype 优先, 值按补码回绕(与 API promote-cast 流水等价)."""

    @staticmethod
    def golden(selfT, other, out=None, **kwargs):
        del kwargs
        selfT = _as_torch(selfT)
        result = _compute_api_scalar(selfT, other)
        if out is not None:
            return [_cast_into_out(result, _as_torch(out))]
        return [result]

    third_party = {"torch": _BitwiseAndScalarCompose}
    tolerance = _API_TOLERANCE


class AclnnInplaceBitwiseAndTensorSpec:
    """ACLNN: 签名随 aclnnInplaceBitwiseAndTensorGetWorkspaceSize(selfRef, other).
    实现即普通版取 out=selfRef, 结果显式 cast 回 selfRef dtype."""

    @staticmethod
    def golden(selfRef, other, **kwargs):
        del kwargs
        selfRef = _as_torch(selfRef)
        other = _as_torch(other)
        result = _compute_api_tensor(selfRef, other)
        return [result.to(selfRef.dtype)]

    third_party = {"torch": _BitwiseAndInplaceTensorCompose}
    tolerance = _API_TOLERANCE


class AclnnInplaceBitwiseAndScalarSpec:
    """ACLNN: 签名随 aclnnInplaceBitwiseAndScalarGetWorkspaceSize(selfRef, other:Scalar)."""

    @staticmethod
    def golden(selfRef, other, **kwargs):
        del kwargs
        selfRef = _as_torch(selfRef)
        result = _compute_api_scalar(selfRef, other)
        return [result.to(selfRef.dtype)]

    third_party = {"torch": _BitwiseAndInplaceScalarCompose}
    tolerance = _API_TOLERANCE


class AclnnBitwiseAndTensorOutSpec(AclnnBitwiseAndTensorSpec):
    """ACLNN: 签名随 aclnnBitwiseAndTensorOutGetWorkspaceSize(self, other, out);
    计算与推导契约同 AclnnBitwiseAndTensorSpec(TensorOut 与 Tensor 等价)."""


class AclnnInplaceBitwiseAndTensorOutSpec(AclnnInplaceBitwiseAndTensorSpec):
    """ACLNN: 签名随 aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize(selfRef, other);
    计算与推导契约同 AclnnInplaceBitwiseAndTensorSpec."""


class TorchBitwiseAndSpec:
    """E2E (torch.bitwise_and / torch.Tensor.bitwise_and): 契约即 torch 语义,
    不叠加 CANN 限制; CPU/设备差异属被测对象."""

    @staticmethod
    def golden(input, other, **kwargs):
        del kwargs
        return [torch.bitwise_and(_as_torch(input), _as_torch(other))]

    third_party = {"torch": _TorchBitwiseAndCompose}
    tolerance = _API_TOLERANCE


class TorchBitwiseAndInplaceSpec:
    """E2E 方法形式(torch.Tensor.bitwise_and_): clone 后原地与, 保持 self dtype,
    不污染框架输入. TTK 8167baa 下该 api_name 存在框架侧 out= 注入 bug (见模块注释)."""

    @staticmethod
    def golden(selfT, other, **kwargs):
        del kwargs
        result = _as_torch(selfT).clone()
        result.bitwise_and_(_as_torch(other))
        return [result]

    third_party = {"torch": _TorchBitwiseAndInplaceCompose}
    tolerance = _API_TOLERANCE


# legacy 函数入口(TestSpec 迁移期保留); 参数名随对应 C/torch 签名, 按位置绑定,
# 统一返回列表(每输出一个元素).


def bitwise_and_golden(x1, x2, **kwargs):
    """legacy kernel golden (numpy 入参)."""
    return BitwiseAndKernelSpec.golden(x1, x2, **kwargs)


def aclnn_bitwise_and_tensor_golden(selfT, other, out=None, **kwargs):
    """legacy aclnn golden: aclnnBitwiseAndTensor."""
    return AclnnBitwiseAndTensorSpec.golden(selfT, other, out=out, **kwargs)


def aclnn_bitwise_and_scalar_golden(selfT, other, out=None, **kwargs):
    """legacy aclnn golden: aclnnBitwiseAndScalar."""
    return AclnnBitwiseAndScalarSpec.golden(selfT, other, out=out, **kwargs)


def aclnn_inplace_bitwise_and_tensor_golden(selfRef, other, **kwargs):
    """legacy aclnn golden: aclnnInplaceBitwiseAndTensor."""
    return AclnnInplaceBitwiseAndTensorSpec.golden(selfRef, other, **kwargs)


def aclnn_inplace_bitwise_and_scalar_golden(selfRef, other, **kwargs):
    """legacy aclnn golden: aclnnInplaceBitwiseAndScalar."""
    return AclnnInplaceBitwiseAndScalarSpec.golden(selfRef, other, **kwargs)


def aclnn_bitwise_and_tensor_out_golden(selfT, other, out=None, **kwargs):
    """legacy aclnn golden: aclnnBitwiseAndTensorOut."""
    return AclnnBitwiseAndTensorOutSpec.golden(selfT, other, out=out, **kwargs)


def aclnn_inplace_bitwise_and_tensor_out_golden(selfRef, other, **kwargs):
    """legacy aclnn golden: aclnnInplaceBitwiseAndTensorOut."""
    return AclnnInplaceBitwiseAndTensorOutSpec.golden(selfRef, other, **kwargs)


def torch_bitwise_and_golden(input, other, **kwargs):
    """legacy e2e golden: torch.bitwise_and."""
    return TorchBitwiseAndSpec.golden(input, other, **kwargs)


def torch_bitwise_and_inplace_golden(selfT, other, **kwargs):
    """legacy e2e golden: torch.Tensor.bitwise_and_."""
    return TorchBitwiseAndInplaceSpec.golden(selfT, other, **kwargs)
