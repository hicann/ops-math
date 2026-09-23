/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mul_aicpu.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "aicpu/math_aicpu_register.h"
#include "cpu_kernel_utils.h"
constexpr int64_t kMaxCoreNumForMul = 4;
#include "cpu_types.h"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

#if (defined __ARM_ARCH) || (defined PLATFORM_AARCH64)
#include <arm_neon.h>
#define MUL_USE_NEON 1
#elif defined(__x86_64__) || defined(_M_X64)
#include <emmintrin.h>
#include <xmmintrin.h>
#define MUL_USE_X86_SIMD 1
#endif

namespace {
const char* const kMul = "Mul";
constexpr uint32_t kInputNum = 2;
constexpr uint32_t kOutputNum = 1;
constexpr int64_t kParallelDataNum = 6 * 1024;
constexpr int64_t kParallelDataNumMid = 33 * 1024;
constexpr int64_t kParallelDataNumSameShape = 7 * 1024;

// The strip loops have to be vectorised to beat Eigen, and the production flag set stops at
// -O2, whose very-cheap vectoriser cost model rejects a loop that needs a scalar epilogue.
// Running the body over a fixed-length block gives the model a known, vector-width-multiple
// trip count, which it accepts. Doing it this way rather than with an optimize("O3")
// attribute matters: that attribute also changes the code GCC emits for the Eigen COMDAT
// template instantiations that share this translation unit, measured here at 1.5x-1.9x
// either way depending on dtype.
constexpr int64_t kMulVecBlock = 64;

// Second, shorter block. A run of 32 is common (an NHWC channel count) and would otherwise
// fall entirely into the scalar tail of a 64-element block -- measured 1.58x behind on
// int16 before this level existed.
constexpr int64_t kMulVecBlockSmall = 16;
constexpr int64_t kMulNeonUnroll = 4;
constexpr int64_t kMulInt16NeonUnroll = 8;
constexpr int64_t kMulInt32NeonUnroll = 8;
constexpr int64_t kMulUint16NeonUnroll = 8;
constexpr int64_t kMulUint32NeonUnroll = 8;

// Stack budget for the short-inner-run tile, in bytes. Fixed size, so the kernel's stack
// stays independent of the tensor shape.
constexpr int64_t kMulTileBytes = 2048;

// Longest innermost run, in bytes, still worth expanding into a tile. Past it the strip is
// already several vectors long and the fill stops paying for itself: measured on int16, a
// run of 3 and a run of 8 both gain (0.91x, 0.67x) while a run of 32 loses (1.30x).
constexpr int64_t kMulTileMaxRunBytes = 64;
constexpr int64_t kMulPairTileRun = 2;
constexpr int64_t kMulTripletTileRun = 3;
constexpr int64_t kMulQuadTileRun = 4;
constexpr int64_t kMulOctetTileRun = 8;
constexpr int32_t kMulShortInnerPlanDims = 2;
constexpr int32_t kMulStrideCarryOffset = 2;
#define MUL_HOT_INLINE __attribute__((always_inline)) inline
// The blocked loops only get vectorised when the function holding them is small enough for
// GCC to analyse on its own; folded into the dispatcher they stay scalar. noinline keeps
// each driver a self-contained loop nest, which is what the optimize attribute used to do
// as a side effect -- without that attribute's effect on the rest of the file.
#define MUL_HOT __attribute__((noinline))
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
#define MUL_HOST_NO_IVOPTS __attribute__((optimize("no-ivopts")))
#else
#define MUL_HOST_NO_IVOPTS
#endif
} // namespace

namespace aicpu {
namespace {

/* ---- NEON multiply cores -------------------------------------------------------------
 * GCC vectorises the blocked loops below for the dtypes it can express with a single
 * instruction. Four cannot be: aarch64 has no 64-bit integer vector multiply, half and
 * bfloat16 arithmetic is emulated through float without the fp16 extension, and complex
 * multiply needs a shuffle sequence. Writing those four by hand is what lets the kernel
 * drop Eigen entirely -- measured on half, 0.173x against Eigen at 4.8M elements.
 *
 * MulNeon<T>::kLanes is 0 where no hand-written core exists; the strip helpers then use the
 * blocked loop and let GCC vectorise it. ARM uses NEON and x86_64 uses SSE2; on other
 * platforms the blocked scalar loop remains available.
 */
template <typename T>
MUL_HOT_INLINE typename std::enable_if<std::is_integral<T>::value && std::is_signed<T>::value, T>::type MulValue(T a,
                                                                                                                 T b)
{
    T result;
    static_cast<void>(__builtin_mul_overflow(a, b, &result));
    return result;
}

template <typename T>
MUL_HOT_INLINE typename std::enable_if<!std::is_integral<T>::value || !std::is_signed<T>::value, T>::type MulValue(T a,
                                                                                                                   T b)
{
    return a * b;
}

MUL_HOT_INLINE uint16_t MulValue(uint16_t left, uint16_t right)
{
    return static_cast<uint16_t>(static_cast<uint32_t>(left) * static_cast<uint32_t>(right));
}

template <typename T>
struct MulIsComplexT {
    static const bool value = std::is_same<T, std::complex<float>>::value ||
                              std::is_same<T, std::complex<double>>::value;
};

/* Complex multiplication the way every packet implementation -- Eigen's included, and the
 * NEON cores below -- computes it: the plain four-multiply form. libstdc++ routes
 * operator* through __mulsc3/__muldc3, which carries the C99 Annex G infinity fix-up, so
 * routing one code path through operator* and another through the packet form would make
 * the result depend on where the vector loop happens to stop. Every same-dtype path goes
 * through these overloads; the diff-type path keeps operator* to stay identical to the
 * previous implementation (see the MulImpl overloads). */
MUL_HOT_INLINE std::complex<float> MulValue(std::complex<float> a, std::complex<float> b)
{
    return std::complex<float>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

MUL_HOT_INLINE std::complex<double> MulValue(std::complex<double> a, std::complex<double> b)
{
    return std::complex<double>(a.real() * b.real() - a.imag() * b.imag(), a.real() * b.imag() + a.imag() * b.real());
}

MUL_HOT_INLINE bool CheckedMulInt64(int64_t a, int64_t b, int64_t& result)
{
    return !__builtin_mul_overflow(a, b, &result);
}

bool CheckedShapeElements(const std::vector<int64_t>& shape, int64_t& elements)
{
    elements = 1;
    for (const int64_t dim : shape) {
        if ((dim < 0) || !CheckedMulInt64(elements, dim, elements)) {
            return false;
        }
    }
    return true;
}

struct MulElementCounts {
    int64_t input0 = 0;
    int64_t input1 = 0;
    int64_t output = 0;
};

MUL_HOT bool GetMulElementCounts(const CpuKernelContext& ctx, const std::vector<int64_t>& input0_shape,
                                 const std::vector<int64_t>& input1_shape, const std::vector<int64_t>& output_shape,
                                 MulElementCounts& elements)
{
    if (!CheckedShapeElements(input0_shape, elements.input0) || !CheckedShapeElements(input1_shape, elements.input1) ||
        !CheckedShapeElements(output_shape, elements.output)) {
        KERNEL_LOG_ERROR("[%s] Shape contains a negative dimension or its element count exceeds the int64 range.",
                         ctx.GetOpType().c_str());
        return false;
    }
    return true;
}

bool ShapeHasOneElement(const std::vector<int64_t>& shape)
{
    return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim == 1; });
}

/* What a leftover element of a run uses: the same core as the blocked loop, so the tail
 * and the blocks cannot disagree. Complex dtypes resolve to the four-multiply MulValue
 * overloads above; non-ARM builds share this path with the blocked loop as well. */
template <typename T>
MUL_HOT_INLINE T MulTail(T a, T b)
{
    return MulValue(a, b);
}

template <typename T>
struct MulNeon {
    /* Zero means "no hand-written core for this dtype"; the strip helpers then use the
     * blocked loop. It is deliberately not 1: complex<double> fills a 128-bit vector with a
     * single element, so a lane count of 1 is a real core, not the absence of one. */
    static const int64_t kLanes = 0;
    static void Step(const T* a, const T* b, T* o) { *o = MulValue(*a, *b); }
    static void StepSplat(const T* a, T v, T* o) { *o = MulValue(*a, v); }
    static void StepSplatLeft(T v, const T* b, T* o) { *o = MulValue(v, *b); }
};

template <typename T>
struct MulNeonUnroll {
    static const int64_t kValue = kMulNeonUnroll;
};

template <>
struct MulNeonUnroll<int16_t> {
    static const int64_t kValue = kMulInt16NeonUnroll;
};

template <>
struct MulNeonUnroll<uint16_t> {
    static const int64_t kValue = kMulUint16NeonUnroll;
};

template <>
struct MulNeonUnroll<int32_t> {
    static const int64_t kValue = kMulInt32NeonUnroll;
};

template <>
struct MulNeonUnroll<uint32_t> {
    static const int64_t kValue = kMulUint32NeonUnroll;
};

template <typename T>
struct MulPreparedSplat {
    using Type = T;
    static Type Prepare(T v) { return v; }
    static void Step(const T* a, Type v, T* o) { MulNeon<T>::StepSplat(a, v, o); }
};

template <typename T>
struct MulPreparedLeftSplat {
    using Type = T;
    static Type Prepare(T v) { return v; }
    static void Step(Type v, const T* b, T* o) { MulNeon<T>::StepSplatLeft(v, b, o); }
};

#ifdef MUL_USE_NEON
/* Bit patterns the half and bfloat16 cores need in order to reproduce Eigen's conversions
 * exactly. Named because a raw 0x7E00 next to a raw 16 says nothing about which field of
 * which format it belongs to. */
constexpr uint32_t kFloatSignMask = 0x80000000U;
constexpr int kFloatToHalfSignShift = 16;
constexpr uint16_t kHalfQuietNanPayload = 0x7E00U;
constexpr int kBfloat16MantissaShift = 16;
constexpr uint32_t kBfloat16RoundingBias = 0x7FFFU;
constexpr uint32_t kBfloat16QuietNan = 0x7FC0U;

template <>
struct MulNeon<Eigen::half> {
    static const int64_t kLanes = 4;
    static float32x4_t Load(const Eigen::half* p)
    {
        return vcvt_f32_f16(vld1_f16(PtrToPtr<const Eigen::half, const __fp16>(p)));
    }
    /* vcvt_f16_f32 keeps a NaN's payload; Eigen's float_to_half_rtne collapses every NaN to
     * sign|0x7E00. Blend the canonical form back in so the two agree bit for bit. */
    static void Store(Eigen::half* p, float32x4_t v)
    {
        const uint16x4_t converted = vreinterpret_u16_f16(vcvt_f16_f32(v));
        const uint16x4_t is_nan = vmovn_u32(vmvnq_u32(vceqq_f32(v, v)));
        const uint16x4_t sign = vshrn_n_u32(vandq_u32(vreinterpretq_u32_f32(v), vdupq_n_u32(kFloatSignMask)),
                                            kFloatToHalfSignShift);
        const uint16x4_t canonical_nan = vorr_u16(sign, vdup_n_u16(kHalfQuietNanPayload));
        vst1_u16(PtrToPtr<Eigen::half, uint16_t>(p), vbsl_u16(is_nan, canonical_nan, converted));
    }
    static void Step(const Eigen::half* a, const Eigen::half* b, Eigen::half* o)
    {
        Store(o, vmulq_f32(Load(a), Load(b)));
    }
    static void StepSplat(const Eigen::half* a, Eigen::half v, Eigen::half* o)
    {
        Store(o, vmulq_f32(Load(a), vdupq_n_f32(static_cast<float>(v))));
    }
    static void StepSplatLeft(Eigen::half v, const Eigen::half* b, Eigen::half* o)
    {
        Store(o, vmulq_f32(vdupq_n_f32(static_cast<float>(v)), Load(b)));
    }
};

template <>
struct MulNeon<Eigen::bfloat16> {
    static const int64_t kLanes = 4;
    static float32x4_t Load(const Eigen::bfloat16* p)
    {
        const uint16x4_t raw = vld1_u16(PtrToPtr<const Eigen::bfloat16, const uint16_t>(p));
        return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(raw), kBfloat16MantissaShift));
    }
    /* Rounding reproduced from Eigen's F32ToBf16 so the results stay bit-identical. */
    static void Store(Eigen::bfloat16* p, float32x4_t v)
    {
        uint32x4_t bits = vreinterpretq_u32_f32(v);
        const uint32x4_t lsb = vandq_u32(vshrq_n_u32(bits, kBfloat16MantissaShift), vdupq_n_u32(1));
        bits = vaddq_u32(bits, vaddq_u32(lsb, vdupq_n_u32(kBfloat16RoundingBias)));
        bits = vshrq_n_u32(bits, kBfloat16MantissaShift);
        bits = vbslq_u32(vceqq_f32(v, v), bits, vdupq_n_u32(kBfloat16QuietNan));
        vst1_u16(PtrToPtr<Eigen::bfloat16, uint16_t>(p), vmovn_u32(bits));
    }
    static void Step(const Eigen::bfloat16* a, const Eigen::bfloat16* b, Eigen::bfloat16* o)
    {
        Store(o, vmulq_f32(Load(a), Load(b)));
    }
    static void StepSplat(const Eigen::bfloat16* a, Eigen::bfloat16 v, Eigen::bfloat16* o)
    {
        Store(o, vmulq_f32(Load(a), vdupq_n_f32(static_cast<float>(v))));
    }
    static void StepSplatLeft(Eigen::bfloat16 v, const Eigen::bfloat16* b, Eigen::bfloat16* o)
    {
        Store(o, vmulq_f32(vdupq_n_f32(static_cast<float>(v)), Load(b)));
    }
};

/* NEON has no 64-bit integer vector multiply. The gain here is halving the load and store
 * instruction count -- the two products are still scalar. Measured 0.826x against Eigen,
 * which does the same thing inside pmul<Packet2l>.
 */
template <>
struct MulNeon<int64_t> {
    static const int64_t kLanes = 2;
    static void Step(const int64_t* a, const int64_t* b, int64_t* o)
    {
        const int64x2_t va = vld1q_s64(a);
        const int64x2_t vb = vld1q_s64(b);
        int64x2_t r = vdupq_n_s64(MulValue(vgetq_lane_s64(va, 0), vgetq_lane_s64(vb, 0)));
        r = vsetq_lane_s64(MulValue(vgetq_lane_s64(va, 1), vgetq_lane_s64(vb, 1)), r, 1);
        vst1q_s64(o, r);
    }
    static void StepSplat(const int64_t* a, int64_t v, int64_t* o)
    {
        const int64x2_t va = vld1q_s64(a);
        int64x2_t r = vdupq_n_s64(MulValue(vgetq_lane_s64(va, 0), v));
        r = vsetq_lane_s64(MulValue(vgetq_lane_s64(va, 1), v), r, 1);
        vst1q_s64(o, r);
    }
    static void StepSplatLeft(int64_t v, const int64_t* b, int64_t* o)
    {
        const int64x2_t vb = vld1q_s64(b);
        int64x2_t r = vdupq_n_s64(MulValue(v, vgetq_lane_s64(vb, 0)));
        r = vsetq_lane_s64(MulValue(v, vgetq_lane_s64(vb, 1)), r, 1);
        vst1q_s64(o, r);
    }
};

template <>
struct MulNeon<uint64_t> {
    static const int64_t kLanes = 2;
    static void Step(const uint64_t* a, const uint64_t* b, uint64_t* o)
    {
        const uint64x2_t va = vld1q_u64(a);
        const uint64x2_t vb = vld1q_u64(b);
        uint64x2_t r = vdupq_n_u64(vgetq_lane_u64(va, 0) * vgetq_lane_u64(vb, 0));
        r = vsetq_lane_u64(vgetq_lane_u64(va, 1) * vgetq_lane_u64(vb, 1), r, 1);
        vst1q_u64(o, r);
    }
    static void StepSplat(const uint64_t* a, uint64_t v, uint64_t* o)
    {
        const uint64x2_t va = vld1q_u64(a);
        uint64x2_t r = vdupq_n_u64(vgetq_lane_u64(va, 0) * v);
        r = vsetq_lane_u64(vgetq_lane_u64(va, 1) * v, r, 1);
        vst1q_u64(o, r);
    }
    static void StepSplatLeft(uint64_t v, const uint64_t* b, uint64_t* o)
    {
        const uint64x2_t vb = vld1q_u64(b);
        uint64x2_t r = vdupq_n_u64(v * vgetq_lane_u64(vb, 0));
        r = vsetq_lane_u64(v * vgetq_lane_u64(vb, 1), r, 1);
        vst1q_u64(o, r);
    }
};

/* (a.re, a.im) * (b.re, b.im) = (a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re), done two
 * complex values at a time. Same operation order as Eigen's pmul<Packet2cf>.
 */
template <>
struct MulNeon<std::complex<float>> {
    static const int64_t kLanes = 2;
    static float32x4_t Mul(float32x4_t va, float32x4_t vb)
    {
        const float32x4_t t1 = vmulq_f32(vtrn1q_f32(va, va), vb);
        float32x4_t t2 = vmulq_f32(vtrn2q_f32(va, va), vrev64q_f32(vb));
        const uint32x4_t flip = {0x80000000U, 0U, 0x80000000U, 0U};
        t2 = vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(t2), flip));
        return vaddq_f32(t1, t2);
    }
    static void Step(const std::complex<float>* a, const std::complex<float>* b, std::complex<float>* o)
    {
        const float32x4_t va = vld1q_f32(PtrToPtr<const std::complex<float>, const float>(a));
        const float32x4_t vb = vld1q_f32(PtrToPtr<const std::complex<float>, const float>(b));
        vst1q_f32(PtrToPtr<std::complex<float>, float>(o), Mul(va, vb));
    }
    static void StepSplat(const std::complex<float>* a, std::complex<float> v, std::complex<float>* o)
    {
        const float32x4_t va = vld1q_f32(PtrToPtr<const std::complex<float>, const float>(a));
        const float pair[4] = {v.real(), v.imag(), v.real(), v.imag()};
        vst1q_f32(PtrToPtr<std::complex<float>, float>(o), Mul(va, vld1q_f32(pair)));
    }
    static void StepSplatLeft(std::complex<float> v, const std::complex<float>* b, std::complex<float>* o)
    {
        const float pair[4] = {v.real(), v.imag(), v.real(), v.imag()};
        const float32x4_t vb = vld1q_f32(PtrToPtr<const std::complex<float>, const float>(b));
        vst1q_f32(PtrToPtr<std::complex<float>, float>(o), Mul(vld1q_f32(pair), vb));
    }
};

template <>
struct MulNeon<std::complex<double>> {
    static const int64_t kLanes = 1;
    static float64x2_t Mul(float64x2_t va, float64x2_t vb)
    {
        const float64x2_t t1 = vmulq_f64(vdupq_laneq_f64(va, 0), vb);
        float64x2_t t2 = vmulq_f64(vdupq_laneq_f64(va, 1), vextq_f64(vb, vb, 1));
        const uint64x2_t flip = {0x8000000000000000ULL, 0ULL};
        t2 = vreinterpretq_f64_u64(veorq_u64(vreinterpretq_u64_f64(t2), flip));
        return vaddq_f64(t1, t2);
    }
    static void Step(const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* o)
    {
        const float64x2_t va = vld1q_f64(PtrToPtr<const std::complex<double>, const double>(a));
        const float64x2_t vb = vld1q_f64(PtrToPtr<const std::complex<double>, const double>(b));
        vst1q_f64(PtrToPtr<std::complex<double>, double>(o), Mul(va, vb));
    }
    static void StepSplat(const std::complex<double>* a, std::complex<double> v, std::complex<double>* o)
    {
        const float64x2_t va = vld1q_f64(PtrToPtr<const std::complex<double>, const double>(a));
        const double pair[2] = {v.real(), v.imag()};
        vst1q_f64(PtrToPtr<std::complex<double>, double>(o), Mul(va, vld1q_f64(pair)));
    }
    static void StepSplatLeft(std::complex<double> v, const std::complex<double>* b, std::complex<double>* o)
    {
        const double pair[2] = {v.real(), v.imag()};
        const float64x2_t vb = vld1q_f64(PtrToPtr<const std::complex<double>, const double>(b));
        vst1q_f64(PtrToPtr<std::complex<double>, double>(o), Mul(vld1q_f64(pair), vb));
    }
};

template <>
struct MulPreparedSplat<std::complex<float>> {
    using Type = float32x4_t;
    static Type Prepare(std::complex<float> v)
    {
        const float32x2_t pair = {v.real(), v.imag()};
        return vcombine_f32(pair, pair);
    }
    static void Step(const std::complex<float>* a, Type v, std::complex<float>* o)
    {
        const float32x4_t va = vld1q_f32(PtrToPtr<const std::complex<float>, const float>(a));
        vst1q_f32(PtrToPtr<std::complex<float>, float>(o), MulNeon<std::complex<float>>::Mul(va, v));
    }
};

template <>
struct MulPreparedSplat<std::complex<double>> {
    using Type = float64x2_t;
    static Type Prepare(std::complex<double> v)
    {
        const float64x2_t pair = {v.real(), v.imag()};
        return pair;
    }
    static void Step(const std::complex<double>* a, Type v, std::complex<double>* o)
    {
        const float64x2_t va = vld1q_f64(PtrToPtr<const std::complex<double>, const double>(a));
        vst1q_f64(PtrToPtr<std::complex<double>, double>(o), MulNeon<std::complex<double>>::Mul(va, v));
    }
};

template <>
struct MulPreparedLeftSplat<std::complex<float>> {
    using Type = float32x4_t;
    static Type Prepare(std::complex<float> v)
    {
        const float32x2_t pair = {v.real(), v.imag()};
        return vcombine_f32(pair, pair);
    }
    static void Step(Type v, const std::complex<float>* b, std::complex<float>* o)
    {
        const float32x4_t vb = vld1q_f32(PtrToPtr<const std::complex<float>, const float>(b));
        vst1q_f32(PtrToPtr<std::complex<float>, float>(o), MulNeon<std::complex<float>>::Mul(v, vb));
    }
};

template <>
struct MulPreparedLeftSplat<std::complex<double>> {
    using Type = float64x2_t;
    static Type Prepare(std::complex<double> v)
    {
        const float64x2_t pair = {v.real(), v.imag()};
        return pair;
    }
    static void Step(Type v, const std::complex<double>* b, std::complex<double>* o)
    {
        const float64x2_t vb = vld1q_f64(PtrToPtr<const std::complex<double>, const double>(b));
        vst1q_f64(PtrToPtr<std::complex<double>, double>(o), MulNeon<std::complex<double>>::Mul(v, vb));
    }
};

/* The dtypes aarch64 multiplies with one instruction. Writing them out rather than leaving
 * them to the auto-vectoriser is what makes the kernel's speed reproducible: the blocked
 * loops below do vectorise, but whether GCC takes them shifts with unrelated edits to this
 * file, which showed up as the same case measuring anywhere from 0.13x to 1.04x.
 */
#define MUL_NEON_DIRECT(T, LANES, VT, LOAD, STORE, MUL, DUP)                                 \
    template <>                                                                              \
    struct MulNeon<T> {                                                                      \
        static const int64_t kLanes = LANES;                                                 \
        static void Step(const T* a, const T* b, T* o) { STORE(o, MUL(LOAD(a), LOAD(b))); }  \
        static void StepSplat(const T* a, T v, T* o) { STORE(o, MUL(LOAD(a), DUP(v))); }     \
        static void StepSplatLeft(T v, const T* b, T* o) { STORE(o, MUL(DUP(v), LOAD(b))); } \
    }
MUL_NEON_DIRECT(int8_t, 16, int8x16_t, vld1q_s8, vst1q_s8, vmulq_s8, vdupq_n_s8);
MUL_NEON_DIRECT(uint8_t, 16, uint8x16_t, vld1q_u8, vst1q_u8, vmulq_u8, vdupq_n_u8);
MUL_NEON_DIRECT(int16_t, 8, int16x8_t, vld1q_s16, vst1q_s16, vmulq_s16, vdupq_n_s16);
MUL_NEON_DIRECT(uint16_t, 8, uint16x8_t, vld1q_u16, vst1q_u16, vmulq_u16, vdupq_n_u16);
MUL_NEON_DIRECT(int32_t, 4, int32x4_t, vld1q_s32, vst1q_s32, vmulq_s32, vdupq_n_s32);
MUL_NEON_DIRECT(uint32_t, 4, uint32x4_t, vld1q_u32, vst1q_u32, vmulq_u32, vdupq_n_u32);
MUL_NEON_DIRECT(float, 4, float32x4_t, vld1q_f32, vst1q_f32, vmulq_f32, vdupq_n_f32);
MUL_NEON_DIRECT(double, 2, float64x2_t, vld1q_f64, vst1q_f64, vmulq_f64, vdupq_n_f64);
#undef MUL_NEON_DIRECT

#endif // MUL_USE_NEON

#ifdef MUL_USE_X86_SIMD
constexpr int64_t kX86SimdBytes = 16;
constexpr int kX86DwordBytes = 4;
constexpr int kX86BfloatShift = 16;
constexpr int kX86DwordShift = 32;
constexpr int32_t kX86LowByteMask = 0xFF;
constexpr int32_t kX86BfloatRoundBias = 0x7FFF;
constexpr int32_t kX86BfloatQuietNan = 0x7FC0;
constexpr int kX86PackLowWords = _MM_SHUFFLE(2, 0, 2, 0);
MUL_HOT_INLINE __m128i MulX86Bytes(__m128i left, __m128i right)
{
    const __m128i zero = _mm_setzero_si128();
    const __m128i mask = _mm_set1_epi16(kX86LowByteMask);
    const __m128i low = _mm_and_si128(_mm_mullo_epi16(_mm_unpacklo_epi8(left, zero), _mm_unpacklo_epi8(right, zero)),
                                      mask);
    const __m128i high = _mm_and_si128(_mm_mullo_epi16(_mm_unpackhi_epi8(left, zero), _mm_unpackhi_epi8(right, zero)),
                                       mask);
    return _mm_packus_epi16(low, high);
}

MUL_HOT_INLINE __m128i MulX86Dwords(__m128i left, __m128i right)
{
    const __m128i even = _mm_mul_epu32(left, right);
    const __m128i odd = _mm_mul_epu32(_mm_srli_si128(left, kX86DwordBytes), _mm_srli_si128(right, kX86DwordBytes));
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(even, kX86PackLowWords), _mm_shuffle_epi32(odd, kX86PackLowWords));
}

MUL_HOT_INLINE __m128i MulX86Qwords(__m128i left, __m128i right)
{
    const __m128i left_high = _mm_srli_epi64(left, kX86DwordShift);
    const __m128i right_high = _mm_srli_epi64(right, kX86DwordShift);
    const __m128i cross = _mm_add_epi64(_mm_mul_epu32(left_high, right), _mm_mul_epu32(right_high, left));
    return _mm_add_epi64(_mm_slli_epi64(cross, kX86DwordShift), _mm_mul_epu32(left, right));
}

template <typename T>
struct MulX86IntegerCore {
    static const int64_t kLanes = kX86SimdBytes / static_cast<int64_t>(sizeof(T));

    static __m128i Multiply(__m128i left, __m128i right)
    {
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            return MulX86Bytes(left, right);
        } else if constexpr (sizeof(T) == sizeof(int16_t)) {
            return _mm_mullo_epi16(left, right);
        } else if constexpr (sizeof(T) == sizeof(int32_t)) {
            return MulX86Dwords(left, right);
        } else {
            return MulX86Qwords(left, right);
        }
    }

    static __m128i Splat(T value)
    {
        if constexpr (sizeof(T) == sizeof(int8_t)) {
            return _mm_set1_epi8(static_cast<int8_t>(value));
        } else if constexpr (sizeof(T) == sizeof(int16_t)) {
            return _mm_set1_epi16(static_cast<int16_t>(value));
        } else if constexpr (sizeof(T) == sizeof(int32_t)) {
            return _mm_set1_epi32(static_cast<int32_t>(value));
        } else {
            return _mm_set1_epi64x(static_cast<int64_t>(value));
        }
    }

    static void Step(const T* left, const T* right, T* output)
    {
        const __m128i left_vec = _mm_loadu_si128(PtrToPtr<const T, const __m128i>(left));
        const __m128i right_vec = _mm_loadu_si128(PtrToPtr<const T, const __m128i>(right));
        _mm_storeu_si128(PtrToPtr<T, __m128i>(output), Multiply(left_vec, right_vec));
    }

    static void StepSplat(const T* left, T right, T* output)
    {
        const __m128i left_vec = _mm_loadu_si128(PtrToPtr<const T, const __m128i>(left));
        _mm_storeu_si128(PtrToPtr<T, __m128i>(output), Multiply(left_vec, Splat(right)));
    }

    static void StepSplatLeft(T left, const T* right, T* output)
    {
        const __m128i right_vec = _mm_loadu_si128(PtrToPtr<const T, const __m128i>(right));
        _mm_storeu_si128(PtrToPtr<T, __m128i>(output), Multiply(Splat(left), right_vec));
    }
};

template <>
struct MulNeon<int8_t> : MulX86IntegerCore<int8_t> {};
template <>
struct MulNeon<uint8_t> : MulX86IntegerCore<uint8_t> {};
template <>
struct MulNeon<int16_t> : MulX86IntegerCore<int16_t> {};
template <>
struct MulNeon<uint16_t> : MulX86IntegerCore<uint16_t> {};
template <>
struct MulNeon<int32_t> : MulX86IntegerCore<int32_t> {};
template <>
struct MulNeon<uint32_t> : MulX86IntegerCore<uint32_t> {};
template <>
struct MulNeon<int64_t> : MulX86IntegerCore<int64_t> {};
template <>
struct MulNeon<uint64_t> : MulX86IntegerCore<uint64_t> {};

template <>
struct MulNeon<float> {
    static const int64_t kLanes = 4;
    static void Step(const float* left, const float* right, float* output)
    {
        _mm_storeu_ps(output, _mm_mul_ps(_mm_loadu_ps(left), _mm_loadu_ps(right)));
    }
    static void StepSplat(const float* left, float right, float* output)
    {
        _mm_storeu_ps(output, _mm_mul_ps(_mm_loadu_ps(left), _mm_set1_ps(right)));
    }
    static void StepSplatLeft(float left, const float* right, float* output)
    {
        _mm_storeu_ps(output, _mm_mul_ps(_mm_set1_ps(left), _mm_loadu_ps(right)));
    }
};

template <>
struct MulNeon<double> {
    static const int64_t kLanes = 2;
    static void Step(const double* left, const double* right, double* output)
    {
        _mm_storeu_pd(output, _mm_mul_pd(_mm_loadu_pd(left), _mm_loadu_pd(right)));
    }
    static void StepSplat(const double* left, double right, double* output)
    {
        _mm_storeu_pd(output, _mm_mul_pd(_mm_loadu_pd(left), _mm_set1_pd(right)));
    }
    static void StepSplatLeft(double left, const double* right, double* output)
    {
        _mm_storeu_pd(output, _mm_mul_pd(_mm_set1_pd(left), _mm_loadu_pd(right)));
    }
};

struct MulX86Bfloat16Core {
    static __m128 Load(const Eigen::bfloat16* input)
    {
        const __m128i raw = _mm_loadl_epi64(PtrToPtr<const Eigen::bfloat16, const __m128i>(input));
        return _mm_castsi128_ps(_mm_slli_epi32(_mm_unpacklo_epi16(raw, _mm_setzero_si128()), kX86BfloatShift));
    }

    static void Store(Eigen::bfloat16* output, __m128 value)
    {
        __m128i bits = _mm_castps_si128(value);
        const __m128i lsb = _mm_and_si128(_mm_srli_epi32(bits, kX86BfloatShift), _mm_set1_epi32(1));
        bits = _mm_srli_epi32(_mm_add_epi32(bits, _mm_add_epi32(lsb, _mm_set1_epi32(kX86BfloatRoundBias))),
                              kX86BfloatShift);
        const __m128i is_nan = _mm_castps_si128(_mm_cmpunord_ps(value, value));
        bits = _mm_or_si128(_mm_andnot_si128(is_nan, bits), _mm_and_si128(is_nan, _mm_set1_epi32(kX86BfloatQuietNan)));
        const __m128i low = _mm_shufflelo_epi16(bits, kX86PackLowWords);
        const __m128i high = _mm_srli_si128(_mm_shufflehi_epi16(bits, kX86PackLowWords), kX86SimdBytes / 2);
        _mm_storel_epi64(PtrToPtr<Eigen::bfloat16, __m128i>(output), _mm_unpacklo_epi32(low, high));
    }
};

template <>
struct MulNeon<Eigen::bfloat16> : MulX86Bfloat16Core {
    static const int64_t kLanes = 4;
    static void Step(const Eigen::bfloat16* left, const Eigen::bfloat16* right, Eigen::bfloat16* output)
    {
        Store(output, _mm_mul_ps(Load(left), Load(right)));
    }
    static void StepSplat(const Eigen::bfloat16* left, Eigen::bfloat16 right, Eigen::bfloat16* output)
    {
        Store(output, _mm_mul_ps(Load(left), _mm_set1_ps(static_cast<float>(right))));
    }
    static void StepSplatLeft(Eigen::bfloat16 left, const Eigen::bfloat16* right, Eigen::bfloat16* output)
    {
        Store(output, _mm_mul_ps(_mm_set1_ps(static_cast<float>(left)), Load(right)));
    }
};

#endif // MUL_USE_X86_SIMD

/**
 * @brief Right-align the two shapes, validate broadcastability and derive the output shape.
 * @return false when a dimension pair is neither equal nor broadcastable
 */
bool PadAndValidate(const std::vector<int64_t>& x, const std::vector<int64_t>& y, int32_t rank, int64_t* xp,
                    int64_t* yp, int64_t* out)
{
    const int32_t x_rank = static_cast<int32_t>(x.size());
    const int32_t y_rank = static_cast<int32_t>(y.size());
    for (int32_t i = 0; i < rank; ++i) {
        xp[i] = (i >= rank - x_rank) ? x[static_cast<size_t>(i - (rank - x_rank))] : 1;
        yp[i] = (i >= rank - y_rank) ? y[static_cast<size_t>(i - (rank - y_rank))] : 1;
        if ((xp[i] < 0) || (yp[i] < 0)) {
            return false;
        }
        if (xp[i] == yp[i]) {
            out[i] = xp[i];
        } else if (xp[i] == 1) {
            out[i] = yp[i];
        } else if (yp[i] == 1) {
            out[i] = xp[i];
        } else {
            return false;
        }
    }
    return true;
}

/**
 * @brief Effective stride per dimension: the operand's natural stride, or 0 where that
 *        dimension is broadcast. A 0 stride is what makes the expansion virtual.
 */
bool EffectiveStrides(const int64_t* padded, const int64_t* out, int32_t rank, int64_t* eff)
{
    int64_t natural[kMulMaxBcastDims];
    natural[rank - 1] = 1;
    for (int32_t d = rank - 2; d >= 0; --d) {
        if (!CheckedMulInt64(natural[d + 1], padded[d + 1], natural[d])) {
            return false;
        }
    }
    for (int32_t d = 0; d < rank; ++d) {
        eff[d] = (padded[d] == out[d]) ? natural[d] : 0;
    }
    return true;
}

int32_t CompactNonUnitDimensions(const int64_t* out, const int64_t* xe, const int64_t* ye, int32_t rank, int64_t* to,
                                 int64_t* tx, int64_t* ty)
{
    int32_t kept = 0;
    for (int32_t d = 0; d < rank; ++d) {
        if (out[d] != 1) {
            to[kept] = out[d];
            tx[kept] = xe[d];
            ty[kept] = ye[d];
            ++kept;
        }
    }
    return kept;
}

bool CanCollapseStride(int64_t current_stride, int64_t next_stride, int64_t next_extent, bool& can_collapse)
{
    can_collapse = (current_stride == 0) && (next_stride == 0);
    if (can_collapse) {
        return true;
    }
    int64_t contiguous_stride = 0;
    if (!CheckedMulInt64(next_stride, next_extent, contiguous_stride)) {
        return false;
    }
    can_collapse = current_stride == contiguous_stride;
    return true;
}

bool SetPlanTotalElements(MulBcastPlan& plan)
{
    int64_t total = 1;
    for (int32_t d = 0; d < plan.ndims; ++d) {
        if (!CheckedMulInt64(total, plan.out_shape[d], total)) {
            return false;
        }
    }
    plan.total_elements = total;
    return true;
}

/**
 * @brief Drop output dimensions of size 1, then merge adjacent dimensions that are already
 *        contiguous for both operands. Fewer, longer levels mean a longer innermost run.
 */
bool DropAndCollapse(const int64_t* out, const int64_t* xe, const int64_t* ye, int32_t rank, MulBcastPlan& plan)
{
    int64_t to[kMulMaxBcastDims];
    int64_t tx[kMulMaxBcastDims];
    int64_t ty[kMulMaxBcastDims];
    const int32_t kept = CompactNonUnitDimensions(out, xe, ye, rank, to, tx, ty);
    if (kept == 0) {
        plan.ndims = 1;
        plan.out_shape[0] = 1;
        plan.x_strides[0] = 0;
        plan.y_strides[0] = 0;
        plan.total_elements = 1;
        return true;
    }

    plan.out_shape[0] = to[0];
    plan.x_strides[0] = tx[0];
    plan.y_strides[0] = ty[0];
    int32_t n = 1;
    for (int32_t d = 1; d < kept; ++d) {
        bool x_ok = false;
        bool y_ok = false;
        if (!CanCollapseStride(plan.x_strides[n - 1], tx[d], to[d], x_ok) ||
            !CanCollapseStride(plan.y_strides[n - 1], ty[d], to[d], y_ok)) {
            return false;
        }
        if (x_ok && y_ok) {
            int64_t collapsed_shape = 0;
            if (!CheckedMulInt64(plan.out_shape[n - 1], to[d], collapsed_shape)) {
                return false;
            }
            plan.out_shape[n - 1] = collapsed_shape;
            plan.x_strides[n - 1] = tx[d];
            plan.y_strides[n - 1] = ty[d];
        } else {
            plan.out_shape[n] = to[d];
            plan.x_strides[n] = tx[d];
            plan.y_strides[n] = ty[d];
            ++n;
        }
    }
    plan.ndims = n;
    return SetPlanTotalElements(plan);
}

/**
 * @brief Whether the output tensor's own shape is the broadcast of the two input shapes,
 *        rank included. Comparing only the element count is not enough: {1} x {1} with an
 *        output declared {1, 1} has a matching count but a different rank, and the previous
 *        implementation rejected it because Bcast::GenerateBcastInfo compared shapes.
 */
bool BroadcastOutputShapeMatches(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape,
                                 const std::vector<int64_t>& out_shape)
{
    const size_t rank = std::max(x_shape.size(), y_shape.size());
    if (out_shape.size() != rank) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        const int64_t xd = (i >= rank - x_shape.size()) ? x_shape[i - (rank - x_shape.size())] : 1;
        const int64_t yd = (i >= rank - y_shape.size()) ? y_shape[i - (rank - y_shape.size())] : 1;
        if ((xd < 0) || (yd < 0) || (out_shape[i] < 0)) {
            return false;
        }
        if ((xd != yd) && (xd != 1) && (yd != 1)) {
            return false;
        }
        const int64_t expected_dim = (xd == yd) ? xd : ((xd == 1) ? yd : xd);
        if (out_shape[i] != expected_dim) {
            return false;
        }
    }
    return true;
}

bool OutputShapeMatches(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape,
                        const std::vector<int64_t>& out_shape)
{
    return (std::max(x_shape.size(), y_shape.size()) <= static_cast<size_t>(kMulMaxBcastDims)) &&
           BroadcastOutputShapeMatches(x_shape, y_shape, out_shape);
}

/**
 * @brief Build the iteration plan. Returns false when the rank exceeds kMulMaxBcastDims or
 *        the shapes do not broadcast.
 */
bool BuildMulBcastPlan(const std::vector<int64_t>& x_shape, const std::vector<int64_t>& y_shape, MulBcastPlan& plan)
{
    const int32_t rank = std::max(static_cast<int32_t>(x_shape.size()), static_cast<int32_t>(y_shape.size()));
    if ((rank <= 0) || (rank > kMulMaxBcastDims)) {
        return false;
    }
    int64_t xp[kMulMaxBcastDims];
    int64_t yp[kMulMaxBcastDims];
    int64_t out[kMulMaxBcastDims];
    if (!PadAndValidate(x_shape, y_shape, rank, xp, yp, out)) {
        return false;
    }
    int64_t xe[kMulMaxBcastDims];
    int64_t ye[kMulMaxBcastDims];
    if (!EffectiveStrides(xp, out, rank, xe) || !EffectiveStrides(yp, out, rank, ye)) {
        return false;
    }
    return DropAndCollapse(out, xe, ye, rank, plan);
}

/* ---- innermost runs ----------------------------------------------------------------
 * always_inline is load-bearing: the strip is entered once per innermost run, and leaving
 * it as a real call costs one call per run. Blocking the bodies is equally load-bearing:
 * without it the loops below stay scalar at the production -O2 and lose to Eigen on 8 of
 * 14 dtypes -- up to 6x on the same-shape path.
 */
/* Each run below walks the body in fixed-length blocks first and finishes the remainder one
 * element at a time. The block's trip count is a compile-time constant multiple of every
 * vector width, which is what gets it past -O2's very-cheap vectoriser cost model. The
 * blocks have to be written out here rather than funnelled through a shared helper taking a
 * callable: with the body behind a lambda GCC still inlines it but no longer vectorises it,
 * measured at 0 NEON instructions either way for all four dtypes tried.
 */

template <typename T>
MUL_HOT_INLINE void MulRunContiguous(const T* x, const T* y, T* o, int64_t n)
{
    int64_t i = 0;
    const int64_t lanes = MulNeon<T>::kLanes;
    if (lanes > 0) {
        const int64_t unroll_lanes = lanes * MulNeonUnroll<T>::kValue;
        for (; n - i >= unroll_lanes; i += unroll_lanes) {
            const T* const x_block = x + i;
            const T* const y_block = y + i;
            T* const out_block = o + i;
            for (int64_t block = 0; block < MulNeonUnroll<T>::kValue; ++block) {
                const int64_t offset = block * lanes;
                MulNeon<T>::Step(x_block + offset, y_block + offset, out_block + offset);
            }
        }
        for (; n - i >= lanes; i += lanes) {
            MulNeon<T>::Step(x + i, y + i, o + i);
        }
    } else if (!MulIsComplexT<T>::value) {
        for (; i + kMulVecBlock <= n; i += kMulVecBlock) {
            for (int64_t k = 0; k < kMulVecBlock; ++k) {
                o[i + k] = MulValue(x[i + k], y[i + k]);
            }
        }
        for (; i + kMulVecBlockSmall <= n; i += kMulVecBlockSmall) {
            for (int64_t k = 0; k < kMulVecBlockSmall; ++k) {
                o[i + k] = MulValue(x[i + k], y[i + k]);
            }
        }
    }
    for (; i < n; ++i) {
        o[i] = MulTail(x[i], y[i]);
    }
}

template <typename T>
MUL_HOT_INLINE void MulRunSplatY(const T* x, T v, T* o, int64_t n)
{
    int64_t i = 0;
    const int64_t lanes = MulNeon<T>::kLanes;
    if (lanes > 0) {
        const typename MulPreparedSplat<T>::Type splat = MulPreparedSplat<T>::Prepare(v);
        const int64_t unroll_lanes = lanes * kMulNeonUnroll;
        for (; n - i >= unroll_lanes; i += unroll_lanes) {
            for (int64_t block = 0; block < kMulNeonUnroll; ++block) {
                const int64_t offset = block * lanes;
                MulPreparedSplat<T>::Step(x + i + offset, splat, o + i + offset);
            }
        }
        for (; i + lanes <= n; i += lanes) {
            MulPreparedSplat<T>::Step(x + i, splat, o + i);
        }
    } else if (!MulIsComplexT<T>::value) {
        for (; i + kMulVecBlock <= n; i += kMulVecBlock) {
            for (int64_t k = 0; k < kMulVecBlock; ++k) {
                o[i + k] = MulValue(x[i + k], v);
            }
        }
        for (; i + kMulVecBlockSmall <= n; i += kMulVecBlockSmall) {
            for (int64_t k = 0; k < kMulVecBlockSmall; ++k) {
                o[i + k] = MulValue(x[i + k], v);
            }
        }
    }
    for (; i < n; ++i) {
        o[i] = MulTail(x[i], v);
    }
}

template <typename T>
MUL_HOT_INLINE void MulRunSplatX(T v, const T* y, T* o, int64_t n)
{
    int64_t i = 0;
    const int64_t lanes = MulNeon<T>::kLanes;
    if (lanes > 0) {
        const typename MulPreparedLeftSplat<T>::Type splat = MulPreparedLeftSplat<T>::Prepare(v);
        // NaN payload/sign propagation is operand-order-sensitive for floating and complex
        // dtypes, so the left-splat entry cannot reuse the right-splat operation by swapping
        // its operands.
        const int64_t unroll_lanes = lanes * kMulNeonUnroll;
        for (; n - i >= unroll_lanes; i += unroll_lanes) {
            for (int64_t block = 0; block < kMulNeonUnroll; ++block) {
                const int64_t offset = block * lanes;
                MulPreparedLeftSplat<T>::Step(splat, y + i + offset, o + i + offset);
            }
        }
        for (; i + lanes <= n; i += lanes) {
            MulPreparedLeftSplat<T>::Step(splat, y + i, o + i);
        }
    } else if (!MulIsComplexT<T>::value) {
        for (; i + kMulVecBlock <= n; i += kMulVecBlock) {
            for (int64_t k = 0; k < kMulVecBlock; ++k) {
                o[i + k] = MulValue(v, y[i + k]);
            }
        }
        for (; i + kMulVecBlockSmall <= n; i += kMulVecBlockSmall) {
            for (int64_t k = 0; k < kMulVecBlockSmall; ++k) {
                o[i + k] = MulValue(v, y[i + k]);
            }
        }
    }
    for (; i < n; ++i) {
        o[i] = MulTail(v, y[i]);
    }
}

/**
 * @brief Non-unit strides. A stride of 0 works here too, since i * 0 re-reads element 0.
 *        Blocking buys nothing: a gather does not vectorise either way.
 */
template <typename T>
MUL_HOT_INLINE void MulRunStrided(const T* x, const T* y, T* o, int64_t n, int64_t x_step, int64_t y_step)
{
    for (int64_t i = 0; i < n; ++i) {
        o[i] = MulTail(x[i * x_step], y[i * y_step]);
    }
}

/**
 * @brief Dispatch one innermost run on its two strides. The branch is loop-invariant, so it
 *        costs nothing beyond the first entry.
 */
template <typename T>
MUL_HOT_INLINE void MulStrip(const T* x, const T* y, T* o, int64_t n, int64_t x_step, int64_t y_step)
{
    if ((x_step == 1) && (y_step == 1)) {
        MulRunContiguous<T>(x, y, o, n);
    } else if ((y_step == 0) && (x_step == 1)) {
        MulRunSplatY<T>(x, *y, o, n);
    } else if ((x_step == 0) && (y_step == 1)) {
        MulRunSplatX<T>(*x, y, o, n);
    } else {
        MulRunStrided<T>(x, y, o, n, x_step, y_step);
    }
}

/**
 * @brief Same-shape and scalar-operand runs.
 */
template <typename T>
MUL_HOT void MulDriveFlat(const T* x, const T* y, T* out, int64_t n, int64_t x_step, int64_t y_step)
{
    MulStrip<T>(x, y, out, n, x_step, y_step);
}

/**
 * @brief Walk the plan: one coordinate decomposition per innermost run, then carry.
 */
MUL_HOT_INLINE void AdvanceMulCoordinates(const MulBcastPlan& plan, int32_t nd, int64_t* coords, int64_t& xo,
                                          int64_t& yo)
{
    for (int32_t d = nd - kMulStrideCarryOffset; d >= 0; --d) {
        coords[d]++;
        xo += plan.x_strides[d];
        yo += plan.y_strides[d];
        if (coords[d] < plan.out_shape[d]) {
            break;
        }
        xo -= plan.out_shape[d] * plan.x_strides[d];
        yo -= plan.out_shape[d] * plan.y_strides[d];
        coords[d] = 0;
    }
}

template <typename T>
MUL_HOT void MulDriveStrides(const T* x, const T* y, T* out, const MulBcastPlan& plan)
{
    const int32_t nd = plan.ndims;
    const int64_t inner = plan.out_shape[nd - 1];
    const int64_t x_step = plan.x_strides[nd - 1];
    const int64_t y_step = plan.y_strides[nd - 1];
    if (nd == 1) {
        MulStrip<T>(x, y, out, inner, x_step, y_step);
        return;
    }
    int64_t coords[kMulMaxBcastDims] = {0};
    int64_t xo = 0;
    int64_t yo = 0;
    T* out_run = out;
    const T* const out_end = out + plan.total_elements;
    for (; out_run < out_end; out_run += inner) {
        MulStrip<T>(x + xo, y + yo, out_run, inner, x_step, y_step);
        AdvanceMulCoordinates(plan, nd, coords, xo, yo);
    }
}

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
MUL_HOT void MulStripFloat(const float* x, const float* y, float* out, int64_t n, int64_t x_step, int64_t y_step)
{
    MulStrip<float>(x, y, out, n, x_step, y_step);
}

MUL_HOT MUL_HOST_NO_IVOPTS void MulDriveStridesFloat(const float* x, const float* y, float* out,
                                                     const MulBcastPlan& plan)
{
    const int32_t nd = plan.ndims;
    const int64_t inner = plan.out_shape[nd - 1];
    const int64_t x_step = plan.x_strides[nd - 1];
    const int64_t y_step = plan.y_strides[nd - 1];
    if (nd == 1) {
        MulStripFloat(x, y, out, inner, x_step, y_step);
        return;
    }
    int64_t coords[kMulMaxBcastDims] = {0};
    int64_t xo = 0;
    int64_t yo = 0;
    float* out_run = out;
    const float* const out_end = out + plan.total_elements;
    for (; out_run < out_end; out_run += inner) {
        MulStripFloat(x + xo, y + yo, out_run, inner, x_step, y_step);
        AdvanceMulCoordinates(plan, nd, coords, xo, yo);
    }
}
#endif

/**
 * @brief Replicate each of `rows` values `c` times into the tile.
 *
 * `c` is a runtime value, and a loop of three iterations costs more in loop overhead than in
 * stores -- measured, this fill and not the multiply was what made the whole tile path lose
 * for the four-byte dtypes. Dispatching the common short lengths to a compile-time constant
 * lets the inner loop unroll away.
 */
template <typename T, int64_t C>
MUL_HOT_INLINE void MulFillConst(const T* bcast, T* tile, int64_t rows)
{
    for (int64_t i = 0; i < rows; ++i) {
        const T v = bcast[i];
        for (int64_t j = 0; j < C; ++j) {
            tile[(i * C) + j] = v;
        }
    }
}

template <typename T>
MUL_HOT_INLINE void MulFillTile(const T* bcast, T* tile, int64_t rows, int64_t c)
{
    switch (c) {
        case kMulPairTileRun:
            MulFillConst<T, kMulPairTileRun>(bcast, tile, rows);
            return;
        case kMulTripletTileRun:
            MulFillConst<T, kMulTripletTileRun>(bcast, tile, rows);
            return;
        case kMulQuadTileRun:
            MulFillConst<T, kMulQuadTileRun>(bcast, tile, rows);
            return;
        case kMulOctetTileRun:
            MulFillConst<T, kMulOctetTileRun>(bcast, tile, rows);
            return;
        default:
            break;
    }
    for (int64_t i = 0; i < rows; ++i) {
        const T v = bcast[i];
        std::fill_n(tile + (i * c), c, v);
    }
}

/**
 * @brief The [rows, 1] x [rows, C] shape with a short C, done by expanding the broadcast
 *        operand into a small stack tile and multiplying long contiguous stretches.
 *
 * A strip loop handles this shape C elements at a time, which for C of 3 or 8 never reaches
 * a full vector and leaves the machine an order of magnitude short of memory bandwidth. The
 * tile turns it into one scalar fill into L1 plus one long vectorised multiply. The tile is
 * a fixed byte budget, so nothing here scales with the tensor.
 */
template <typename T, bool XIsBcast>
MUL_HOT void MulDriveShortInner(const T* x, const T* y, T* out, int64_t rows, int64_t c)
{
    if (c == 0) {
        return;
    }
    const T* bcast = XIsBcast ? x : y;
    const T* dense = XIsBcast ? y : x;
    constexpr int64_t kTileElems = kMulTileBytes / static_cast<int64_t>(sizeof(T));
    T tile[kTileElems];
    const int64_t per_tile = (kTileElems / c) > 0 ? (kTileElems / c) : 1;
    for (int64_t base = 0; base < rows; base += per_tile) {
        const int64_t r = ((rows - base) < per_tile) ? (rows - base) : per_tile;
        MulFillTile<T>(bcast + base, tile, r, c);
        if (XIsBcast) {
            MulRunContiguous<T>(tile, dense + (base * c), out + (base * c), r * c);
        } else {
            MulRunContiguous<T>(dense + (base * c), tile, out + (base * c), r * c);
        }
    }
}

/**
 * @brief Whether the plan is the two-level shape the tile driver handles: one operand
 *        broadcast along a short innermost run, the other contiguous across both levels.
 */
template <typename T>
bool IsShortInnerTile(const MulBcastPlan& plan, bool& x_is_bcast)
{
    if (plan.ndims != kMulShortInnerPlanDims) {
        return false;
    }
    const int64_t c = plan.out_shape[1];
    if (c >= (kMulTileMaxRunBytes / static_cast<int64_t>(sizeof(T)))) {
        return false;
    }
    if ((plan.x_strides[1] == 0) && (plan.y_strides[1] == 1) && (plan.x_strides[0] == 1) && (plan.y_strides[0] == c)) {
        x_is_bcast = true;
        return true;
    }
    if ((plan.y_strides[1] == 0) && (plan.x_strides[1] == 1) && (plan.y_strides[0] == 1) && (plan.x_strides[0] == c)) {
        x_is_bcast = false;
        return true;
    }
    return false;
}

} // namespace

template <typename T>
uint32_t MulCpuKernel::MulNoBcast(const CpuKernelContext& ctx, int64_t x_num, int64_t y_num, int64_t out_num) const
{
    T* x = static_cast<T*>(ctx.Input(kFirstInputIndex)->GetData());
    T* y = static_cast<T*>(ctx.Input(kSecondInputIndex)->GetData());
    T* out = static_cast<T*>(ctx.Output(kFirstOutputIndex)->GetData());
    if (x_num == y_num) {
        MulDriveFlat<T>(x, y, out, out_num, 1, 1);
    } else if (x_num == 1) {
        MulDriveFlat<T>(x, y, out, out_num, 0, 1);
    } else {
        MulDriveFlat<T>(x, y, out, out_num, 1, 0);
    }
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MulCpuKernel::MulBcastByStride(const CpuKernelContext& ctx, const MulBcastPlan& plan) const
{
    T* x = static_cast<T*>(ctx.Input(kFirstInputIndex)->GetData());
    T* y = static_cast<T*>(ctx.Input(kSecondInputIndex)->GetData());
    T* out = static_cast<T*>(ctx.Output(kFirstOutputIndex)->GetData());
    bool x_is_bcast = false;
    if (IsShortInnerTile<T>(plan, x_is_bcast)) {
        if (x_is_bcast) {
            MulDriveShortInner<T, true>(x, y, out, plan.out_shape[0], plan.out_shape[1]);
        } else {
            MulDriveShortInner<T, false>(x, y, out, plan.out_shape[0], plan.out_shape[1]);
        }
        return KERNEL_STATUS_OK;
    }
#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__)
    if constexpr (std::is_same<T, float>::value) {
        MulDriveStridesFloat(x, y, out, plan);
    } else {
        MulDriveStrides<T>(x, y, out, plan);
    }
#else
    MulDriveStrides<T>(x, y, out, plan);
#endif
    return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t MulCpuKernel::MulCompute(const CpuKernelContext& ctx) const
{
    Tensor* x_tensor = ctx.Input(kFirstInputIndex);
    Tensor* y_tensor = ctx.Input(kSecondInputIndex);
    Tensor* out_tensor = ctx.Output(kFirstOutputIndex);
    KERNEL_CHECK_NULLPTR(x_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 0 data failed.",
                         ctx.GetOpType().c_str())
    KERNEL_CHECK_NULLPTR(y_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get input 1 data failed.",
                         ctx.GetOpType().c_str())
    KERNEL_CHECK_NULLPTR(out_tensor->GetData(), KERNEL_STATUS_PARAM_INVALID, "[%s] Get output data failed.",
                         ctx.GetOpType().c_str())

    std::vector<int64_t> x_shape = x_tensor->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> y_shape = y_tensor->GetTensorShape()->GetDimSizes();
    std::vector<int64_t> out_shape = out_tensor->GetTensorShape()->GetDimSizes();
    KERNEL_CHECK_FALSE(OutputShapeMatches(x_shape, y_shape, out_shape), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Input rank exceeds the supported limit or output shape does not match the broadcast.",
                       ctx.GetOpType().c_str())

    const bool same_shape = x_shape == y_shape;
    const bool x_has_one_element = ShapeHasOneElement(x_shape);
    const bool y_has_one_element = ShapeHasOneElement(y_shape);
    if (same_shape || x_has_one_element || y_has_one_element) {
        int64_t out_num = 0;
        KERNEL_CHECK_FALSE(CheckedShapeElements(out_shape, out_num), KERNEL_STATUS_PARAM_INVALID,
                           "[%s] Shape contains a negative dimension or its element count exceeds the int64 range.",
                           ctx.GetOpType().c_str())
        const int64_t x_num = x_has_one_element ? 1 : out_num;
        const int64_t y_num = y_has_one_element ? 1 : out_num;
        return MulNoBcast<T>(ctx, x_num, y_num, out_num);
    }

    const size_t raw_rank = std::max(x_shape.size(), y_shape.size());
    KERNEL_CHECK_FALSE(raw_rank <= static_cast<size_t>(kMulMaxBcastDims), KERNEL_STATUS_PARAM_INVALID,
                       "[%s] Rank of output should less than [%d] but get [%zu].", ctx.GetOpType().c_str(),
                       kMulMaxBcastDims, raw_rank)

    MulBcastPlan plan;
    if (!BuildMulBcastPlan(x_shape, y_shape, plan)) {
        KERNEL_LOG_ERROR("[%s] Generate broadcast info failed.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return MulBcastByStride<T>(ctx, plan);
}

uint32_t MulCpuKernel::MulSameTypeCompute(const CpuKernelContext& ctx) const
{
    auto data_type = static_cast<DataType>(ctx.Input(kFirstInputIndex)->GetDataType());
    KERNEL_LOG_INFO("%s Compute begin, dtype[%s].", kMul, DTypeStr(data_type).c_str());
    switch (data_type) {
        case DT_FLOAT16:
            return MulCompute<Eigen::half>(ctx);
        case DT_BFLOAT16:
            return MulCompute<Eigen::bfloat16>(ctx);
        case DT_FLOAT:
            return MulCompute<float>(ctx);
        case DT_DOUBLE:
            return MulCompute<double>(ctx);
        case DT_INT8:
            return MulCompute<int8_t>(ctx);
        case DT_INT16:
            return MulCompute<int16_t>(ctx);
        case DT_INT32:
            return MulCompute<int32_t>(ctx);
        case DT_INT64:
            return MulCompute<int64_t>(ctx);
        case DT_UINT8:
            return MulCompute<uint8_t>(ctx);
        case DT_UINT16:
            return MulCompute<uint16_t>(ctx);
        case DT_UINT32:
            return MulCompute<uint32_t>(ctx);
        case DT_UINT64:
            return MulCompute<uint64_t>(ctx);
        case DT_COMPLEX64:
            return MulCompute<std::complex<float>>(ctx);
        case DT_COMPLEX128:
            return MulCompute<std::complex<double>>(ctx);
        default:
            KERNEL_LOG_ERROR("[%s] Data type of input is not supported, input data type is [%s].",
                             ctx.GetOpType().c_str(), DTypeStr(data_type).c_str());
            return KERNEL_STATUS_PARAM_INVALID;
    }
}

int64_t GetMulParallelCoreNum(const CpuKernelContext& ctx, int64_t data_num)
{
    uint32_t min_core_num = 1;
    int64_t max_core_num = std::max(static_cast<int64_t>(min_core_num),
                                    static_cast<int64_t>(aicpu::CpuKernelUtils::GetCPUNum(ctx)) - kResvCpuNum);
    if (data_num <= kParallelDataNumMid) {
        max_core_num = std::min(max_core_num, kMaxCoreNumForMul);
    }
    if (max_core_num > data_num) {
        max_core_num = data_num;
    }
    if (max_core_num < 1) {
        max_core_num = 1;
    }
    return max_core_num;
}

/* Diff-type keeps the library operator* for complex operands, which is what the previous
 * implementation called, so its special-value behaviour stays bit-for-bit identical; the
 * four-multiply form is deliberately confined to the same-dtype paths. The non-complex
 * overloads keep the exact body the previous implementation had. */
template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_same<TIn1, TOut>::value && MulIsComplexT<TIn1>::value, void>::type inline MulImpl(
    TIn1 a, TIn2 b, TOut& output)
{
    output = a * static_cast<TIn1>(b);
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_same<TIn1, TOut>::value && !MulIsComplexT<TIn1>::value, void>::type inline MulImpl(
    TIn1 a, TIn2 b, TOut& output)
{
    output = MulValue(a, static_cast<TIn1>(b));
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_same<TIn2, TOut>::value && MulIsComplexT<TIn2>::value, void>::type inline MulImpl(
    TIn1 a, TIn2 b, TOut& output)
{
    output = static_cast<TIn2>(a) * b;
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<std::is_same<TIn2, TOut>::value && !MulIsComplexT<TIn2>::value, void>::type inline MulImpl(
    TIn1 a, TIn2 b, TOut& output)
{
    output = MulValue(static_cast<TIn2>(a), b);
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<!std::is_same<TIn1, TOut>::value && !std::is_same<TIn2, TOut>::value &&
                            MulIsComplexT<TOut>::value,
                        void>::type inline MulImpl(TIn1 a, TIn2 b, TOut& output)
{
    output = static_cast<TOut>(a) * static_cast<TOut>(b);
}

template <typename TIn1, typename TIn2, typename TOut>
typename std::enable_if<!std::is_same<TIn1, TOut>::value && !std::is_same<TIn2, TOut>::value &&
                            !MulIsComplexT<TOut>::value,
                        void>::type inline MulImpl(TIn1 a, TIn2 b, TOut& output)
{
    output = MulValue(static_cast<TOut>(a), static_cast<TOut>(b));
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t BcastCompute(const CpuKernelContext& ctx, const Bcast& bcast, int64_t data_num)
{
    auto in0 = PtrToPtr<void, TIn1>(ctx.Input(0)->GetData());
    auto in1 = PtrToPtr<void, TIn2>(ctx.Input(1)->GetData());
    auto out = PtrToPtr<void, TOut>(ctx.Output(0)->GetData());
    if (data_num >= kParallelDataNum) {
        int64_t max_core_num = GetMulParallelCoreNum(ctx, data_num);
        if (max_core_num == 0) {
            KERNEL_LOG_ERROR("Mul max_core_num is zero, division by zero.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        auto sharder_mul = [&in0, &in1, &out, &bcast](int64_t start, int64_t end) {
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*(in0 + bcast.GetBroadcastXIndex(i)), *(in1 + bcast.GetBroadcastYIndex(i)), *(out + i));
            }
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_mul),
                            "Mul Compute failed.")
    } else {
        for (int64_t i = 0; i < data_num; ++i) {
            MulImpl(*(in0 + bcast.GetBroadcastXIndex(i)), *(in1 + bcast.GetBroadcastYIndex(i)), *(out + i));
        }
    }
    return KERNEL_STATUS_OK;
}

template <typename TIn1, typename TIn2, typename TOut>
void SpecialCompute(BcastShapeType type, int64_t start, int64_t end, CpuKernelContext& ctx)
{
    auto in1 = PtrToPtr<void, TIn1>(ctx.Input(0)->GetData());
    auto in2 = PtrToPtr<void, TIn2>(ctx.Input(1)->GetData());
    auto output = PtrToPtr<void, TOut>(ctx.Output(0)->GetData());
    switch (type) {
        case BcastShapeType::SAME_SHAPE:
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*(in1 + i), *(in2 + i), *(output + i));
            }
            break;
        case BcastShapeType::X_ONE_ELEMENT:
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*in1, *(in2 + i), *(output + i));
            }
            break;
        case BcastShapeType::Y_ONE_ELEMENT:
            for (int64_t i = start; i < end; ++i) {
                MulImpl(*(in1 + i), *in2, *(output + i));
            }
            break;
        default:
            KERNEL_LOG_WARN("Invalid type [%d]", static_cast<int32_t>(type));
            break;
    }
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t NoBcastCompute(CpuKernelContext& ctx, int64_t element_num_in0, int64_t element_num_in1, int64_t data_num)
{
    BcastShapeType type = (element_num_in0 == element_num_in1 ?
                               BcastShapeType::SAME_SHAPE :
                               (element_num_in0 == 1 ? BcastShapeType::X_ONE_ELEMENT : BcastShapeType::Y_ONE_ELEMENT));
    if (data_num >= kParallelDataNumSameShape) {
        int64_t max_core_num = GetMulParallelCoreNum(ctx, data_num);
        if (max_core_num == 0) {
            KERNEL_LOG_ERROR("Mul max_core_num is zero, division by zero.");
            return KERNEL_STATUS_PARAM_INVALID;
        }
        auto sharder_mul = [type, &ctx](int64_t start, int64_t end) {
            SpecialCompute<TIn1, TIn2, TOut>(type, start, end, ctx);
        };
        KERNEL_HANDLE_ERROR(CpuKernelUtils::ParallelFor(ctx, data_num, data_num / max_core_num, sharder_mul),
                            "Mul Compute failed.")
    } else {
        SpecialCompute<TIn1, TIn2, TOut>(type, 0, data_num, ctx);
    }
    return KERNEL_STATUS_OK;
}

template <typename TIn1, typename TIn2, typename TOut>
uint32_t MulDiffTypeCompute(CpuKernelContext& ctx, const MulElementCounts& elements, const Bcast* bcast)
{
    if (bcast == nullptr) {
        return NoBcastCompute<TIn1, TIn2, TOut>(ctx, elements.input0, elements.input1, elements.output);
    }
    return BcastCompute<TIn1, TIn2, TOut>(ctx, *bcast, elements.output);
}

using MulDiffTypeCall = uint32_t (*)(CpuKernelContext&, const MulElementCounts&, const Bcast*);

uint32_t DispatchMulDiffType(CpuKernelContext& ctx, const MulDiffTypeCall& compute)
{
    Tensor* tensor_in0 = ctx.Input(0);
    auto shape_in0 = tensor_in0->GetTensorShape()->GetDimSizes();
    Tensor* tensor_in1 = ctx.Input(1);
    auto shape_in1 = tensor_in1->GetTensorShape()->GetDimSizes();
    MulElementCounts elements;
    {
        const auto shape_out = ctx.Output(0)->GetTensorShape()->GetDimSizes();
        if (!GetMulElementCounts(ctx, shape_in0, shape_in1, shape_out, elements)) {
            return KERNEL_STATUS_PARAM_INVALID;
        }
        if (!BroadcastOutputShapeMatches(shape_in0, shape_in1, shape_out)) {
            KERNEL_LOG_ERROR("[%s] Output shape does not match the broadcast of the input shapes.",
                             ctx.GetOpType().c_str());
            return KERNEL_STATUS_PARAM_INVALID;
        }
    }

    bool no_need_bcast = (shape_in0 == shape_in1) || (elements.input0 == 1) || (elements.input1 == 1);
    if (no_need_bcast) {
        return compute(ctx, elements, nullptr);
    }

    Bcast bcast(shape_in0, shape_in1);
    if (!bcast.IsValid()) {
        KERNEL_LOG_ERROR("[%s] broadcast failed.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return compute(ctx, elements, &bcast);
}

using MulDiffTypeRow = std::unordered_map<int32_t, MulDiffTypeCall>;
using MulDiffTypeCalls = std::unordered_map<int32_t, MulDiffTypeRow>;

MulDiffTypeRow MakeUint8MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<uint8_t, int8_t, int16_t>},
            {DT_INT16, MulDiffTypeCompute<uint8_t, int16_t, int16_t>},
            {DT_INT32, MulDiffTypeCompute<uint8_t, int32_t, int32_t>},
            {DT_INT64, MulDiffTypeCompute<uint8_t, int64_t, int64_t>},
            {DT_BFLOAT16, MulDiffTypeCompute<uint8_t, Eigen::bfloat16, Eigen::bfloat16>},
            {DT_FLOAT16, MulDiffTypeCompute<uint8_t, Eigen::half, Eigen::half>},
            {DT_FLOAT, MulDiffTypeCompute<uint8_t, float, float>},
            {DT_DOUBLE, MulDiffTypeCompute<uint8_t, double, double>},
            {DT_COMPLEX64, MulDiffTypeCompute<uint8_t, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<uint8_t, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeInt8MulDiffTypeRow()
{
    return {{DT_INT16, MulDiffTypeCompute<int8_t, int16_t, int16_t>},
            {DT_INT32, MulDiffTypeCompute<int8_t, int32_t, int32_t>},
            {DT_INT64, MulDiffTypeCompute<int8_t, int64_t, int64_t>},
            {DT_BFLOAT16, MulDiffTypeCompute<int8_t, Eigen::bfloat16, Eigen::bfloat16>},
            {DT_FLOAT16, MulDiffTypeCompute<int8_t, Eigen::half, Eigen::half>},
            {DT_FLOAT, MulDiffTypeCompute<int8_t, float, float>},
            {DT_DOUBLE, MulDiffTypeCompute<int8_t, double, double>},
            {DT_UINT8, MulDiffTypeCompute<int8_t, uint8_t, int16_t>},
            {DT_COMPLEX64, MulDiffTypeCompute<int8_t, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<int8_t, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeInt16MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<int16_t, int8_t, int16_t>},
            {DT_INT32, MulDiffTypeCompute<int16_t, int32_t, int32_t>},
            {DT_INT64, MulDiffTypeCompute<int16_t, int64_t, int64_t>},
            {DT_BFLOAT16, MulDiffTypeCompute<int16_t, Eigen::bfloat16, Eigen::bfloat16>},
            {DT_FLOAT16, MulDiffTypeCompute<int16_t, Eigen::half, Eigen::half>},
            {DT_FLOAT, MulDiffTypeCompute<int16_t, float, float>},
            {DT_DOUBLE, MulDiffTypeCompute<int16_t, double, double>},
            {DT_UINT8, MulDiffTypeCompute<int16_t, uint8_t, int16_t>},
            {DT_COMPLEX64, MulDiffTypeCompute<int16_t, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<int16_t, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeInt32MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<int32_t, int8_t, int32_t>},
            {DT_INT16, MulDiffTypeCompute<int32_t, int16_t, int32_t>},
            {DT_INT64, MulDiffTypeCompute<int32_t, int64_t, int64_t>},
            {DT_BFLOAT16, MulDiffTypeCompute<int32_t, Eigen::bfloat16, Eigen::bfloat16>},
            {DT_FLOAT16, MulDiffTypeCompute<int32_t, Eigen::half, Eigen::half>},
            {DT_FLOAT, MulDiffTypeCompute<int32_t, float, float>},
            {DT_DOUBLE, MulDiffTypeCompute<int32_t, double, double>},
            {DT_UINT8, MulDiffTypeCompute<int32_t, uint8_t, int32_t>},
            {DT_COMPLEX64, MulDiffTypeCompute<int32_t, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<int32_t, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeInt64MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<int64_t, int8_t, int64_t>},
            {DT_INT16, MulDiffTypeCompute<int64_t, int16_t, int64_t>},
            {DT_INT32, MulDiffTypeCompute<int64_t, int32_t, int64_t>},
            {DT_BFLOAT16, MulDiffTypeCompute<int64_t, Eigen::bfloat16, Eigen::bfloat16>},
            {DT_FLOAT16, MulDiffTypeCompute<int64_t, Eigen::half, Eigen::half>},
            {DT_FLOAT, MulDiffTypeCompute<int64_t, float, float>},
            {DT_DOUBLE, MulDiffTypeCompute<int64_t, double, double>},
            {DT_UINT8, MulDiffTypeCompute<int64_t, uint8_t, int64_t>},
            {DT_COMPLEX64, MulDiffTypeCompute<int64_t, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<int64_t, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeBfloat16MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<Eigen::bfloat16, int8_t, Eigen::bfloat16>},
            {DT_INT16, MulDiffTypeCompute<Eigen::bfloat16, int16_t, Eigen::bfloat16>},
            {DT_INT32, MulDiffTypeCompute<Eigen::bfloat16, int32_t, Eigen::bfloat16>},
            {DT_INT64, MulDiffTypeCompute<Eigen::bfloat16, int64_t, Eigen::bfloat16>},
            {DT_FLOAT16, MulDiffTypeCompute<Eigen::bfloat16, Eigen::half, float>},
            {DT_FLOAT, MulDiffTypeCompute<Eigen::bfloat16, float, float>},
            {DT_DOUBLE, MulDiffTypeCompute<Eigen::bfloat16, double, double>},
            {DT_UINT8, MulDiffTypeCompute<Eigen::bfloat16, uint8_t, Eigen::bfloat16>},
            {DT_COMPLEX64, MulDiffTypeCompute<Eigen::bfloat16, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<Eigen::bfloat16, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeFloat16MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<Eigen::half, int8_t, Eigen::half>},
            {DT_INT16, MulDiffTypeCompute<Eigen::half, int16_t, Eigen::half>},
            {DT_INT32, MulDiffTypeCompute<Eigen::half, int32_t, Eigen::half>},
            {DT_INT64, MulDiffTypeCompute<Eigen::half, int64_t, Eigen::half>},
            {DT_FLOAT, MulDiffTypeCompute<Eigen::half, float, float>},
            {DT_BFLOAT16, MulDiffTypeCompute<Eigen::half, Eigen::bfloat16, float>},
            {DT_DOUBLE, MulDiffTypeCompute<Eigen::half, double, double>},
            {DT_UINT8, MulDiffTypeCompute<Eigen::half, uint8_t, Eigen::half>},
            {DT_COMPLEX64, MulDiffTypeCompute<Eigen::half, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<Eigen::half, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeFloatMulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<float, int8_t, float>},
            {DT_INT16, MulDiffTypeCompute<float, int16_t, float>},
            {DT_INT32, MulDiffTypeCompute<float, int32_t, float>},
            {DT_INT64, MulDiffTypeCompute<float, int64_t, float>},
            {DT_BFLOAT16, MulDiffTypeCompute<float, Eigen::bfloat16, float>},
            {DT_FLOAT16, MulDiffTypeCompute<float, Eigen::half, float>},
            {DT_DOUBLE, MulDiffTypeCompute<float, double, double>},
            {DT_UINT8, MulDiffTypeCompute<float, uint8_t, float>},
            {DT_COMPLEX64, MulDiffTypeCompute<float, std::complex<float>, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<float, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeDoubleMulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<double, int8_t, double>},
            {DT_INT16, MulDiffTypeCompute<double, int16_t, double>},
            {DT_INT32, MulDiffTypeCompute<double, int32_t, double>},
            {DT_INT64, MulDiffTypeCompute<double, int64_t, double>},
            {DT_BFLOAT16, MulDiffTypeCompute<double, Eigen::bfloat16, double>},
            {DT_FLOAT16, MulDiffTypeCompute<double, Eigen::half, double>},
            {DT_FLOAT, MulDiffTypeCompute<double, float, double>},
            {DT_UINT8, MulDiffTypeCompute<double, uint8_t, double>},
            {DT_COMPLEX64, MulDiffTypeCompute<double, std::complex<float>, std::complex<double>>},
            {DT_COMPLEX128, MulDiffTypeCompute<double, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeComplex64MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<std::complex<float>, int8_t, std::complex<float>>},
            {DT_INT16, MulDiffTypeCompute<std::complex<float>, int16_t, std::complex<float>>},
            {DT_INT32, MulDiffTypeCompute<std::complex<float>, int32_t, std::complex<float>>},
            {DT_INT64, MulDiffTypeCompute<std::complex<float>, int64_t, std::complex<float>>},
            {DT_BFLOAT16, MulDiffTypeCompute<std::complex<float>, Eigen::bfloat16, std::complex<float>>},
            {DT_FLOAT16, MulDiffTypeCompute<std::complex<float>, Eigen::half, std::complex<float>>},
            {DT_FLOAT, MulDiffTypeCompute<std::complex<float>, float, std::complex<float>>},
            {DT_DOUBLE, MulDiffTypeCompute<std::complex<float>, double, std::complex<double>>},
            {DT_UINT8, MulDiffTypeCompute<std::complex<float>, uint8_t, std::complex<float>>},
            {DT_COMPLEX128, MulDiffTypeCompute<std::complex<float>, std::complex<double>, std::complex<double>>}};
}

MulDiffTypeRow MakeComplex128MulDiffTypeRow()
{
    return {{DT_INT8, MulDiffTypeCompute<std::complex<double>, int8_t, std::complex<double>>},
            {DT_INT16, MulDiffTypeCompute<std::complex<double>, int16_t, std::complex<double>>},
            {DT_INT32, MulDiffTypeCompute<std::complex<double>, int32_t, std::complex<double>>},
            {DT_INT64, MulDiffTypeCompute<std::complex<double>, int64_t, std::complex<double>>},
            {DT_BFLOAT16, MulDiffTypeCompute<std::complex<double>, Eigen::bfloat16, std::complex<double>>},
            {DT_FLOAT16, MulDiffTypeCompute<std::complex<double>, Eigen::half, std::complex<double>>},
            {DT_FLOAT, MulDiffTypeCompute<std::complex<double>, float, std::complex<double>>},
            {DT_DOUBLE, MulDiffTypeCompute<std::complex<double>, double, std::complex<double>>},
            {DT_UINT8, MulDiffTypeCompute<std::complex<double>, uint8_t, std::complex<double>>},
            {DT_COMPLEX64, MulDiffTypeCompute<std::complex<double>, std::complex<float>, std::complex<double>>}};
}

const MulDiffTypeCalls& GetMulDiffTypeCalls()
{
    static const MulDiffTypeCalls kCalls = {
        {DT_UINT8, MakeUint8MulDiffTypeRow()},          {DT_INT8, MakeInt8MulDiffTypeRow()},
        {DT_INT16, MakeInt16MulDiffTypeRow()},          {DT_INT32, MakeInt32MulDiffTypeRow()},
        {DT_INT64, MakeInt64MulDiffTypeRow()},          {DT_BFLOAT16, MakeBfloat16MulDiffTypeRow()},
        {DT_FLOAT16, MakeFloat16MulDiffTypeRow()},      {DT_FLOAT, MakeFloatMulDiffTypeRow()},
        {DT_DOUBLE, MakeDoubleMulDiffTypeRow()},        {DT_COMPLEX64, MakeComplex64MulDiffTypeRow()},
        {DT_COMPLEX128, MakeComplex128MulDiffTypeRow()}};
    return kCalls;
}

uint32_t MulCpuKernel::Compute(CpuKernelContext& ctx)
{
    if (NormalCheck(ctx, kInputNum, kOutputNum) != KERNEL_STATUS_OK) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    if ((input0->GetDataSize() == 0) || (input1->GetDataSize() == 0)) {
        KERNEL_LOG_INFO("[%s] Input is empty tensor.", ctx.GetOpType().c_str());
        return KERNEL_STATUS_OK;
    }

    auto dtype_in1 = ctx.Input(kFirstInputIndex)->GetDataType();
    auto dtype_in2 = ctx.Input(kSecondInputIndex)->GetDataType();
    auto dtype_out = ctx.Output(kFirstOutputIndex)->GetDataType();
    KERNEL_LOG_DEBUG("Mul kernel get input1 dtype[%s], input2 dtype[%s], output dtype[%s].",
                     DTypeStr(dtype_in1).c_str(), DTypeStr(dtype_in2).c_str(), DTypeStr(dtype_out).c_str());
    if (dtype_in1 == dtype_in2) {
        return MulSameTypeCompute(ctx);
    }

    const auto& func_map = GetMulDiffTypeCalls().find(dtype_in1);
    if (func_map != GetMulDiffTypeCalls().end()) {
        const auto& funcs = func_map->second.find(dtype_in2);
        if (funcs != func_map->second.end()) {
            return DispatchMulDiffType(ctx, funcs->second);
        }
    }
    return KERNEL_STATUS_PARAM_INVALID;
}

OPS_MATH_REGISTER_CPU_KERNELV2(kMul, MulCpuKernel);
} // namespace aicpu
