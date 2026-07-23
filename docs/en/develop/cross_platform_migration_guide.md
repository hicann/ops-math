# Operator Cross-Platform Migration Guide

This guide introduces adaptation points and solutions for operator migration between multiple platforms. Taking operator migration from Atlas A2 series to Ascend 950 series as an example, compares hardware architecture differences and involved adaptation points, and provides relevant operator adaptation examples.

## 1. Hardware Architecture and Specification Parameter Comparison
<!--
### Atlas A2 Series Hardware Architecture

<div align="center">
  <img src="../figures/Atlas A2硬件架构.png" width="900" alt="Atlas A2 Hardware Architecture" />
</div>

### Ascend 950 Series Hardware Architecture

<div align="center">
  <img src="../figures/Ascend 950硬件架构.png" width="900" alt="Ascend 950 Hardware Architecture" />
</div>
-->
### Generation Specification Parameter Comparison

Usually divided into multiple product models according to different application scenarios, processes or hardware configurations. Each model may have certain differences in performance, resource configuration, etc. For ease of explanation and direct comparison, this section selects representative configurations as parameter display and difference analysis objects. Other related adjustments please refer to actual manual or official release.

<table>
  <tr>
    <th colspan="2" style="width: 25%;">Specification Item</th>
    <th style="width:37.5%;">Atlas A2</th>
    <th style="width:37.5%;">Ascend 950</th>
  </tr>
  <tr>
    <td rowspan="4">AICore</td>
    <td>Core Count</td>
    <td>24</td>
    <td>32</td>
  </tr>
  <tr>
    <td>Frequency</td>
    <td>1.8</td>
    <td>1.65</td>
  </tr>
  <tr>
    <td>Cube Compute Specification</td>
    <td>353T/376T @BF16,FP16</td>
    <td>426T@BF16,FP16 757T@FP8,HIFP8,MXFP8,INT8 1514T@MXFP4</td>
  </tr>
  <tr>
    <td>Vector Compute Specification (FP16)</td>
    <td>23.5T</td>
    <td>54T</td>
  </tr>
  <tr>
    <td rowspan="2">Memory</td>
    <td>Memory Capacity (GB)</td>
    <td>64</td>
    <td>128</td>
  </tr>
  <tr>
    <td>Memory Bandwidth</td>
    <td>1.6TB/s</td>
    <td>1.6TB/s</td>
  </tr>
</table>

## 2. Hardware Capability Changes Introduced Adaptation Points

<table>
  <tr>
    <th style="width: 25%;">Hardware Unit</th>
    <th style="width:35%;">Hardware Capability Change</th>
    <th style="width:40%;">Typical Impact Scope</th>
  </tr>
  <tr>
    <td rowspan="5">Transfer Unit</td>
    <td>Removed L1 to GM data path</td>
    <td>Kernels relying on L1 direct write-back to GM need to change to L1→UB→GM or L0C/FIXPIPE→GM path; related DataCopy links, event synchronization and buffer planning need adjustment</td>
  </tr>
  <tr>
    <td>Removed GM to L0A, L0B data path</td>
    <td>GM→L0A/L0B direct connection no longer available, need to complete through GM→L1→L0A/L0B; L1 partitioning strategy and MTE1/2 pipeline need restructuring</td>
  </tr>
  <tr>
    <td>ND DMA flexible data transfer, supports on-the-fly ND->NZ conversion</td>
    <td>Can use ND2NZ/DN2NZ to complete format conversion in MTE2 stage, reducing intermediate buffer and format conversion overhead; need to pay attention to stride, alignment and NZ shape mapping</td>
  </tr>
  <tr>
    <td>Supports Cube-&gt;Vector efficient internal data path: L1&lt;-&gt;UB、L0C-&gt;UB、FIXP-&gt;UB</td>
    <td>Can perform intermediate accumulation/activation/fusion on UB side (such as K-cut accumulation, post-processing), reducing GM round-trips; corresponding synchronization and pipeline partitioning need adjustment</td>
  </tr>
  <tr>
    <td>Introduced collective communication accelerator CCU1.0</td>
    <td>Communication-computation fusion operators adjust HcclServerType in Eager mode; use CCU series GE interfaces in Graph mode</td>
  </tr>
  <tr>
    <td rowspan="3">Compute Unit</td>
    <td>Vector added Regbase paradigm</td>
    <td>Original Membase-based access patterns, alignment methods, register count assumptions need re-examination; templates/tiling may need update to Regbase version</td>
  </tr>
  <tr>
    <td>Cube no longer supports int4_t</td>
    <td>All operators using int4_t need to switch to supported data types (such as int8), and update quantization solution logic</td>
  </tr>
  <tr>
    <td>Does not support 4:2 sparse matrix computation</td>
    <td>Kernels originally relying on 4:2 sparse feature for speedup need to change to dense or other supported sparse strategies, and update performance expectation documentation</td>
  </tr>
  <tr>
    <td rowspan="1">Storage Unit</td>
    <td>Local Buffer memory improvement: Cube L0C 256KB、Vector UB 256KB</td>
    <td>Larger L0C/UB allows increasing basic block and double buffer capacity, reducing K-cut/partitioning rounds; need to re-evaluate L1/L0/UB ratio and tile size</td>
  </tr>
  <tr>
    <td rowspan="2">Others</td>
    <td>Multi-core simultaneous access to Global Memory same address performance optimization</td>
    <td>Templates involving matrix multiplication related operators can be optimized</td>
  </tr>
  <tr>
    <td>SIMT</td>
    <td>After SIMT introduction, can use thread-level parallel processing for branches/irregular computation, but need to adapt thread partitioning, shared memory and synchronization semantics; some Vector implementations can be migrated to SIMT version</td>
  </tr>
</table>

## 3. Recommended Migration Steps

1. Confirm whether the compute units (Cube/Vector) involved in the operator and corresponding unit supported data types have differences between platforms.
2. Confirm whether the data transfer units involved (ND-&gt;NZ, GM&lt;-&gt;Lx, collective communication, etc.) have differences between platforms.
3. Modify item by item according to hardware capability change points (Vector architecture, Cube supported data types, L1/L0/UB size, CCU communication, etc.).
4. Refer to operator migration examples to adjust/complete Atlas A2/Ascend 950 branch logic.

## 4. Operator Migration Examples

### Cube Matrix Computation Class Operators

#### Global Memory Same Address Access Conflict Optimization

Ascend 950 hardware adds same address request parallel processing feature, no need to additionally avoid same address access conflicts in various core partitioning scenarios. During migration, can simplify the core partitioning strategy designed for "staggered conflict avoidance" on Atlas A2 to more regular sliding window templates (such as row group window + column direction round-trip scanning), reducing invalid offsets and redundant address transformations. In practice, suggest first retaining original tile size with functional equivalence as goal, then gradually relaxing core partitioning constraints, combining profiling data to observe key indicators such as MAC utilization, MTE2 utilization, L2 hit rate, confirming whether template adjustment brings stable benefits.
<!--
<div align="center">
  <img src="../figures/SWAT滑动窗口模板.png" width="900" alt="SWAT Sliding Window Template" />
</div>
-->
#### Tile Size Adjustment

L0C size on Atlas A2 is 128KB, Ascend 950 increases to 256KB, meaning single time can carry larger accumulation result blocks. During migration, can prioritize increasing Tile block partitioning granularity or increasing K direction single-round processing depth to reduce partitioning and K-cut rounds, lowering loop control and transfer overhead. At the same time need to re-balance L1/L0/UB capacity budget, avoid L0C enlargement squeezing A/B/scale buffer causing pipeline breakpoints.

### Vector Vector Computation Class Operators

#### SIMT

Ascend 950 series added SIMT unit. SIMT has significant advantages over SIMD in handling non-regular discrete access, suitable for scenarios with discontinuous addresses, large access span variation, inconsistent branch paths (such as scatter/gather, index rearrangement, sparse update, etc.).

During migration, suggest first identifying "access-dominated" and "low vectorization efficiency" operator sub-processes: if original SIMD implementation has large number of masked branches, high invalid lane ratio, or needs complex address assembly, can rewrite that part as SIMT path, usually can reduce control overhead and increase effective access throughput.

In practice, need to focus on the following: first, thread task partitioning needs to match data sparsity, avoid extremely unbalanced thread load; second, reduce pipeline stalls caused by high-frequency random access, try to complete index regularization and bucketing upstream; third, decouple boundary processing from main path, avoid introducing too many branches in hot loops. Suggest comparing "pure SIMD implementation" and "SIMD+SIMT hybrid implementation" after migration, selecting optimal strategy according to data distribution, rather than fixing single path.

**Taking gather_v2 operator as example: SIMD vs SIMT implementation comparison**

gather_v2 operator performs gather based on units of merged tail axis, therefore template selection basis is: tail axis ≤ 2048 uses SIMT template, tail axis > 2048 uses SIMD template. Because when tail axis is small, need to discretely access multiple discontinuous small block addresses, simt efficiency is high. Below compares core differences of two implementations:

**1. Programming Model Difference**

SIMD implementation uses traditional vectorized programming model, needs explicit UB buffer and pipeline queue management:

```cpp
// SIMD: Use queue mechanism to manage data buffer
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueue_;
TBuf<QuePosition::VECCALC> indexBuf_;

// SIMD: Process row by row, explicit transfer and synchronization
for (int64_t j = 0; j < rows; j++) {
    INDICES_T index = GetIndex(yIdx, indiceEndIdx);  // Scalar read index
    int64_t xIndex = index * tilingData_->innerSize;
    DataCopyPad(xLocal[j * colsAlign], xGm[offset], dataCoptExtParams, dataCopyPadExtParams); // Batch continuous data transfer in
}
inQueue_.EnQue<int8_t>(xLocal);  // Enqueue waiting for output
```

SIMT uses thread-level parallel model, each thread independently processes elements:

```cpp
// SIMT: Use thread-level parallel, no explicit buffer management needed
__simt_vf__ LAUNCH_BOUND(2048) void GatherSimt(...) {
    for (INDEX_SIZE_T index = Simt::GetThreadIdx();
         index < currentCoreElements;
         index += Simt::GetThreadNum()) {  // Thread jumping parallel
        // Each thread independently calculates single-point index and accesses
        INDEX_SIZE_T gatherI = Simt::UintDiv(yIndex, m0, shift0);
        INDICES_T indicesValue = indices[gatherI];  // Directly access GM based on single-point index gatherI
        y[yIndex] = idxOutOfBound ? 0 : x[xIndex];  // Directly write back to GM
    }
}
```

**2. Access Pattern Difference**

| Feature | SIMD Implementation | SIMT Implementation |
|------|----------|----------|
| Data Access | Explicit transfer to UB through DataCopyPad | Thread directly accesses GM through `__gm__` pointer |
| Buffer Management | Need AllocTensor/EnQue/DeQue/FreeTensor | No explicit buffer needed, hardware automatically manages |
| Synchronization Mechanism | Explicit event synchronization (Hard_Event::MTE2_V, etc.) | Implicit synchronization between threads |

**3. Applicable Scenario Difference**

SIMD suitable for scenarios accessing large continuous address blocks, efficiently processing continuous data through vectorized instructions;

SIMT suitable for discrete access, thread parallel processing;

#### Regbase

Ascend 950 series introduced Regbase programming paradigm. Compared to traditional Membase (Vector API) programming, Regbase is closer to underlying hardware register operations, providing finer vectorization control capability.

**Features**

- Uses underlying APIs under `AscendC::MicroAPI` namespace
- Directly operates registers `RegTensor<T>` rather than explicitly managing UB buffer queues
- Implements flexible element-level mask control through `MaskReg`

**Comparison with Membase Programming Model**

| Feature | Membase (Traditional Vector API) | Regbase (MicroAPI) |
|------|---------------------------|---------------------|
| Data Carrier | `LocalTensor<T>` + Queue mechanism | `RegTensor<T>` register |
| Memory Management | Explicit Alloc/EnQue/DeQue/Free | Register automatic allocation |
| Mask Control | Function parameter control | `MaskReg` register control |
| Data Transfer | `DataCopy`/`DataCopyPad` | `MicroAPI::DataCopy` + distribution mode |

**Code Example**

```cpp
__simd_vf__ __aicore__ void GenIndexBuf(ubuf int32_t* helpAddr, int32_t colFactor)
{
    // Declare register tensors
    AscendC::MicroAPI::RegTensor<int32_t> v0;
    AscendC::MicroAPI::RegTensor<int32_t> v1;
    AscendC::MicroAPI::RegTensor<int32_t> vd1;

    // Create full mask
    AscendC::MicroAPI::MaskReg preg =
        AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();

    // Scalar copy to register
    AscendC::MicroAPI::Duplicate(v1, colFactor, preg);
    // Generate sequence [0, 1, 2, ...]
    AscendC::MicroAPI::Arange(v0, 0);
    // Vector operation
    AscendC::MicroAPI::Div(vd1, v0, v1, preg);
    AscendC::MicroAPI::Mul(vd2, vd1, v1, preg);
    AscendC::MicroAPI::Sub(vd3, v0, vd2, preg);
    // Register data write back to UB
    AscendC::MicroAPI::DataCopy(helpAddr, vd3, preg);
}
```

```cpp
// Dynamic mask: Processing incomplete tail data
__simd_vf__ __aicore__ void GatherProcess(ubuf int8_t* curYAddr, uint16_t repeatimes, uint16_t computeSize)
{
    MicroAPI::RegTensor<int8_t> vregTemp;
    MicroAPI::MaskReg preg;

    for (uint16_t r = 0; r < repeatTimes; r++) {
        // Update mask based on remaining element count
        preg = MicroAPI::UpdateMask<int8_t>(sreg);
        // Create address offset register
        MicroAPI::AddrReg offset = MicroAPI::CreateAddrReg<int8_t>(r, computeSize);
        MicroAPI::DataCopy(vregTemp, curXAddr, offset);
        // Masked data store
        MicroAPI::DataCopy(curYAddr, vregTemp, offset, preg);
    }
}
```

```cpp
// Data aggregation
__VEC_SCOPE__
{
    MicroAPI::RegTensor<uint32_t> indicesReg;
    MicroAPI::RegTensor<int32_t> vd0;

    for (uint16_t indices = 0; indices < indicesLoopNum; indices++) {
        // Load index (E2B distribution mode: broadcast scalar to vector)
        MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_E2B_B32>(indicesReg, indicesAddr);
        // Gather data aggregation based on index
        MicroAPI::DataCopyGather(vd0, curXAddr, indicesReg, preg);
        // Data block copy output
        MicroAPI::DataCopy<int32_t, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            curYAddr, vd0, blockStride, preg);
    }
}
```

**Key Regbase API Description**

| API Category | API Name | Function Description |
|---------|---------|----------|
| Register Type | `RegTensor<T>` | Vector register tensor type |
| Mask Type | `MaskReg` | Mask register type |
| Mask Creation | `CreateMask<T, Pattern>()` | Create mask (ALL/HALF etc. modes) |
| Mask Update | `UpdateMask<T>(count)` | Dynamically update mask based on remaining element count |
| Scalar Operation | `Duplicate(reg, val, mask)` | Copy scalar value to all register elements |
| Sequence Generation | `Arange(reg, start)` | Generate consecutive sequence |
| Arithmetic Operation | `Add/Sub/Mul/Div(dst, src1, src2, mask)` | Vector arithmetic operation |
| Scalar Operation | `Adds/Muls(dst, src, scalar, mask)` | Vector and scalar operation |
| Type Conversion | `Cast<DT, ST>(dst, src, mask)` | Data type conversion |
| Comparison Operation | `Compare<T, CMPMODE>(mask, src1, src2, pred)` | Vector comparison generates mask |
| Data Load | `DataCopy<T, LoadDist>(reg, addr)` | Load from UB to register |
| Data Store | `DataCopy<T>(addr, reg, mask)` | Store from register to UB |
| Gather | `DataCopyGather(dst, base, indices, mask)` | Gather data based on index |
| Address Offset | `CreateAddrReg<T>(loop, stride)` | Create loop address offset register |

**Distribution Mode (LoadDist) Description**

| Mode | Description | Typical Use |
|------|------|----------|
| `DIST_NORM` | Normal continuous load | Continuous data processing |
| `DIST_UNPACK_B16` | 16-bit unpack load | FP16/BF16 to FP32 |
| `DIST_BRC_B32/B16` | Broadcast load | Scalar scale broadcast |
| `DIST_E2B_B32` | Scalar to vector broadcast | Index value broadcast |

**Migration Suggestions**

1. Scenarios suitable for Regbase: Need fine control of register allocation, complex mask logic, Gather/Scatter access patterns
2. Scenarios retaining Membase: Simple continuous data transfer and computation, double buffer pipeline
3. Hybrid use: Can combine both paradigms in same operator, use Regbase for core computation logic, use Membase for data transfer management

### Cube-Vector Fusion Class Operators

#### MTE Data Transfer Path Changes

Ascend 950 new architecture introduced direct paths between UB2L1 & L0C2UB, achieving fast transfer of matrix computation data, aiming to simplify CV fusion operator development and improve performance.
<!--
<div align="center">
  <img src="../figures/Ascend950新增CV直连通路.png" width="700" alt="Ascend950 New CV Direct Path" />
</div>
-->

**Matrix Transfer In**

Enable UB to L1 (UB2L1) direct path, through DataCopy interface, supports fusion operator's vector computation result directly transferred to L1.

**Matrix Transfer Out**

Enable L0C to UB (L0C2UB) direct path, through DataCopy interface, supports fusion operator's matrix computation result directly transferred to UB for subsequent vector computation.

For K-cut or multi-stage fusion scenarios, can change "L0C transfer back to GM then read back to UB" to "L0C direct to UB accumulation/post-processing", reducing GM round-trip bandwidth pressure and latency. During migration, suggest completing intermediate result merging, activation/quantization pre-processing on UB side, and explicitly organizing MTE1/MTE2/MTE3 and compute unit event synchronization order, ensuring cross-unit pipeline continuity, avoiding data visibility or synchronization timing issues introduced by new paths. Key interface definitions can refer to:

```cpp
// 1.New: Transfer in interface adds UB2L1 Nd2Nz transfer in, supports Src&Dst both LocalTensor form
template <typename T>
__aicore__ inline void DataCopy(const LocalTensor<T>& dst, const LocalTensor<T>& src, const Nd2NzParams& intriParams)；

// 2.New: Transfer out interface adds L0C2UB transfer out, supports direct transfer from L0C to UB, supports Src&Dst both LocalTensor form
template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
__aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsC310<config.format>& intriParams);
template <CO2Layout format = CO2Layout::ROW_MAJOR>
struct FixpipeParamsC310 {
    // ...
    uint8_t dualDstCtl = 0;
};

// 3.Capability enhancement: Inter-core synchronization interface adds mode 3
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId)

```

#### Inter-Core Synchronization Semaphore Matching

`CrossCoreSetFlag` and `CrossCoreWaitFlag` are inter-core synchronization semaphore interfaces, widely used for data dependency and collaborative control between multiple cores. Essentially implements decoupling and orderly advancement of data processing stages between different AICores through "semaphore" method, commonly used for pipeline control, double buffer switching, cross-core collaboration, etc.

- `CrossCoreSetFlag`: Current core (or thread) actively sets specified flag signal after completing certain stage of data processing, informing dependent party (usually other cores or downstream pipeline stages) that this stage is complete and can continue executing subsequent flow.
- `CrossCoreWaitFlag`: Current core (or thread) needs to wait for certain flag signal to be set (i.e., dependent data or event completed), will only continue executing downward after detecting flag.

The essence of this semaphore set is to ensure consistent synchronization sequence between multiple threads/multiple pipeline stages, preventing data race or deadlock and other hardware anomalies due to resources not ready or dependencies not completed. For detailed explanation of CrossCoreSetFlag and CrossCoreWaitFlag inter-core synchronization interfaces, please refer to [Ascend C API](https://www.hiascend.com/document/redirect/CannCommunityAscendCApi).

On Ascend 950, `CrossCoreWaitFlag` and `CrossCoreSetFlag` counts must strictly match, and suggest designing in pairs in "produce first then consume" order within same synchronization semantic domain. On Atlas A2, if there are extra `CrossCoreSetFlag` semaphores between operators, HWTS will perform special processing to clear counter. Ascend 950 series, to reduce hardware overhead, no longer relies on such fallback mechanism, requires single operator inter-core synchronization semaphore one-to-one matching, otherwise will inevitably cause stuck.

During migration, please focus on checking the following issues: first, exception branch early return causing only `Set` executed without corresponding `Wait` (or vice versa); second, multi-stage pipeline reusing same `flagId` but lifecycle overlap, causing "cross-stage crosstalk"; third, condition-triggered synchronization in loop but loop boundary not aligned, causing iteration count inconsistency. Above issues may be masked on Atlas A2, will directly expose as blocking timeout or deadlock on Ascend 950. For operators with complex cross-core pipeline, can first build minimum dataset for single-stage verification, then gradually stack double buffer and multi-stage to reduce complexity of locating synchronization issues.

### Collective Communication Class Operators

Ascend 950 introduced collective communication accelerator CCU1.0, reducing access requirements and scheduling latency. To effectively utilize this feature, changed operator cross-chip communication method from A2's AICPU to CCU communication.

**Eager Mode**

In the second stage interface of aclnn two-stage interface, specify collective communication type for operator executor aclOpExecutor.

Taking [MatmulAllReduce](https://gitcode.com/cann/ops-transformer/tree/master/mc2/matmul_all_reduce) operator migration adaptation as example:
Set NnopbaseSetHcclServerType enumeration value, A2 is NNOPBASE_HCCL_SERVER_AICPU, 950 is NNOPBASE_HCCL_SERVER_TYPE_CCU.

```CPP
// ...
aclnnStatus aclnnMatmulAllReduce(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // ...
    if (NnopbaseSetHcclServerType) {
        if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
            NnopbaseSetHcclServerType(executor, NnopbaseHcclServerType::NNOPBASE_HCCL_SERVER_TYPE_CCU);
        }
    }
    // ...
    return ACLNN_SUCCESS;
}
```

**Graph Mode**

1. Used for resource calculation and allocation, in CalcParamFunc callback interface involving附属流 related information, distinguish collective communication type of附属流 for GE context.
2. Used for setting main stream/附属流 custom task and parameter customization, in GenerateTask callback interface, distinguish two sets of GE KernelLaunch interfaces, separately call AICPU communication or CCU communication creation and customization process.

Static graph GE side creates communication task type, A2 is aicpu kfc server + kfc_stream; 950 is ccu server + ccu_stream. Involved code file: [matmul_all_reduce_gen_task.cpp](https://gitcode.com/cann/ops-transformer/blob/master/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
ge::Status MatmulAllReduceCalcParamFunc(gert::ExeResGenerationContext *context)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "ccu server", "ccu_stream");
    }
    // A2
    return Mc2GenTaskOpsUtils::CommonKFCMc2CalcParamFunc(context, "aicpu kfc server", "kfc_stream");
}
// ...
```

Static graph GenTask invocation interface has differences, process has differences. Involved code file: [matmul_all_reduce_gen_task.cpp](https://gitcode.com/cann/ops-transformer/blob/master/mc2/matmul_all_reduce/op_graph/matmul_all_reduce_gen_task.cpp)

```CPP
// ...
// A2
ge::Status MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(
    const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>>& tasks) {
    // ...
    // aicpu task
    ge::KernelLaunchInfo aicpu_task =
        ge::KernelLaunchInfo::CreateAicpuKfcTask(context, SO_NAME.c_str(), KERNEL_NAME_V1.c_str());
    // ...
}

// 950
ge::Status Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks) {
    // ...
    // ccu task
    ge::KernelLaunchInfo ccuTask = ge::KernelLaunchInfo::CreateCcuTask(context, ccuGroups);
    // ...
}

ge::Status MatmulAllReduceGenTaskFunc(const gert::ExeResGenerationContext *context, std::vector<std::vector<uint8_t>> &tasks)
{
    if (Mc2GenTaskOpsUtils::IsTargetPlatformNpuArch(context->GetNodeName(), NPUARCH_A5)) {
        // 950
        return Mc2Arch35GenTaskOpsUtils::Mc2Arch35GenTaskCallBack(context, tasks);
    }
    // A2
    return MatmulAllReduceGenTaskOpsUtils::MatmulAllReduceGenTaskCallback(context, tasks);
}
// ...
```

## 5. Common Issues and Performance Tuning Suggestions (FAQ/Performance Tips)

If operator performance on Ascend 950 decreases instead of increases, can prioritize checking:

1. Whether still using Atlas A2's staggered core partitioning template
2. Whether CCU communication is not enabled and still using AICPU
3. Whether tiling still uses Atlas A2's L1/L0/UB partitioning strategy, causing Ascend 950's larger on-chip cache not fully utilized
