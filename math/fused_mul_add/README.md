# FusedMulAdd

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                              |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- **算子功能**：三元 element-wise 融合算子，把 `Mul → Add` 子图融合为单次
  kernel 启动，减少一次 GM 中间数据搬运。
- **计算公式**：

$$y = x_1 \cdot x_2 + x_3$$

  三个输入按 NumPy 广播规则两两对齐后逐元素计算。

## 参数说明

| 参数名 | 输入/输出 | 描述                                                 | 数据类型                  | 数据格式 |
| :----: | :-------: | :--------------------------------------------------- | :-----------------------: | :------: |
| x1     | 输入      | 乘法第一个输入张量。                                 | FLOAT16, FLOAT, INT32     | ND       |
| x2     | 输入      | 乘法第二个输入张量；与 `x1` 须可广播。               | FLOAT16, FLOAT, INT32     | ND       |
| x3     | 输入      | 加法侧输入张量；与 `x1 * x2` 结果须可广播。          | FLOAT16, FLOAT, INT32     | ND       |
| y      | 输出      | `x1 * x2 + x3` 的结果；shape 为三者广播后的统一形状。 | FLOAT16, FLOAT, INT32     | ND       |

## 约束说明

- `x1`、`x2`、`x3`、`y` 必须为**同一种 dtype**；不支持 mix-dtype。
- 支持任意 NumPy 广播形态（含标量 `[1]`、单维 broadcast、跨 rank broadcast）。
- 支持动态 shape 与动态 rank。

## 实现方案

| 层 | 文件 | 说明 |
| --- | --- | --- |
| 计算图原型 | `op_graph/fused_mul_add_proto.h` | `REG_OP(FusedMulAdd)`，三输入一输出 |
| 算子定义 | `op_host/fused_mul_add_def.cpp` | `OpDef::AddConfig("ascend950", ...)` |
| InferShape | `op_host/fused_mul_add_infershape.cpp` | 复用 `Ops::Base::InferShape4Broadcast(ctx, 3)` |
| Tiling | `op_host/arch35/fused_mul_add_tiling_arch35.{h,cpp}` | 按 dtype 分支调用 `Ops::Base::BroadcastBaseTiling<OpDag>` |
| DAG | `op_kernel/arch35/fused_mul_add_dag.h` | fp32/fp16 通路在 fp32 中间精度下用 `Vec::Mul + Vec::Add`；int32 通路用 `Vec::Mul + Vec::Add` |
| Struct | `op_kernel/arch35/fused_mul_add_struct.h` | `BRC_TEMP_SCH_MODE_KEY_DECL/SEL` |
| Kernel 入口 | `op_kernel/fused_mul_add_apt.cpp` | `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` + `BroadcastSch<schMode, OpDag>` |

### fp16 / fp32 通路

```
In0/In1/In2 -- CopyInBrc -- Cast(->fp32) -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- Cast(->T,RINT) -- CopyOut -- Out0
```

全部输入先 Cast 到 fp32，再用 `Vec::Mul + Vec::Add` 两段式按公式
`y = x1 * x2 + x3` 顺序计算，最后 Cast 回 T 写出。

> 此处不使用 `Vec::FusedMulAdd`。该 API 底层实现会 in-place 写回 src2 buffer，
> 在 broadcast 大 tensor 跨 tile 复用输入 UB buffer 时会污染下一 tile 的输入，
> 导致 fp32/fp16 huge broadcast 场景精度错误。使用普通 `Vec::Mul + Vec::Add`
> 保证所有 placeholder 派生的 cast 结果只读。

### int32 通路

```
In0/In1/In2 -- CopyInBrc -- Vec::Mul(x1,x2) -- Vec::Add(+x3) -- CopyOut -- Out0
```

## 调用说明

| 调用方式   | 样例代码                                                     | 说明                                                         |
| ---------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 图模式 | [test_geir_fused_mul_add](examples/arch35/test_geir_fused_mul_add.cpp) | 通过[算子IR](op_graph/fused_mul_add_proto.h)构图方式调用 FusedMulAdd 算子。 |