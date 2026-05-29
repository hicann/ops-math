# Power 算子

## 1. 功能描述

逐元素计算：

```
y = exp(power * log(x * scale + shift))
```

支持的输入数据类型：`fp16` / `bf16` / `fp32`（fp16、bf16 在 kernel 内 cast 为 fp32 计算后回写）。

## 2. 算子属性

| 属性 | 类型 | 必选 | 默认值 | 含义 |
|---|---|---|---|---|
| `power` | float | 是 | -      | 幂指数 |
| `scale` | float | 否 | `1.0`  | 输入线性缩放因子 |
| `shift` | float | 否 | `0.0`  | 输入线性平移量 |

## 3. 实现要点

- 使用 **Elementwise 模板**（`ElewiseBaseTiling` + `ElementwiseSchWithScalar` + `DAGSch`）。
- 在 tiling 层完成所有 `scale / power / shift` 标量计算与分支决策，最终通过 `culType` 枚举值
  + dtype 编码出 `tilingKey`，在 kernel 中实例化对应的 DAG，避免 kernel 内运行时分支：

| culType | 计算 |
|---|---|
| `ALL_ZEROS` | `y = 0` |
| `BROADCAST_SCALAR` | `y = bcastVal` (host 端预计算 `pow(shift, power)`、异常值 `+inf`/`NaN`，或 `power=0` 时的 `1.0`) |
| `LINEAR` (`power=1`) | `y = x*scale + shift` （fused MulAdd 语义：`Duplicate(shift) + Axpy(scale)`） |
| `SQUARE` (`power=2`) | `base = x*scale + shift; y = base * base` |
| `CUBE` (`power=3`) | `base = x*scale + shift; y = base^3` |
| `GENERIC_POW_POS` | 通用，`power>0`，零底数映射为 0 |
| `GENERIC_POW_NEG` | 通用，`power<0`，零底数映射为 `+inf` |

通用分支在 kernel 内通过 `Compare + Select` 合并三种取值（正/负/零底数）：

```
absBase   = |base|
logAbs    = log(absBase)
rawExp    = exp(power * logAbs)
posVal    = rawExp                  // base > 0
negVal    = rawExp * negScalar      // negScalar = ±1 (整数 power) 或 NaN (非整数 power)
zeroVal   = 0 / +inf                // 由 tilingKey 区分
tmp       = base > 0 ? posVal : negVal
y         = base == 0 ? zeroVal : tmp
```

`isclose` 判等方法参考 `math/is_close`：`|a-b| <= atol + rtol*|b|`，`atol=1e-8`，`rtol=1e-5`。

## 4. 不支持范围

- 不支持广播；输入与输出 shape 一致。
- 输入 dtype 仅支持 `fp16/bf16/fp32`；其它类型在 tiling 层会直接报错。
- 本算子目前仅生成 ascend950 平台的二进制，不包含 aclnn 接口模块。

## 5. 目录结构

```
power/
├── CMakeLists.txt
├── README.md
├── docs/DESIGN.md
├── op_graph/power_proto.h
├── op_host/
│   ├── power_def.cpp
│   ├── power_infershape.cpp
│   ├── arch35/
│   │   ├── power_tiling_arch35.h
│   │   └── power_tiling_arch35.cpp
│   └── config/ascend950/{power_binary.json, power_simplified_key.ini}
└── op_kernel/
    ├── power_apt.cpp
    └── arch35/{power_struct.h, power_dag.h}
```
