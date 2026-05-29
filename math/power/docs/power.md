# Power

[📄 查看源码](https://gitcode.com/cann/ops-math/tree/master/math/power)

> 本算子仅提供 **GE IR** 通路，**不提供 aclnn 接口**。在计算图中以 `Power` 算子节点的形式接入，由图编译器调度执行。

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>      |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                              |    √     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对输入张量 `x` 逐元素执行线性变换后再做幂运算。

- 计算公式：

$$
y_i = \big(\text{scale} \cdot x_i + \text{shift}\big)^{\text{power}}
$$

  其中 `power` / `scale` / `shift` 均为标量属性，在 host 端 tiling 阶段完成全部分支决策与可预计算的常数折叠，kernel 端按 `tilingKey` 路由到对应的 DAG，无运行时分支。

- 等价 PyTorch 表达：

  ```python
  base = scale * x + shift
  y = torch.pow(base, power)
  ```

- 实现要点：

  | 类别 | 内容 |
  |---|---|
  | 计算模板 | Elementwise（`ElewiseBaseTiling` + `ElementwiseSch` + `DAGSch`） |
  | 计算精度 | `fp16`/`bf16` 在 kernel 内 cast 到 `fp32` 计算后回写到原 dtype |
  | 分支前移 | `culType` × `dtype` × `schMode` 三段编码到 `tilingKey`，kernel 模板实例化时已选定 DAG |
  | 性能优化 | `power ∈ {1,2,3}` 走乘法展开；`power ∉ {0,1,2,3}` 走 `exp(power·log(|base|))` 通用路径 |

- 算子内部分支（host 端 `culTypeEnum`）：

  | culType | 触发条件 | 计算 |
  |---|---|---|
  | `ALL_ZEROS` | `scale·power == 0` 且 `shift==0` 且 `power>0` | `y = 0` |
  | `BROADCAST_SCALAR` | `power==0`，或 `scale==0` 且 `power≠0` | `y = bcastVal`（host 端预算 `pow(shift,power)`、`1.0`、`NaN`、`+inf` 之一） |
  | `LINEAR` | `scale·power≠0` 且 `power==1` | `y = x·scale + shift` |
  | `SQUARE` | `scale·power≠0` 且 `power==2` | `y = (x·scale + shift)^2` |
  | `CUBE` | `scale·power≠0` 且 `power==3` | `y = (x·scale + shift)^3` |
  | `GENERIC_POW_POS` | `scale·power≠0` 且 `power>0` 且 `power∉{1,2,3}` | 通用幂运算，`base==0` 时输出 `0` |
  | `GENERIC_POW_NEG` | `scale·power≠0` 且 `power<0` | 通用幂运算，`base==0` 时输出 `+inf` |

  其中 `power∈{0,1,2,3}` 的 `IsClose` 判等容差为 `atol=1e-8`、`rtol=1e-5`，与 `math/is_close` 对齐。

## 算子参数说明

  <table style="undefined;table-layout: fixed; width: 1280px"><colgroup>
  <col style="width: 140px">
  <col style="width: 100px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 140px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（Tensor）</td>
      <td>输入</td>
      <td>幂运算的底数前置量，公式中的 x。</td>
      <td>shape 与 y 完全一致；不支持广播。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>不限制（含 0 维标量，标量按 [1] 处理）</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y（Tensor）</td>
      <td>输出</td>
      <td>幂运算结果，公式中的 y。</td>
      <td>dtype 与 x 一致；shape 与 x 一致。</td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>与 x 保持一致</td>
      <td>×</td>
    </tr>
    <tr>
      <td>power（attr，float）</td>
      <td>属性</td>
      <td>幂指数，公式中的 power。</td>
      <td>可选，默认 <code>1.0</code>。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale（attr，float）</td>
      <td>属性</td>
      <td>输入线性缩放因子，公式中的 scale。</td>
      <td>可选，默认 <code>1.0</code>。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>shift（attr，float）</td>
      <td>属性</td>
      <td>输入线性平移量，公式中的 shift。</td>
      <td>可选，默认 <code>0.0</code>。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>标量</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 异常值约定

逐元素遵循下表（`base = scale·x + shift`，整数 power 指 `power == floor(power)` 且有限）：

  | 场景 | 输出值 | 说明 |
  |---|---|---|
  | `base > 0` | `exp(power · log(base))` | 通用主路径 |
  | `base < 0` 且 `power` 为整数 | `(-1)^power · exp(power · log(|base|))` | 由 host 预置 `negScalar = ±1`，kernel 端用 `Compare + Select` 合并 |
  | `base < 0` 且 `power` 非整数 | `NaN` | 实数域未定义，host 预置 `negScalar = NaN`，乘加传递得 NaN |
  | `base == 0` 且 `power > 0` | `0` | GENERIC_POW_POS / ALL_ZEROS |
  | `base == 0` 且 `power < 0` | `+inf` | GENERIC_POW_NEG / BROADCAST_SCALAR，IEEE 754 语义 |
  | `power == 0`（含 `0^0`） | `1` | 按约定，走 BROADCAST_SCALAR，host 预置 `bcastVal = 1.0` |

## 约束说明

- **数据类型**：输入 `x` 与输出 `y` 必须为 `FLOAT16` / `BFLOAT16` / `FLOAT` 三者之一，且二者完全相同；tiling 在 host 端会拒绝其它 dtype 并返回 `GRAPH_FAILED`。
- **形状一致**：不支持广播，`x` 与 `y` 的 shape 必须完全一致；0 维标量在 host 端被视为 `[1]`。
- **平台**：当前仅生成 `ascend950` 平台的 kernel 二进制；其它平台编译时不下发本算子。
- **接口形式**：本算子**不**提供 aclnn 单算子接口，仅作为图算子节点存在，需通过 GE IR 或图编译器接入；如需 host 直调（kernel-launch）场景请直接复用本仓的 kernel 源码而非 aclnn API。
- **属性默认值**：缺省时 `power=1.0`、`scale=1.0`、`shift=0.0`，等价于 identity 映射（`y = x`），此时走 `LINEAR` 路径。
- **确定性计算**：本算子默认确定性实现，相同输入与属性下每次执行输出一致。

## 调用方式

由于不提供 aclnn 接口，无法通过 `aclnnPowerGetWorkspaceSize / aclnnPower` 两段式 API 调用。推荐接入方式：

1. **图算子方式**：在通过 GE IR / ATC 构图时，将 `Power` 作为节点加入计算图，由图编译器自动完成 tiling、kernel 选择与下发。属性 `power` / `scale` / `shift` 通过节点 attr 注入。
2. **Kernel 直调方式**：参考 `op_kernel/power_apt.cpp` 的 `power(...)` 入口与 `op_host/arch35/power_tiling_arch35.cpp` 的 `Tiling4Power(...)` 回调，自行构造 `PowerTilingData` 并通过 `<<<>>>` 直接发射 kernel。tilingKey 由 `GET_TPL_TILING_KEY(schMode, culType, dType)` 编码，三段含义见 `op_kernel/arch35/power_struct.h`。

## 相关文档

- [Power README](../README.md) — 文件清单、目录结构与对外承诺。
