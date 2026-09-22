# ReduceMeanWithCastFusionPass

## 融合模式

该融合将图中的 `ReduceMeanWithCast` 算子拆解为 `Cast` + `ReduceMean`。当 `ReduceMeanWithCast` 携带有效的 `dtype` 属性（不为 `DT_UNDEFINED`）时，先插入 `Cast` 节点将输入转换为目标数据类型，再连接 `ReduceMean`；当 `dtype` 为 `DT_UNDEFINED` 时，不插入 `Cast`，直接替换为 `ReduceMean`。

**场景一：dtype 有效（插入 Cast）**

![](../../../docs/zh/figures/ReduceMeanWithCastFusionPass_1.png)

**场景二：dtype 为 DT_UNDEFINED（不插入 Cast）**

![](../../../docs/zh/figures/ReduceMeanWithCastFusionPass_2.png)

替换后 `ReduceMean` 的 `keep_dims`、`noop_with_empty_axes` 属性继承自原 `ReduceMeanWithCast` 节点；`Cast` 的 `dst_type` 取自原节点的 `dtype` 属性。

## 使用约束

- 版本约束：GE 编译器版本不低于 9.1.0 时融合生效。
- 属性约束：
  - `keep_dims`：布尔型，默认 `false`，原样继承到 `ReduceMean`。
  - `noop_with_empty_axes`：布尔型，默认 `true`，原样继承到 `ReduceMean`。
  - `dtype`：目标数据类型，默认 `DT_UNDEFINED`；为 `DT_UNDEFINED` 时不插入 `Cast`，否则插入 `Cast` 并以其作为 `dst_type`。
- 不校验输入 shape 与数据类型，相关约束遵循底层 `Cast`、`ReduceMean` 算子约束。

## 支持的型号

该融合规则不区分平台，具体支持的型号以底层 `Cast`、`ReduceMean` 算子的产品支持情况为准。
