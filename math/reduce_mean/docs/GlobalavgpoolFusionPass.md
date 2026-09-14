# Globalavgpoolpass

## 融合模式

该融合将图中的 `GlobalAveragePool` 算子替换为 `ReduceMean` 算子，沿空间维度（从第 2 维起的所有维度）求均值。如下图所示。

![](../../../docs/zh/figures/GlobalavgpoolFusionPass.png)

替换后 `ReduceMean` 的属性设置为 `keep_dims=true`、`noop_with_empty_axes=true`。

## 使用约束

- 版本约束：GE编译器版本不低于 9.1.0 时融合生效。
- 输入维度约束：`GlobalAveragePool` 输入 `x` 的维度必须为 3D、4D 或 5D，其余维度不融合。
- 数据类型与数据格式遵循底层 `GlobalAveragePool`/`ReduceMean` 算子约束。

## 支持的型号

该融合规则不区分平台，具体支持的型号以底层 `GlobalAveragePool`、`ReduceMean` 算子的产品支持情况为准。
