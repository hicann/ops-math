# LogSpaceD 测试交付清单

仅交付正式 Golden、可重复执行的 ST 用例和 Host UT 源码。生成、审计脚本以及测试报告、日志、编译缓存不属于目标仓交付内容。

| 位置（ST 文件均在 st/arch35） | 用途 | 条数 |
| --- | --- | ---: |
| assets/golden.py | 唯一正式 TTK Golden；CPU Promote 允许 FP64 容器，不改变生产 dtype 契约 | — |
| ttk_kernel_log_space_d_st.csv | 基础正向用例 | 255 |
| ttk_kernel_log_space_d_coverage_st.csv | 数值、默认属性、载体、分核及大索引边界补充 | 330 |
| ttk_kernel_log_space_d_non_arange_st.csv | assist=[37] 的独立冒烟用例 | 1 |
| ttk_kernel_log_space_d_coverage_negative.csv | Kernel 非法参数、输入/输出描述符；scalar 另需 GEIR 验证 | 34 |
| exception_geir.csv | GEIR/Kernel 异常探针与合法控制组 | 21 |
| exception_attribute_transport.csv | GEIR Float 属性有限性与传输边界 | 7 |
| ttk_geir_log_space_d_infer_coverage.csv | 显式 [-1]/[-2] 建图描述，运行时输入仍为 [1] | 24 |
| ttk_geir_log_space_d_determinism.csv | 既有正向子集的重复执行检查，不计入新增正向数量 | 27 |
| ut/op_host/test_log_space_d_infershape.cpp | 正式推导规则 UT | — |
| ut/op_host/arch35/test_log_space_d_tiling_arch35.cpp | Tiling 契约与边界 UT | — |

三份正向 CSV 合计 586 个唯一用例，可用于 Kernel 或 GEIR；性能使用同一全量清单，不另交付重复的性能子集。输入固定为一维 [1]，输出为一维 [steps]。

正向测试使用原生 TTK：`--plugin <算子目录>/tests/assets/golden.py --compare cross_check --golden-mode Promote --pc 1 --xpu-perf -c true -d true`。Kernel 性能使用 `--task-prof true --run 100`；GEIR 确定性专项使用 `--deterministic-level 1 --run 10`。

异常测试使用 `--golden-mode Disable --task-prof false`，分别解释合法控制、目标代码拒绝和框架提前拒绝，不能用原生 PASS/FAIL 总数代替拦截结论。Kernel 会将 scalar 归一成 [1]；GEIR 裸 Infinity 传输也存在限制，对应限制不能当作算子通过证据。

交付源码只保留正式 V2 InferShape，不包含本地临时 V1 桥接。执行前需重新构建，并通过冒烟确认运行环境实际分派到目标 V2 和 Kernel；历史 V1 桥接测试结果不能替代 V2-only 通路验证。用例不绑定 V1 回调，但 GEIR 的可执行性仍取决于 CANN 环境的注册分派支持。

用例生成和契约审计脚本位于生成工程 `LogSpaceD_package/tests/case_generation`。所有执行结果通过 TTK 的 `-o` 指定到生成工程内的新目录，禁止覆盖已冻结的最终结果或写入目标仓。
