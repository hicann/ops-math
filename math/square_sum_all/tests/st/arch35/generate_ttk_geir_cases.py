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
"""Generate the deterministic 152-case SquareSumAll TTK GEIR suite.

GEIR 与 kernel 跑同一份 Device Kernel，但入口链路不同：GEIR 先建 GE 图，依次触发
InferShape（op_host）、InferDataType（op_graph）、Tiling、binary 选择，再执行并比对。
kernel 模式的输出 shape/dtype 直接来自 CSV，不经过这两个推导函数——所以 GEIR 是
InferShape 与 InferDataType 唯一的端到端看护点。

执行时同时开启静态图与动态图（`ttk geir -d`）：动态图把输入输出 desc 全部置为 -1
（graph_builder.py 的 is_dynamic 分支），GE 只能依赖算子自身的推导函数。

基础回归预算150条，另加NCHW、NHWC两条格式路由用例。本算子有两个轴是**常量，
不是可覆盖维度**：

- **tiling key 恒为 0**：square_sum_all_tiling.cpp 里是无分支的 SetTilingKey(0)，
  当前支持的ND、NCHW、NHWC格式复用同一套Kernel。不存在多tiling key可覆盖。
- **类型组合恒为 float32**：op_def四个参数都只声明DT_FLOAT。正向格式覆盖为ND输入到ND输出，
  以及NCHW/NHWC输入到ND输出；私有格式由Host Tiling提前拒绝，不进入GEIR正向用例。

因此基础150条投给真正变化的轴，新增2条专门验证公有格式选路：

| 轴 | 取值来源 | 覆盖 |
| :-- | :-- | :-- |
| blockDim | usedCoreNum = clamp(N/4096, 1, 56) | 1~56 全覆盖，这是本算子唯一的 tiling 形态变化 |
| 每核 tile 循环 | ceil(coreElements / 4096) | 1~6 |
| 尾块（两级） | tile 内 %64、核内 %4096 | 对齐/非对齐/仅尾块 |
| rank | InferShape 逐维遍历 | 1~8 |
| shape 形态 | 因子分解后旋转 | 1 出现在首/中/尾 |
| 异常值 | 16 个 profile | NaN、±Inf、零、有符号零、次正规、有限值平方上溢、两路隔离 |
| 分布 | uniform / normal | 各 75 条 |
"""

import argparse
import ast
import math
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from generate_ttk_cases import (
    CSV_HEADERS,
    FP32_ATOL,
    FP32_RTOL,
    MAX_CORE_COUNT,
    MAX_DIMENSION,
    REQUIRED_MISMATCH_RATIO,
    SPECIAL_PROFILE_NAMES,
    TILE_ELEMENTS,
    CaseSeed,
    _elements,
    _input_ranges,
    _make_targeted_row,
    _render,
    _tile_loops,
    _used_cores,
    _validated_output_path,
    _write_output,
)

BASE_CASE_COUNT = 150
TARGETED_CASE_COUNT = 2
CASE_COUNT = BASE_CASE_COUNT + TARGETED_CASE_COUNT
MAX_RANK = 8
PROFILE_COUNT = 12
MAX_TILE_LOOPS = 6
# 最大规模由"最长核跑满 6 个 tile"决定；两路输入合计约 11MB，实体卡可承受。
MAX_ELEMENTS = MAX_CORE_COUNT * TILE_ELEMENTS * MAX_TILE_LOOPS + 1


def _factor_shape(rank: int, element_count: int, phase: int) -> Tuple[int, ...]:
    """Split element_count into exactly `rank` factors, then rotate them.

    质数或小元素数会自然退化成"若干个 1 加一个大维"，旋转让那个大维出现在首、
    中、尾各个位置——InferShape 逐维遍历，1 出现的位置是它的敏感点。
    """
    factors = [1] * rank
    remaining = element_count
    index = 0
    divisor = 2
    while index < rank - 1 and remaining > 1 and divisor * divisor <= remaining:
        if remaining % divisor == 0:
            factors[index] = divisor
            remaining //= divisor
            index += 1
        else:
            divisor += 1
    factors[rank - 1] = remaining
    rotation = phase % rank
    rotated = factors[rotation:] + factors[:rotation]
    assert math.prod(rotated) == element_count
    return tuple(rotated)


# 向量寄存器（64 元素）与 32B DMA 边界，外加 tile 边界 4096±1。
BOUNDARY_ELEMENTS = (
    1,
    2,
    7,
    8,
    15,
    16,
    31,
    32,
    33,
    63,
    64,
    65,
    66,
    95,
    96,
    127,
    128,
    129,
    191,
    192,
    255,
    256,
    257,
    4095,
    4096,
    4097,
)

# 每个 rank 各挑 5 个元素数，压 InferShape 的逐维遍历。
RANK_FORM_ELEMENTS = (63, 257, 3584, 65536, 262144)

FILLER_ELEMENTS = (
    511,
    512,
    1023,
    1024,
    2047,
    2048,
    8191,
    8192,
    12288,
    24576,
    49152,
    98304,
    131072,
    524288,
    MAX_DIMENSION,
)


def _build_geir_seeds() -> List[CaseSeed]:
    seeds: List[CaseSeed] = []

    # A) 每个 Host tiling 核数各一条（56 条）。blockDim 是本算子唯一的 tiling 形态
    #    变化，必须逐个覆盖；rank 同步轮转，顺带把 InferShape 的 rank 维度铺开。
    for core_count in range(1, MAX_CORE_COUNT + 1):
        if core_count == 1:
            element_count = 127
        else:
            element_count = (
                TILE_ELEMENTS * core_count + (core_count * 29) % TILE_ELEMENTS
            )
        rank = (core_count - 1) % MAX_RANK + 1
        seeds.append(
            CaseSeed(
                category=f"core_{core_count:02d}",
                shape=_factor_shape(rank, element_count, core_count),
                profile_index=(core_count - 1) % PROFILE_COUNT,
            )
        )

    # B) 向量与 DMA 边界（26 条），rank 轮转。
    for index, element_count in enumerate(BOUNDARY_ELEMENTS):
        rank = index % MAX_RANK + 1
        seeds.append(
            CaseSeed(
                category=f"vector_n{element_count}",
                shape=_factor_shape(rank, element_count, index),
                profile_index=(index + 5) % PROFILE_COUNT,
            )
        )

    # C) 56 核满载时，最长核的 tile 循环数恰好落在 1~6 的下方/等于/上方（18 条）。
    for tile_count in range(1, MAX_TILE_LOOPS + 1):
        boundary = MAX_CORE_COUNT * TILE_ELEMENTS * tile_count
        for offset in (-1, 0, 1):
            element_count = boundary + offset
            rank = (tile_count * 3 + offset) % MAX_RANK + 1
            seeds.append(
                CaseSeed(
                    category=f"tile_{tile_count}_{offset + 1}",
                    shape=_factor_shape(rank, element_count, tile_count + offset),
                    profile_index=(tile_count * 2 + offset) % PROFILE_COUNT,
                )
            )

    # D) 逐 rank 的 shape 形态扫描（8 rank × 5 = 40 条）。
    for rank in range(1, MAX_RANK + 1):
        for index, element_count in enumerate(RANK_FORM_ELEMENTS):
            seeds.append(
                CaseSeed(
                    category=f"rank{rank}_n{element_count}",
                    shape=_factor_shape(rank, element_count, rank + index),
                    profile_index=(rank * 5 + index) % PROFILE_COUNT,
                )
            )

    seeds = _dedupe_by_shape(seeds)
    seeds = _top_up(seeds)
    assert len(seeds) == BASE_CASE_COUNT, len(seeds)

    # 把 16 个异常值 profile 均匀铺到元素数够大的种子上，避免全挤在小 shape。
    eligible_indexes = [
        index for index, seed in enumerate(seeds) if _elements(seed.shape) >= 65
    ]
    assert len(eligible_indexes) >= len(SPECIAL_PROFILE_NAMES)
    special_indexes = [
        eligible_indexes[
            index * (len(eligible_indexes) - 1) // (len(SPECIAL_PROFILE_NAMES) - 1)
        ]
        for index in range(len(SPECIAL_PROFILE_NAMES))
    ]
    assert len(set(special_indexes)) == len(SPECIAL_PROFILE_NAMES)
    for seed_index, special_profile in zip(special_indexes, SPECIAL_PROFILE_NAMES):
        seeds[seed_index] = replace(seeds[seed_index], special_profile=special_profile)
    return seeds


def _dedupe_by_shape(seeds: Sequence[CaseSeed]) -> List[CaseSeed]:
    seen = set()
    unique = []
    for seed in seeds:
        if seed.shape in seen:
            continue
        seen.add(seed.shape)
        unique.append(seed)
    return unique


def _top_up(seeds: List[CaseSeed]) -> List[CaseSeed]:
    """补齐到 BASE_CASE_COUNT，shape 不与既有重复。"""
    if len(seeds) > BASE_CASE_COUNT:
        raise SystemExit(
            f"seed groups already exceed the budget: {len(seeds)} > {BASE_CASE_COUNT}"
        )
    used = {seed.shape for seed in seeds}
    filler_index = 0
    for element_count in FILLER_ELEMENTS:
        for rank in range(1, MAX_RANK + 1):
            for phase in range(rank):
                if len(seeds) == BASE_CASE_COUNT:
                    return seeds
                shape = _factor_shape(rank, element_count, phase)
                if shape in used:
                    continue
                used.add(shape)
                seeds.append(
                    CaseSeed(
                        category=f"mixed_n{element_count}",
                        shape=shape,
                        profile_index=filler_index % PROFILE_COUNT,
                    )
                )
                filler_index += 1
    assert len(seeds) == BASE_CASE_COUNT, f"filler pool exhausted at {len(seeds)}"
    return seeds


def _case_name(
    distribution: str, case_index: int, seed: CaseSeed, profile_name: str
) -> str:
    distribution_code = "u" if distribution == "uniform" else "n"
    raw_name = (
        f"ssag_{distribution_code}_{case_index:03d}_r{len(seed.shape)}_"
        f"{seed.category}_{profile_name}"
    )
    return re.sub(r"[^a-zA-Z0-9_]", "_", raw_name)


def _build_rows() -> List[Dict[str, str]]:
    seeds = _build_geir_seeds()
    rows = []
    # 150 条预算下不再"同一 shape 跑两种分布"，而是让 150 个 shape 各自带一种分布：
    # 同样的条数换来一倍的结构覆盖，两种分布仍各占 75 条。
    for case_index, seed in enumerate(seeds, start=1):
        distribution = "uniform" if case_index % 2 == 1 else "normal"
        priority = "1" if distribution == "uniform" else "2"
        element_count = _elements(seed.shape)
        ranges, profile_name, profile_key = _input_ranges(seed, distribution)
        shape_pair = (seed.shape, seed.shape)
        row = {header: "" for header in CSV_HEADERS}
        row.update(
            {
                "testcase_name": _case_name(
                    distribution, case_index, seed, profile_name
                ),
                "network_name": "square_sum_all_geir_regression",
                "op_name": "square_sum_all",
                "input_shapes": repr(shape_pair),
                "input_dtypes": repr(("float32", "float32")),
                "input_formats": repr(("ND", "ND")),
                "output_shapes": repr(((1,), (1,))),
                "output_dtypes": repr(("float32", "float32")),
                "output_formats": repr(("ND", "ND")),
                "attributes": "{}",
                "input_data_ranges": ranges,
                # TTK 每对为 (rtol, 允许失配比例)；TestSpec 另行强制 atol=2^-16
                # 与 max_abs_error<=1e-2，两路标量输出各判一次。
                "precision_tolerances": repr(
                    (
                        (FP32_RTOL, REQUIRED_MISMATCH_RATIO),
                        (FP32_RTOL, REQUIRED_MISMATCH_RATIO),
                    )
                ),
                "absolute_precision": repr(FP32_ATOL),
                "output_inplace_indexes": "()",
                "output_shape_unknown_indexes": "()",
                "is_enabled": "True",
                "remark": (
                    f"mode=geir; distribution={distribution}; category={seed.category}; "
                    f"profile={profile_key}; rank={len(seed.shape)}; elements={element_count}; "
                    f"expected_cores={_used_cores(element_count)}; "
                    f"max_core_tile_loops={_tile_loops(element_count)}; tiling_key=0"
                ),
                "priority": priority,
                "manual_input_binaries": "()",
                "manual_golden_binaries": "()",
            }
        )
        rows.append(row)
    rows.extend(
        [
            _make_targeted_row(
                "ssag_x_151_fmt_nchw",
                "square_sum_all_geir_regression",
                ((2, 3, 4, 5), (2, 3, 4, 5)),
                ("NCHW", "NCHW"),
                ("ND", "ND"),
                "mode=geir; category=format; profile=NCHW-to-ND; rank=4; elements=120; tiling_key=0",
            ),
            _make_targeted_row(
                "ssag_x_152_fmt_nhwc",
                "square_sum_all_geir_regression",
                ((2, 4, 5, 3), (2, 4, 5, 3)),
                ("NHWC", "NHWC"),
                ("ND", "ND"),
                "mode=geir; category=format; profile=NHWC-to-ND; rank=4; elements=120; tiling_key=0",
            ),
        ]
    )
    _audit(rows)
    return rows


def _audit(rows: Sequence[Dict[str, str]]) -> None:
    assert len(rows) == CASE_COUNT
    assert all(row["is_enabled"] == "True" for row in rows)
    assert all(row["op_name"] == "square_sum_all" for row in rows)
    names = [row["testcase_name"] for row in rows]
    assert len(set(names)) == CASE_COUNT

    base_rows = rows[:BASE_CASE_COUNT]
    targeted_rows = rows[BASE_CASE_COUNT:]
    assert len(targeted_rows) == TARGETED_CASE_COUNT
    shapes = [ast.literal_eval(row["input_shapes"])[0] for row in base_rows]
    # 两路输入必须逐维同 shape，否则 Tiling 直接判非法。
    assert all(
        ast.literal_eval(row["input_shapes"])[0]
        == ast.literal_eval(row["input_shapes"])[1]
        for row in base_rows
    )
    # shape 全部互不相同：150 条预算下每一条都换来一个新的结构。
    assert len(set(shapes)) == BASE_CASE_COUNT

    assert all(1 <= len(shape) <= MAX_RANK for shape in shapes)
    # 算子对单维只要求大于 0（Tiling 只拒绝 <=0 和 int64 溢出）；真正需要设上限的
    # 是总元素数，它决定两路输入的显存占用。
    assert all(dimension >= 1 for shape in shapes for dimension in shape)
    assert all(_elements(shape) <= MAX_ELEMENTS for shape in shapes)

    rank_counts = Counter(len(shape) for shape in shapes)
    assert set(rank_counts) == set(range(1, MAX_RANK + 1)), rank_counts
    assert min(rank_counts.values()) >= 10, rank_counts

    element_counts = [_elements(shape) for shape in shapes]
    # blockDim 1~56 必须全覆盖——这是本算子唯一的 tiling 形态变化。
    assert set(_used_cores(count) for count in element_counts) == set(
        range(1, MAX_CORE_COUNT + 1)
    )
    assert set(range(1, MAX_TILE_LOOPS + 1)).issubset(
        set(_tile_loops(count) for count in element_counts)
    )
    required_boundaries = {
        1,
        63,
        64,
        65,
        127,
        128,
        129,
        4095,
        4096,
        4097,
        MAX_CORE_COUNT * TILE_ELEMENTS - 1,
        MAX_CORE_COUNT * TILE_ELEMENTS,
        MAX_CORE_COUNT * TILE_ELEMENTS + 1,
    }
    assert required_boundaries.issubset(set(element_counts)), required_boundaries - set(
        element_counts
    )

    # 尾块两级都要有样本：tile 内 %64 非零、核内 %4096 非零。
    assert any(count % 64 != 0 for count in element_counts)
    assert any(count % TILE_ELEMENTS != 0 for count in element_counts)
    assert any(count % 64 == 0 for count in element_counts)

    # 1 必须在首、中、尾三种位置都出现过（InferShape 逐维遍历的敏感点）。
    multi_rank = [shape for shape in shapes if len(shape) >= 3]
    assert any(shape[0] == 1 for shape in multi_rank)
    assert any(shape[-1] == 1 for shape in multi_rank)
    assert any(any(dim == 1 for dim in shape[1:-1]) for shape in multi_rank)

    priority_counts = Counter(row["priority"] for row in base_rows)
    assert priority_counts == {"1": BASE_CASE_COUNT // 2, "2": BASE_CASE_COUNT // 2}, (
        priority_counts
    )
    for special_profile in SPECIAL_PROFILE_NAMES:
        assert sum(f":{special_profile};" in row["remark"] for row in base_rows) == 1

    private_formats = {"FRACTAL_Z", "C1HWNCoC0", "NC1HWC0"}
    for row in rows:
        formats = set(ast.literal_eval(row["input_formats"]))
        formats.update(ast.literal_eval(row["output_formats"]))
        assert formats.isdisjoint(private_formats)
    for row, format_name in zip(targeted_rows, ("NCHW", "NHWC")):
        input_shapes = ast.literal_eval(row["input_shapes"])
        assert all(len(shape) == 4 for shape in input_shapes)
        assert ast.literal_eval(row["input_formats"]) == (format_name, format_name)
        assert ast.literal_eval(row["output_formats"]) == ("ND", "ND")


def _print_summary(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    base_rows = rows[:BASE_CASE_COUNT]
    shapes = [ast.literal_eval(row["input_shapes"])[0] for row in base_rows]
    element_counts = [_elements(shape) for shape in shapes]
    print(f"output={str(output_path)!r}")
    print(
        f"cases={len(rows)} uniform={BASE_CASE_COUNT // 2} normal={BASE_CASE_COUNT // 2} "
        f"targeted={TARGETED_CASE_COUNT} base_unique_shapes={len(set(shapes))}"
    )
    print(
        "rank_counts="
        + repr(dict(sorted(Counter(len(shape) for shape in shapes).items())))
    )
    print(
        f"blockdim_covered=1-{MAX_CORE_COUNT} ({len(set(_used_cores(n) for n in element_counts))} values)"
    )
    print(
        f"tile_loops={sorted(set(_tile_loops(n) for n in element_counts))} tiling_keys=[0]"
    )
    print(
        f"element_range={min(element_counts)}-{max(element_counts)} special_cases={len(SPECIAL_PROFILE_NAMES)}"
    )


def main() -> None:
    default_output = Path(__file__).with_name("ttk_geir_square_sum_all.csv")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that the checked-in CSV is up to date",
    )
    args = parser.parse_args()
    output_path = _validated_output_path(args.output)

    rows = _build_rows()
    rendered = _render(rows)
    if args.check:
        if (
            not output_path.is_file()
            or output_path.read_text(encoding="utf-8") != rendered
        ):
            raise SystemExit(f"generated CSV is stale: {str(output_path)!r}")
        print("generated_csv_is_current=yes")
    else:
        _write_output(output_path, rendered)
    _print_summary(rows, output_path)


if __name__ == "__main__":
    main()
