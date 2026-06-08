# SpaceToBatch

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    √     |
| <term>Atlas 推理系列产品</term>                               |    √     |
| <term>Atlas 训练系列产品</term>                               |    √     |

## 功能说明

- 算子功能：将空间维度的数据按块重新排列到批次维度，并对空间维度补零。

- 功能描述：
  首先根据 paddings 参数对输入的空间维度进行 zero-padding，然后将 padded 空间划分为 `block_size × block_size` 的块，每个块搬移到批次维度。输出批次维度变为输入的 `block_size × block_size` 倍，空间维度相应缩小。

- 计算公式：
  设输入 x 为 4D NHWC 张量 `[N, H_in, W_in, C]`，block_size = bs，paddings = `[[pad_top, pad_bottom], [pad_left, pad_right]]`。

  H_padded = H_in + pad_top + pad_bottom，W_padded = W_in + pad_left + pad_right，

  H_out = H_padded / bs，W_out = W_padded / bs。

  要求 H_padded 和 W_padded 均能被 bs 整除。

  输出 y 形状为 `[N * bs * bs, H_out, W_out, C]`。

  坐标映射（输出 → 输入）：
  ```
  n     = n_out / (bs * bs)
  bh    = (n_out % (bs * bs)) / bs
  bw    = n_out % bs
  h_in  = h_out * bs + bh - pad_top
  w_in  = w_out * bs + bw - pad_left

  若 0 ≤ h_in < H_in 且 0 ≤ w_in < W_in：
    y[n_out, h_out, w_out, c] = x[n, h_in, w_in, c]
  否则：
    y[n_out, h_out, w_out, c] = 0
  ```

## 参数说明

<table style="undefined;table-layout: fixed; width: 1480px">
  <colgroup>
    <col style="width: 120px">
    <col style="width: 100px">
    <col style="width: 350px">
    <col style="width: 250px">
    <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>4D 输入张量，形状为 [N, H_in, W_in, C]</td>
      <td>INT8、UINT8、INT16、UINT16、INT32、INT64、FLOAT16、FLOAT、DOUBLE</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>paddings</td>
      <td>输入</td>
      <td>空间维度 zero-padding 量，2D 张量，形状为 [2, 2]，值为 [[pad_top, pad_bottom], [pad_left, pad_right]]</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>block_size</td>
      <td>属性</td>
      <td>空间块大小，H 和 W 方向使用相同值</td>
      <td>Int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>4D 输出张量，形状为 [N * bs * bs, H_out, W_out, C]</td>
      <td>与 x 一致</td>
      <td>NHWC</td>
    </tr>
  </tbody>
</table>

## 约束说明

1. 输入张量 x 必须为 4D NHWC 格式。
2. block_size 必须大于 0。
3. paddings 形状为 [2, 2]，每个元素 >= 0。
4. H_padded = H_in + pad_top + pad_bottom 必须能被 block_size 整除。
5. W_padded = W_in + pad_left + pad_right 必须能被 block_size 整除。
6. block_size 为编译期常量（算子属性），非运行时 tensor 输入。

## 数据类型组合

| 输入 x        | 输入 paddings | 输出 y        | 组合数 |
|---------------|--------------|---------------|--------|
| 9 种 BasicType | INT32、INT64  | 与 x 一致      | 18     |

## 算子对偶关系

|                    | SpaceToBatch          | BatchToSpaceND        |
|--------------------|-----------------------|-----------------------|
| 空间预处理          | zero-padding（补零）   | crops（裁剪）          |
| block_shape 来源   | REQUIRED_ATTR Int      | INPUT tensor           |
| 数据流方向          | 空间 → 批次            | 批次 → 空间            |
| 批次维度变化        | N → N × bs × bs        | N × bs × bs → N       |
| 数据格式            | NHWC                   | ND                     |

## 实现架构

- **切分策略**：单切分源地址复用（策略 2.1），ubAxis 作为 TilingKey 模板参数
- **搬运模式**：TBuf + Ping-Pong 双缓冲，CopyIn 按输出 layout 排布 UB，CopyOut 从同一 buffer 搬出
- **padding 零值**：Duplicate 预清零 + 选择性覆盖有效区间；无 padding 的 block 跳过 Duplicate
- **LoopMode**：单切分下每个 block 处理固定 bw 值，LoopMode 迭代 w_out 区间

## 调用说明

| 调用方式  | 样例代码 | 说明 |
| :-------- | :------- | :--- |
| 图模式调用 | [test_geir_space_to_batch](./examples/test_geir_space_to_batch.cpp) | 通过[算子IR](./op_graph/space_to_batch_proto.h)构图方式调用 SpaceToBatch 算子。 |
