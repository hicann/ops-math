# LogSpaceD

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

LogSpaceD接收固定shape的辅助Tensor `assist`，按TBE单端FP32语义生成长度为`steps`的对数间隔序列。`assist`固定为rank-1 `[1]`，只承担ABI输入槽和dtype profile，其数值不参与计算。

动态能力：`DynamicRankSupportFlag=true`。建图时可声明`[-2]`（unknown rank），但运行时不放宽契约：`assist.shape=[1]`，输出rank固定为1且shape由属性推导为`[steps]`。

令`n=steps`。当`n>1`时`ratio=(end-start)/(n-1)`，否则`ratio=end-start`；所有属性转换、插值和后续数学运算均为FP32：

$$
x_i=\operatorname{FP32}(i)\times ratio+\operatorname{FP32}(start),\quad i=0,1,\ldots,steps-1.
$$

输出采用TBE三分支连续语义：

$$
y_i^{(32)}=\begin{cases}
\exp(x_i\ln(base)), & base>0,\\
\left[1-2\left(|x_i|-2\left\lfloor |x_i|/2\right\rfloor\right)\right]
\exp(x_i\ln|base|), & base<0,\\
1, & base=0\ \land\ x_i=0,\\
0, & base=0\ \land\ x_i\ne0.
\end{cases}
$$

`dtype=0/1`均为兼容no-op，输出类型始终与`assist`相同：FP16→FP16、FP32→FP32、BF16→BF16。`steps=0`时输入仍为`[1]`、输出为`[0]`；`steps=1`时全局索引为0，结果只由`start`决定。

## 参数说明

<table style="table-layout: fixed; width: 1576px">
<colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 200px">
<col style="width: 200px">
<col style="width: 170px">
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
<tr><td>assist</td><td>输入</td><td>固定rank-1 [1]载体，数值不参与计算。</td><td>FLOAT16、FLOAT、BFLOAT16</td><td>ND</td></tr>
<tr><td>y</td><td>输出</td><td>对数间隔序列，shape为[steps]；数据类型与assist相同，dtype为兼容no-op。</td><td>FLOAT16、FLOAT、BFLOAT16</td><td>ND</td></tr>
<tr><td>start</td><td>属性</td><td>指数域起点，必选。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>end</td><td>属性</td><td>指数域终点，必选。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>steps</td><td>可选属性</td><td>输出元素数，默认值为100。</td><td>INT64</td><td>-</td></tr>
<tr><td>base</td><td>可选属性</td><td>幂底数，默认值为10.0。</td><td>FLOAT</td><td>-</td></tr>
<tr><td>dtype</td><td>可选属性</td><td>历史兼容属性，仅允许0或1，默认值为1；两者均不改变输出类型。</td><td>INT64</td><td>-</td></tr>
</tbody>
</table>

## 约束说明

- 建图支持`DynamicRankSupportFlag=true`，可声明`[-2]` unknown rank；运行时`assist`必须为`[1]`，`y`必须为`[steps]`，两者shape不绑定且不原地复用。
- `assist`支持FLOAT16、FLOAT、BFLOAT16输入，输出与输入同型；与`dtype=0/1`组成六种合法组合。
- `dtype`仅允许0或1，均为兼容no-op，不改变输出类型。
- 不支持FLOAT64输入、FLOAT64输出或FLOAT64数学运算；所有组合均使用FP32主计算链，FLOAT16输出仅在末端转换一次。
- `start`、`end`、`base`必须为有限FLOAT值；负底数使用连续余数符号分支，零底数使用`x_i==0 ? 1 : 0`分支，不采用通用Pow特殊值语义。
- `steps`必须非负并且只决定输出长度；输入长度固定为1。仅支持ND格式，不支持广播、端点Tensor、`axis`或`endpoint=False`。
- 仅支持Ascend 950PR/Ascend 950DT的独立GEIR调用通路，算法Workspace为0。ACLNN组合通路的 `result` 可能是多维张量，不能据此改写独立GEIR的运行时 assist 约束。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| :--- | :--- | :--- |
| GE图模式 | - | 图IR定义参见[LogSpaceD原型](op_graph/log_space_d_proto.h)。算子原型为`LogSpaceD(assist, start, end, steps=100, base=10.0, dtype=1) -> y`。 |
