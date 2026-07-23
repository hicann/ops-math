# Quantization Introduction

Quantization is widely used in deep learning models, especially during inference. Through quantization, models can run more efficiently on hardware, reducing computational resource consumption and accelerating inference process, while lowering model storage requirements.

CANN operator quantization refers to the computation process of converting input Tensor of matrix (cube) operators such as Matmul in neural networks from high bit to low bit, while generating corresponding quantization parameter scale. When low bit cube computation is completed, can convert low bit values back to high bit values through quantization parameter scale, thereby ensuring correctness of overall computation result (effect is approximately equivalent to direct high bit computation), and effectively improving computation efficiency.

- Static quantization: Uses pre-determined quantization parameters for quantization. In inference scenarios, quantization of weight generally uses static quantization, quantization operator performance will be better.
- Dynamic quantization: Uses input data to compute quantization parameters online for quantization. In inference scenarios, quantization of activation generally uses dynamic quantization, which can better adapt to data changes with higher precision; in training scenarios, to improve quantization precision, dynamic quantization is also generally used. Note that dynamic quantization, because quantization parameters are generated online, quantization operator performance will be slightly worse.

## Quantization Mode

Quantization mode (also called quantization granularity) refers to using different quantization computation levels for different input Tensors of operator. Common quantization computation modes include:

>Note:
>
>- m, n, k variables respectively represent different axis sizes of Tensor computation.
>- Left matrix, right matrix respectively refer to two input Tensors used for matrix multiplication computation in cube operator. Generally left matrix represents activation, right matrix represents weight. Please understand and use according to actual situation.

- pertensor quantization (abbreviated as T quantization): Quantization object can be either left matrix or right matrix, each Tensor shares one same quantization parameter.

  Assuming left matrix shape is (m, k), right matrix shape is (k, n), k is reduce axis, generated quantization parameter shape is (1, ).
<!--
  ![Schematic Diagram](../figures/pertensor量化.png)
-->
- perchannel quantization (abbreviated as C quantization): Quantization object is right matrix, each channel uses independent quantization parameter respectively.

  Assuming right matrix shape is (k, n), k is reduce axis, generated quantization parameter shape is (n, ).
<!--
  ![Schematic Diagram](../figures/perchannel量化.png)
-->
- pertoken quantization (abbreviated as K quantization): Quantization object is left matrix, each token uses independent quantization parameter respectively.

  Assuming left matrix shape is (m, k), k is reduce axis, generated quantization parameter shape is (m, ).
<!--
  ![Schematic Diagram](../figures/pertoken量化.png)
-->
- pergroup quantization (abbreviated as G quantization): Quantization object can be either left matrix or right matrix, groups data on reduce axis, each group uses independent quantization parameter.
  - Assuming left matrix shape is (m, k), k is reduce axis, groups on k axis, group size is gs, generated quantization parameter shape is (m, k/gs).
  - Assuming right matrix shape is (k, n), k is reduce axis, groups on k axis, group size is gs, generated quantization parameter shape is (k/gs, n).
<!--
  ![Schematic Diagram](../figures/pergroup量化.png)
-->
- perblock quantization (abbreviated as B quantization): Quantization object can be either left matrix or right matrix, blocks data on all axes, each block uses independent quantization parameter.

  - Assuming left matrix shape is (m, k), k is reduce axis, groups data on m, k axes respectively by (bs, bs) blocks, bs is block size, generated quantization parameter shape is (m/bs, k/bs).
  - Assuming right matrix shape is (k, n), k is reduce axis, groups data on k, n axes respectively by (bs, bs) blocks, bs is block size, generated quantization parameter shape is (k/bs, n/bs).
<!--
  ![Schematic Diagram](../figures/perblock量化.png)
--> 

## Common Combined Quantization

- Full quantization: Generally refers to mode that quantizes both left and right matrices, including
  - pertensor-perchannel quantization mode (abbreviated as T-C quantization mode)
  - pertoken-perchannel quantization mode (abbreviated as K-C quantization mode)
  - pergroup-perblock quantization mode (abbreviated as G-B quantization mode)
  - pertensor-perchannel-pergroup quantization mode (abbreviated as T-CG quantization mode)
  - perblock-perblock quantization mode (abbreviated as B-B quantization mode)
- Pseudo quantization: Generally refers to mode that quantizes weight matrix, including perchannel quantization mode (abbreviated as C quantization mode).
- MX quantization: Essentially Microscaling quantization, maintains model precision at extremely low bits (such as 1bit) through dynamically adjusting scaling factors. Here refers to pergroup-pergroup quantization mode (abbreviated as G-G quantization mode), which is a special case where quantization parameter type is FLOAT8_E8M0 and group size is 32.
