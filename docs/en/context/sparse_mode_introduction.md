# sparseMode Introduction

In the large model field, sparseMode usually refers to the sparsity design of parameters or activations in the model architecture or calculation formula, opposite to dense mode (DenseMode).

This section introduces common sparseMode and corresponding scenario descriptions.

| sparseMode | Meaning                                  | Note               |
| ---------- | --------------------- | ------------------ |
| 0          | defaultMask mode.                     | -    |
| 1          | allMask mode.                         | -    |
| 2          | leftUpCausal mode.                    | -    |
| 3          | rightDownCausal mode.                 | -    |
| 4          | band mode.                            | -    |
| 5          | prefix non-compressed mode.                    | varlen scenario not supported. |
| 6          | prefix compressed mode.                      | -       |
| 7          | varlen outer cut scenario, rightDownCausal mode. | Only varlen scenario supported. |
| 8          | varlen outer cut scenario, leftUpCausal mode.    | Only varlen scenario supported. |

The working principle of attenMask is to mask the value of the transpose matrix product of query (Q) and key (K) at the position where Mask is True.
<!-- as shown below:

![Schematic Diagram](../figures/QK转置图.png)

-->
The $QK^T$ matrix will be masked at the position where attenMask is True.
<!-- with the following effect:

![Schematic Diagram](../figures/遮挡QK图.png)
-->
## sparseMode=0

When sparseMode is 0, it represents defaultMask mode.

- No mask passed: If attenMask is not passed, no mask operation is performed, attenMask value is None, ignore preTokens and nextTokens values. 
<!--Masked $QK^T$ matrix schematic is as follows:

  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵.png)
-->
- nextTokens value is 0, preTokens greater than or equal to Sq, indicating causal scenario sparse, attenMask should pass in lower triangular matrix, at this time the part between preTokens and nextTokens needs to be calculated.
 <!--Masked $QK^T$ matrix schematic is as follows:

  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵1.png) 
-->
  attenMask should pass in lower triangular matrix.
  <!-- schematic as follows:
  
  ![Schematic Diagram](../figures/attenmask下三角.png)
-->
- preTokens less than Sq, nextTokens less than Skv, and both greater than or equal to 0, indicating band scenario, at this time the part between preTokens and nextTokens needs to be calculated. 
<!--Masked $QK^T$ matrix schematic is as follows:

  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵2.png)     
-->  
  attenMask should pass in band shape matrix.
  <!-- schematic as follows:

  ![Schematic Diagram](../figures/attenmask_band形状矩阵.png)
-->
- nextTokens is negative, taking preTokens=9, nextTokens=-3 as an example, the part between preTokens and nextTokens needs to be calculated. <!--Masked $QK^T$ schematic is as follows:-->

  **Note: When nextTokens is negative, preTokens value must be greater than or equal to the absolute value of nextTokens, and the absolute value of nextTokens is less than Skv.**
  <!-- 
  
  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵3.png) 
-->
- preTokens is negative, taking nextTokens=7, preTokens=-3 as an example, the part between preTokens and nextTokens needs to be calculated. <!--Masked $QK^T$ schematic is as follows:-->

  **Note: When preTokens is negative, nextTokens value must be greater than or equal to the absolute value of preTokens, and the absolute value of preTokens is less than Sq.**

  <!-- 
  
  ![Schematic Diagram](../figures/sparsemode为0遮挡矩阵4.png) 
-->
  
## sparseMode=1

When sparseMode is 1, it represents allMask, that is, passing in the complete attenMask matrix.

In this scenario, ignore nextTokens and preTokens values. <!--Masked $QK^T$ matrix schematic is as follows:-->
<!--
![Schematic Diagram](../figures/sparsemode为1遮挡矩阵.png) 
-->
## sparseMode=2

When sparseMode is 2, it represents leftUpCausal mode mask, corresponding to the lower triangular scenario divided by the upper left vertex (parameter starting point is upper left corner).

In this scenario, ignore preTokens and nextTokens values. <!--Masked $QK^T$ matrix schematic is as follows:-->
<!--
![Schematic Diagram](../figures/sparsemode为2遮挡矩阵.png)
-->
The passed attenMask is an optimized compressed lower triangular matrix (2048\*2048). <!--compressed lower triangular matrix schematic (same below):-->
<!--
![Schematic Diagram](../figures/attenmask压缩下三角.png) 
-->
## sparseMode=3

When sparseMode is 3, it represents rightDownCausal mode mask, corresponding to the lower triangular scenario divided by the lower right vertex (parameter starting point is lower right corner).

In this scenario, ignore preTokens and nextTokens values. <!--Masked $QK^T$ matrix schematic is as follows:-->
<!--
![Schematic Diagram](../figures/sparsemode为3遮挡矩阵.png)
-->
## sparseMode=4

When sparseMode is 4, it represents band scenario, that is, calculating the part between preTokens and nextTokens, parameter starting point is lower right corner, preTokens and nextTokens must have intersection. attenMask is an optimized compressed lower triangular matrix (2048\*2048). <!--Masked $QK^T$ matrix schematic is as follows:-->
<!--
![Schematic Diagram](../figures/sparsemode为4遮挡矩阵.png)
-->
## sparseMode=5

When sparseMode is 5, it represents prefix non-compressed scenario, that is, on the basis of rightDownCausal, adding a matrix with length Sq and width N on the left. The value of N is obtained from the optional input prefix. For example, the following figure shows prefix passing array [4,5] in batch=2 scenario. The N value of each batch axis can be different. Parameter starting point is upper left corner.

In this scenario, ignore preTokens and nextTokens values. <!--Masked $QK^T$ matrix schematic is as follows:-->
<!--
![Schematic Diagram](../figures/sparsemode为5遮挡矩阵.png)

attenMask should pass in matrix schematic as follows:

![Schematic Diagram](../figures/attenmask矩阵.png)
-->
## sparseMode=6

When sparseMode is 6, it represents prefix compressed scenario, that is, in prefix scenario, attenMask is an optimized compressed lower triangular + rectangular matrix (3072\*2048): where the upper part is [2048, 2048] lower triangular matrix, the lower part is [1024, 2048] rectangular matrix, the left half of the rectangular matrix is all 0, the right half is all 1. <!--attenMask should pass in matrix schematic as follows. In this scenario, ignore preTokens and nextTokens values.
-->
<!--
![Schematic Diagram](../figures/sparsemode为6遮挡矩阵.png)
-->
## sparseMode=7

When sparseMode is 7, it indicates varlen and long sequence outer cut scenario (that is, long sequence performs multi-card query sequence length cutting in the model script); users need to ensure that before outer cut it is a scenario using sparseMode 3; in current mode, users need to set preTokens and nextTokens (starting point is lower right vertex), and need to ensure parameters are correct, otherwise there will be accuracy problems.

Masked $QK^T$ matrix schematic is as follows, in the second batch query is cut, key and value are not cut, 4x6 mask matrix is cut into 2x6 and 2x6 masks, calculated on card 1 and card 2 respectively:

- The last block mask of card 1 is band type mask, configure preTokens=6 (ensure greater than or equal to the last Skv), nextTokens=-2, actual_seq_qlen should pass in {3,5}, actual_seq_kvlen should pass in {3,9}.
- The mask type of card 2 remains unchanged after cutting, sparseMode is 3, actual_seq_qlen should pass in {2,7,11}, actual_seq_kvlen should pass in {6,11,15}.

<!--
![Schematic Diagram](../figures/sparsemode为7遮挡矩阵.png)
-->
**Note**

- sparseMode=7, band represents the sparse type of the last non-empty tensor Batch; if there is only one batch, users need to configure parameters according to band mode requirements; when sparseMode=7, users need to input 2048x2048 lower triangular mask as input to this fusion operator.
- Band mode sparse parameters generated based on sparseMode=3 outer cut should meet the following conditions:
  - preTokens >= last_Skv.
  - last_Sq-last_Skv <= nextTokens <= 0.
  - Current mode does not support optional input pse.
- Non-band mode batch should satisfy: Sq <= Skv.

## sparseMode=8

When sparseMode is 8, it indicates varlen and long sequence outer cut scenario; users need to ensure that before outer cut it is a scenario using sparseMode 2; in current mode, users need to set preTokens and nextTokens (starting point is lower right vertex), and need to ensure parameters are correct, otherwise there will be accuracy problems.

Masked $QK^T$ matrix schematic is as follows, in the second batch query is cut, key and value are not cut, 5x4 mask matrix is cut into 2x4 and 3x4 masks, calculated on card 1 and card 2 respectively:

- The mask type of card 1 remains unchanged after cutting, sparseMode is 2, actual_seq_qlen should pass in {3,5}, actual_seq_kvlen should pass in {3,7}.
- The first block mask of card 2 is band type mask, configure preTokens=4 (ensure greater than or equal to the first Skv), nextTokens=1, actual_seq_qlen should pass in {3,8,12}, actual_seq_kvlen should pass in {4,9,13}.
<!--
![Schematic Diagram](../figures/sparsemode为8遮挡矩阵.png)
-->
**Note**:

- sparseMode=8, band represents the sparse type of the first non-empty tensor Batch; if there is only one batch, users need to configure parameters according to band mode requirements; when sparseMode=8, users need to input 2048x2048 lower triangular mask as input to this fusion operator.

- Band mode sparse parameters generated based on sparseMode=2 outer cut should meet the following conditions:
  - preTokens >= first_Skv.
  - nextTokens >= first_Sq - first_Skv, configure according to actual situation.
  - Current mode does not support optional input pse.
  