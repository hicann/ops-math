# ProdEnvMatA

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT</term>                     |     √    |
| <term>Atlas A3系列产品</term>    |    √     |
| <term>Atlas A2系列产品</term>    |    √     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                               |    ×     |
| <term>Atlas训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：prod_env_mat_a是DeepMD-kit的环境矩阵计算算子，计算原子的SeA（Smoothed Angular）环境矩阵描述子。包含两大阶段：第一阶段为邻居列表格式化，计算原子间距离并按类型排序，根据sel_a分段填充邻居列表nlist；第二阶段为环境矩阵计算，使用spline5_switch五次样条平滑函数计算4分量描述子和12分量导数，并用davg/dstd进行归一化。

- 计算公式：

对每个中心原子$i$及其邻居$j$，计算相对距离$r_{ij}=\|\mathbf{r}_j-\mathbf{r}_i\|$。若$r_{ij} \leq r_{cut\_r}$，通过五次样条平滑函数$s=\text{spline5\_switch}(r_{ij}, r_{cut\_r\_smth}, r_{cut\_r})$计算原始4分量描述子：

$$
\mathbf{d}_{ij} = s \cdot \left(\frac{1}{r_{ij}},\ \frac{r_{ij,x}}{r_{ij}^2},\ \frac{r_{ij,y}}{r_{ij}^2},\ \frac{r_{ij,z}}{r_{ij}^2}\right)
$$

归一化后的描述子和导数为：

$$
\text{descrpt}_{ij} = \frac{\mathbf{d}_{ij} - \text{davg}_{t_i}}{\text{dstd}_{t_i}}, \quad \text{descrpt\_deriv}_{ij} = \frac{\mathbf{v}_{ij}}{\text{dstd}_{t_i}}
$$

其中$t_i$为中心原子$i$的类型，$\text{davg}_{t_i}$和$\text{dstd}_{t_i}$为该类型对应的均值和标准差，$\mathbf{v}_{ij}$为描述子对中心原子坐标的导数。若$r_{ij} > r_{cut\_r}$，$s=0$，描述子取$-\text{davg}/\text{dstd}$。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>coord</td>
      <td>输入</td>
      <td>所有原子的三维坐标，shape为(nsample, nall*3)，维度必须为2。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>type</td>
      <td>输入</td>
      <td>所有原子的类型索引，shape为(nsample, nall)，维度必须为2，shape[0]与coord一致。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>natoms</td>
      <td>输入</td>
      <td>原子计数信息，shape为(2+ntypes,)，维度必须为1，shape[0]≥3。值为[nloc, nall, ntype_0_count, ...]，infershape阶段读取natoms[0]获取nloc。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>box</td>
      <td>输入</td>
      <td>周期性边界框（3×3展平），shape为(nsample, 9)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mesh</td>
      <td>输入</td>
      <td>网格信息（用于复制邻居），shape为(nsample, 6)。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>davg</td>
      <td>输入</td>
      <td>描述子均值（用于归一化），shape为(ntypes, nnei*4)，维度必须为2。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dstd</td>
      <td>输入</td>
      <td>描述子标准差（用于归一化），shape为(ntypes, nnei*4)，维度必须为2，dtype与coord/davg一致，元素不能为0。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>descrpt</td>
      <td>输出</td>
      <td>归一化后的环境矩阵描述子，shape为(nsample, nloc*nnei*4)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>descrpt_deriv</td>
      <td>输出</td>
      <td>描述子导数，shape为(nsample, nloc*nnei*12)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rij</td>
      <td>输出</td>
      <td>中心原子到邻居的相对坐标，shape为(nsample, nloc*nnei*3)。</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>nlist</td>
      <td>输出</td>
      <td>格式化后的邻居列表索引，shape为(nsample, nloc*nnei)，元素值≥-1（-1表示无邻居）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rcut_a</td>
      <td>属性</td>
      <td>角度截断半径（当前未使用，保留兼容），≥0。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r</td>
      <td>属性</td>
      <td>径向截断半径，>0且≥rcut_r_smth。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rcut_r_smth</td>
      <td>属性</td>
      <td>平滑起始半径，>0且≤rcut_r。</td>
      <td>Float</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_a</td>
      <td>属性</td>
      <td>每种类型的角向邻居选取数量，非空列表，元素≥0。nnei=sum(sel_a)。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sel_r</td>
      <td>属性</td>
      <td>每种类型的径向邻居选取数量（本算子不使用），元素≥0。</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- coord为2维Tensor，shape为(nsample, nall*3)，dtype为float32。
- type为2维Tensor，shape为(nsample, nall)，dtype为int32，shape[0]与coord.shape[0]一致，元素值≥0（type<0的虚拟原子跳过）。
- natoms为1维Tensor，shape为(2+ntypes,)，dtype为int32，shape[0]≥3。natoms[0]=nloc（本地原子数），natoms[1]=nall（总原子数），后续元素为各类型原子计数。
- box为2维Tensor，shape为(nsample, 9)，dtype为float32。
- mesh为2维Tensor，shape为(nsample, 6)，dtype为int32。
- davg为2维Tensor，shape为(ntypes, nnei*4)，dtype为float32，与coord/dstd的dtype一致。
- dstd为2维Tensor，shape为(ntypes, nnei*4)，dtype为float32，与coord/davg的dtype一致，元素不能为0（除法分母）。
- rcut_a必须≥0。
- rcut_r必须>0且≥rcut_r_smth。
- rcut_r_smth必须>0且≤rcut_r。
- sel_a必须为非空列表，所有元素≥0。nnei=sum(sel_a)决定输出shape的第二维。
- sel_r元素必须≥0（本算子不使用sel_r，保留兼容）。
- natoms[0]（nloc）决定输出shape的第二维，infershape阶段通过值依赖读取natoms输入。
- nlist输出元素值≥-1，-1表示该位置无邻居（未填满时padding为-1）。
- 输入Tensor需为连续内存布局（contiguous）：即数据在内存中按行主序紧密排列，不含stride跳步或间隙（概念详见[非连续的Tensor](../../docs/zh/context/non_contiguous_tensor.md)）。若传入非连续Tensor（如转置、切片得到的视图），框架会自动连续化处理，产生额外拷贝开销。
- 算子仅支持2D输入，不支持标量、1D或8D场景。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_prod_env_mat_a.cpp">test_geir_prod_env_mat_a</a></td>
    <td>参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
</tbody>
</table>
